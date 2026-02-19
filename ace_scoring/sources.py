"""Source preparation for the ACE scorer.

Handles three modes:
1. From tool_history (RL training / eval with tools)
2. From response URLs via Jina Reader (replaces Firecrawl)
3. Build source text for Stage 2 grounding prompts
"""

import re

import httpx

JINA_READER_BASE = "https://r.jina.ai"


def sources_from_tool_history(tool_history: dict) -> list[dict]:
    """Build autograder-compatible sources from tool call history.

    Browsed pages get full markdown content. Unbrowsed search results
    get snippet-only entries.
    """
    sources = []
    seen_urls: set[str] = set()

    for page in tool_history.get("browsed_pages", []):
        url = page.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append({
            "source_number": len(sources) + 1,
            "source_title": page.get("title", url),
            "source_link": url,
            "relevant_text": [],
            "webpage_content": {
                "title": page.get("title", ""),
                "url": url,
                "markdown": page.get("markdown", ""),
                "text": "",
                "error": None,
                "source_type": "webpage",
            },
        })

    for search in tool_history.get("searches", []):
        for result in search.get("results", {}).get("organic_results", []):
            url = result.get("link", "")
            if url in seen_urls or not url:
                continue
            seen_urls.add(url)
            sources.append({
                "source_number": len(sources) + 1,
                "source_title": result.get("title", url),
                "source_link": url,
                "relevant_text": [],
                "webpage_content": {
                    "title": result.get("title", ""),
                    "url": url,
                    "markdown": result.get("snippet", ""),
                    "text": result.get("snippet", ""),
                    "error": None,
                    "source_type": "webpage",
                },
            })

    return sources


def enrich_snippet_sources(
    sources: list[dict],
    min_content_chars: int = 500,
    max_chars_per_source: int = 30000,
    max_sources_to_enrich: int = 10,
) -> list[dict]:
    """Scrape full page content for sources that only have snippets.

    Sources from search results only have ~150-char snippets. This fetches
    actual page content via Jina Reader so Stage 2 grounding checks are
    meaningful. Modifies sources in-place and returns the list.
    """
    enriched = 0
    for src in sources:
        if enriched >= max_sources_to_enrich:
            break
        wc = src.get("webpage_content", {})
        markdown = wc.get("markdown", "")
        if len(markdown) >= min_content_chars:
            continue  # already has real content

        url = src.get("source_link", "")
        if not url:
            continue

        try:
            resp = httpx.get(
                f"{JINA_READER_BASE}/{url}",
                headers={"Accept": "text/markdown"},
                timeout=15.0,
            )
            resp.raise_for_status()
            page_markdown = resp.text[:max_chars_per_source]

            title = url
            for line in page_markdown.split("\n"):
                if line.strip().startswith("# "):
                    title = line.strip()[2:].strip()
                    break

            wc["markdown"] = page_markdown
            wc["title"] = title
            src["source_title"] = title
            enriched += 1
        except Exception:
            pass  # keep original snippet if fetch fails

    return sources


def sources_from_response_urls(
    response_text: str,
    max_sources: int = 10,
    max_chars_per_source: int = 30000,
) -> list[dict]:
    """Scrape URLs found in response text via Jina Reader.

    Replaces Firecrawl for cases without tool_history.
    """
    urls = re.findall(r'https?://[^\s)<>\[\]"\']+', response_text)
    urls = list(dict.fromkeys(urls))[:max_sources]

    sources = []
    for url in urls:
        try:
            resp = httpx.get(
                f"{JINA_READER_BASE}/{url}",
                headers={"Accept": "text/markdown"},
                timeout=20.0,
            )
            resp.raise_for_status()
            markdown = resp.text[:max_chars_per_source]

            title = url
            for line in markdown.split("\n"):
                if line.strip().startswith("# "):
                    title = line.strip()[2:].strip()
                    break

            sources.append({
                "source_number": len(sources) + 1,
                "source_title": title,
                "source_link": url,
                "relevant_text": [],
                "webpage_content": {
                    "title": title,
                    "url": url,
                    "markdown": markdown,
                    "text": "",
                    "error": None,
                    "source_type": "webpage",
                },
            })
        except Exception as e:
            sources.append({
                "source_number": len(sources) + 1,
                "source_title": url,
                "source_link": url,
                "relevant_text": [],
                "webpage_content": {
                    "title": "",
                    "url": url,
                    "markdown": "",
                    "text": "",
                    "error": str(e),
                    "source_type": "webpage",
                },
            })

    return sources


def build_source_text_for_grounding(
    sources: list[dict],
    product_source_map: list[dict] | None = None,
    product_name: str | None = None,
    max_total_chars: int = 40000,
) -> str:
    """Build concatenated source text for Stage 2 grounding prompts.

    When product_source_map + product_name are provided, only includes
    sources mapped to that product (autograder-style per-product checking).
    Otherwise includes all sources.
    """
    if product_source_map and product_name:
        mapping = next(
            (m for m in product_source_map if m["product_name"] == product_name),
            None,
        )
        if mapping:
            indices = set(mapping.get("source_indices", []))
            filtered = [s for s in sources if (s["source_number"] - 1) in indices]
        else:
            filtered = sources
    else:
        filtered = sources

    # Truncate per-source to avoid losing later sources entirely.
    # Budget ~equal chars per source, with a minimum of 2K each.
    n_sources = len(filtered)
    if n_sources == 0:
        return ""
    per_source_budget = max(2000, max_total_chars // n_sources)

    parts = []
    for src in filtered:
        title = src.get("source_title", "")
        wc = src.get("webpage_content", {})
        markdown = wc.get("markdown", "")
        if not markdown:
            continue
        if len(markdown) > per_source_budget:
            markdown = markdown[:per_source_budget] + "\n[truncated]"
        parts.append(f"Source: {title}\n{markdown}")

    return "\n---\n".join(parts)

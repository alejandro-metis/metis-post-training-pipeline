"""Source preparation for the ACE scorer.

Handles three modes:
1. From tool_history (RL training / eval with tools)
2. From response URLs via Jina Reader (for models without tool use)
3. Build source text for Stage 2 grounding prompts
"""

import re

from ace_scoring.jina import extract_title, fetch_page


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
        sources.append(
            _make_source(
                len(sources) + 1,
                page.get("title", url),
                url,
                markdown=page.get("markdown", ""),
            )
        )

    for search in tool_history.get("searches", []):
        for result in search.get("results", {}).get("organic_results", []):
            url = result.get("link", "")
            if url in seen_urls or not url:
                continue
            seen_urls.add(url)
            sources.append(
                _make_source(
                    len(sources) + 1,
                    result.get("title", url),
                    url,
                    markdown=result.get("snippet", ""),
                    text=result.get("snippet", ""),
                )
            )

    return sources


def sources_from_annotations(annotations: list[dict], response_text: str) -> list[dict]:
    """Build autograder-compatible sources from Responses API URL citations.

    Each unique URL becomes a source with empty markdown (to be enriched
    via Jina). Citation text (derived from response_text character indices)
    stored as relevant_text.
    """
    sources: list[dict] = []
    seen_urls: dict[str, int] = {}  # url -> index in sources list

    for ann in annotations:
        url = ann.get("url", "")
        if not url:
            continue
        if url not in seen_urls:
            seen_urls[url] = len(sources)
            sources.append(
                _make_source(
                    len(sources) + 1,
                    ann.get("title", url),
                    url,
                    markdown="",  # enrich_snippet_sources will fetch via Jina
                )
            )
        # Derive cited text from character indices in response_text
        # (SDK annotation has start_index/end_index but no cited_text field)
        start = ann.get("start_index", 0)
        end = ann.get("end_index", 0)
        if 0 <= start < end <= len(response_text):
            cited_text = response_text[start:end]
            sources[seen_urls[url]]["relevant_text"].append(cited_text)

    return sources


def enrich_snippet_sources(
    sources: list[dict],
    min_content_chars: int = 500,
    max_chars_per_source: int = 30000,
    max_sources_to_enrich: int = 10,
) -> tuple[list[dict], bool]:
    """Scrape full page content for sources that only have snippets.

    Sources from search results only have ~150-char snippets. This fetches
    actual page content via Jina Reader so Stage 2 grounding checks are
    meaningful. Modifies sources in-place.

    Returns (sources, had_transient_failures).
    """
    enriched = 0
    had_transient_failures = False
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

        # Skip URLs that permanently failed in a previous rescore run
        if wc.get("enrichment_error"):
            continue

        page_markdown, err = fetch_page(url, max_chars=max_chars_per_source)
        if err:
            print(f"[ace_scoring] Failed to enrich {url}: {err}")
            # Mark non-transient failures so future rescores skip them.
            # Timeouts, connection errors, and rate limits are transient.
            transient = err.startswith("Connection failed:") or err in (
                "HTTP 429",
                "HTTP 502",
                "HTTP 503",
                "HTTP 504",
            )
            if not transient:
                wc["enrichment_error"] = err
            else:
                had_transient_failures = True
            continue

        title = extract_title(page_markdown, fallback=url)
        wc["markdown"] = page_markdown
        wc["title"] = title
        src["source_title"] = title
        enriched += 1

    return sources, had_transient_failures


def sources_from_response_urls(
    response_text: str,
    max_sources: int = 10,
    max_chars_per_source: int = 30000,
) -> list[dict]:
    """Scrape URLs found in response text via Jina Reader.

    For scoring models that don't use tools but embed URLs directly
    in their responses (e.g., GPT zeroshot). Not currently called in
    the pipeline but kept for this use case.
    """
    urls = re.findall(r'https?://[^\s)<>\[\]"\']+', response_text)
    urls = list(dict.fromkeys(urls))[:max_sources]

    sources = []
    for url in urls:
        markdown, err = fetch_page(url, timeout=20.0, max_chars=max_chars_per_source)
        if err:
            print(f"[ace_scoring] Failed to fetch {url}: {err}")
            sources.append(_make_source(len(sources) + 1, url, url, error=err))
        else:
            title = extract_title(markdown, fallback=url)
            sources.append(
                _make_source(
                    len(sources) + 1,
                    title,
                    url,
                    markdown=markdown,
                )
            )

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    number: int,
    title: str,
    url: str,
    *,
    markdown: str = "",
    text: str = "",
    error: str | None = None,
) -> dict:
    """Create an autograder-compatible source dict."""
    return {
        "source_number": number,
        "source_title": title,
        "source_link": url,
        "relevant_text": [],
        "webpage_content": {
            "title": title,
            "url": url,
            "markdown": markdown,
            "text": text,
            "error": error,
            "source_type": "webpage",
        },
    }

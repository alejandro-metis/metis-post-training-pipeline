"""Jina Reader utilities for fetching and parsing web pages.

Single source of truth for JINA_READER_BASE URL, page fetching, and
markdown title extraction. Used by sources.py, scorer.py, ace_eval.py,
and ace_search_tool.py.
"""

import os

import httpx

JINA_READER_BASE = "https://r.jina.ai"


def fetch_page(
    url: str, timeout: float = 15.0, max_chars: int = 30000
) -> tuple[str, str]:
    """Fetch page content via Jina Reader.

    Returns (markdown, error). On success error is empty string.
    On failure markdown is empty and error describes what went wrong.
    Uses JINA_API_KEY env var for authenticated requests (higher rate limits).
    """
    try:
        headers = {"Accept": "text/markdown"}
        api_key = os.environ.get("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = httpx.get(
            f"{JINA_READER_BASE}/{url}",
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        markdown = resp.text[:max_chars]
        if not markdown or len(markdown) < 100:
            return "", "Page has no content or failed to load"
        return markdown, ""
    except httpx.HTTPStatusError as e:
        return "", f"HTTP {e.response.status_code}"
    except Exception as e:
        return "", f"Connection failed: {e}"


def extract_title(markdown: str, fallback: str = "") -> str:
    """Extract first H1 heading from markdown content."""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback

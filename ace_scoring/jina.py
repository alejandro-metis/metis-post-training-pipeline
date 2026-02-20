"""Jina Reader utilities for fetching and parsing web pages.

Single source of truth for JINA_READER_BASE URL, page fetching, and
markdown title extraction. Used by sources.py, scorer.py, ace_eval.py,
and ace_search_tool.py.
"""

import os
import time

import httpx

JINA_READER_BASE = "https://r.jina.ai"


def fetch_page(
    url: str, timeout: float = 30.0, max_chars: int = 30000, retries: int = 2
) -> tuple[str, str]:
    """Fetch page content via Jina Reader.

    Returns (markdown, error). On success error is empty string.
    On failure markdown is empty and error describes what went wrong.
    Uses JINA_API_KEY env var for authenticated requests (higher rate limits).
    Retries on timeouts and 5xx errors.
    """
    headers = {"Accept": "text/markdown"}
    api_key = os.environ.get("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
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
            if e.response.status_code >= 500 and attempt < retries:
                last_error = f"HTTP {e.response.status_code}"
                time.sleep(2 * attempt)
                continue
            return "", f"HTTP {e.response.status_code}"
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = f"Connection failed: {e}"
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
        except Exception as e:
            return "", f"Connection failed: {e}"

    return "", last_error


def extract_title(markdown: str, fallback: str = "") -> str:
    """Extract first H1 heading from markdown content."""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback

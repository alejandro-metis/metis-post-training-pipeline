"""
Web search and page browsing tools for ACE tasks.

Two verl BaseTool subclasses:
  - AceSearchTool: Search via SearchAPI.io (Google)
  - AceBrowseTool: Read page content via Jina Reader

Both tools share per-trajectory state so the grounding wrapper
and reward function can access the full tool call history.

Requires SEARCHAPI_IO_KEY env var (set via Modal secret).
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

import httpx

from ace_scoring.jina import JINA_READER_BASE, extract_title
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)

SEARCHAPI_BASE = "https://www.searchapi.io/api/v1/search"
MAX_RESULTS = 5
MAX_SEARCHES_PER_ROLLOUT = 5
MAX_BROWSES_PER_ROLLOUT = 5

# ---------------------------------------------------------------------------
# Shared state manager — both tools read/write per-trajectory state
# ---------------------------------------------------------------------------

_trajectory_state: dict[str, dict] = {}


def get_trajectory_state(instance_id: str) -> dict:
    """Get or create shared state for a trajectory."""
    if instance_id not in _trajectory_state:
        _trajectory_state[instance_id] = {
            "searches": [],  # [{query, results}]
            "browsed_pages": [],  # [{url, title, markdown}]
            "search_count": 0,
            "browse_count": 0,
        }
    return _trajectory_state[instance_id]


def release_trajectory_state(instance_id: str) -> None:
    """Clean up shared state for a trajectory."""
    _trajectory_state.pop(instance_id, None)


def get_tool_history(instance_id: str) -> dict:
    """Get full tool history for a trajectory (used by grounding wrapper / reward)."""
    state = _trajectory_state.get(instance_id, {})
    return {
        "searches": state.get("searches", []),
        "browsed_pages": state.get("browsed_pages", []),
    }


# ---------------------------------------------------------------------------
# AceSearchTool — web search via SearchAPI.io
# ---------------------------------------------------------------------------


class AceSearchTool(BaseTool):
    """Web search tool for ACE tasks using SearchAPI.io."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._api_key = config.get("api_key") or os.environ.get("SEARCHAPI_IO_KEY", "")
        self._max_results = config.get("max_results", MAX_RESULTS)
        self._max_searches = config.get(
            "max_searches_per_rollout", MAX_SEARCHES_PER_ROLLOUT
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        get_trajectory_state(instance_id)  # init shared state
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        state = get_trajectory_state(instance_id)

        if state["search_count"] >= self._max_searches:
            return (
                ToolResponse(
                    text="Search limit reached. Use the information you already have."
                ),
                0.0,
                {"search_limited": True},
            )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    SEARCHAPI_BASE,
                    params={
                        "q": query,
                        "engine": "google",
                        "api_key": self._api_key,
                        "num": self._max_results,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results_text = self._format_results(data)

            state["searches"].append({"query": query, "results": data})
            state["search_count"] += 1

            return (
                ToolResponse(text=results_text),
                0.0,
                {"query": query, "num_results": len(data.get("organic_results", []))},
            )

        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            return (
                ToolResponse(text=f"Search failed: {e}"),
                0.0,
                {"error": str(e)},
            )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        release_trajectory_state(instance_id)

    @staticmethod
    def _format_results(data: dict) -> str:
        organic = data.get("organic_results", [])
        if not organic:
            return "No results found."

        parts = []
        for i, result in enumerate(organic, 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            parts.append(f"{i}. {title}\n   {snippet}\n   URL: {link}")

        return "Search Results:\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# AceBrowseTool — page browsing via Jina Reader
# ---------------------------------------------------------------------------


class AceBrowseTool(BaseTool):
    """Page browsing tool for ACE tasks using Jina Reader.

    Fetches full page content as markdown from any URL.
    Use after searching to get detailed product info, prices, and links.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._max_browses = config.get(
            "max_browses_per_rollout", MAX_BROWSES_PER_ROLLOUT
        )
        self._timeout = config.get("timeout", 20.0)
        self._max_content_chars = config.get("max_content_chars", 8000)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        get_trajectory_state(instance_id)  # init shared state
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        url = parameters.get("url", "")
        state = get_trajectory_state(instance_id)

        if state["browse_count"] >= self._max_browses:
            return (
                ToolResponse(
                    text="Browse limit reached. Use the information you already have."
                ),
                0.0,
                {"browse_limited": True},
            )

        try:
            jina_url = f"{JINA_READER_BASE}/{url}"
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    jina_url,
                    headers={"Accept": "text/markdown"},
                )
                resp.raise_for_status()
                markdown = resp.text

            title = extract_title(markdown, fallback=url)

            # Truncate to avoid blowing up context
            truncated = markdown[: self._max_content_chars]
            if len(markdown) > self._max_content_chars:
                truncated += "\n\n[Content truncated]"

            # Store full content for grounding wrapper (not truncated)
            state["browsed_pages"].append(
                {
                    "url": url,
                    "title": title,
                    "markdown": markdown,
                }
            )
            state["browse_count"] += 1

            return (
                ToolResponse(text=f"Page content from {url}:\n\n{truncated}"),
                0.0,
                {"url": url, "content_length": len(markdown)},
            )

        except Exception as e:
            logger.warning(f"Browse failed for URL '{url}': {e}")
            return (
                ToolResponse(text=f"Failed to load page: {e}"),
                0.0,
                {"error": str(e)},
            )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # Don't release shared state here — AceSearchTool.release handles it
        # (only one tool should own cleanup)
        pass

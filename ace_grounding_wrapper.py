"""Convert tool call history into ACE autograder-compatible JSON.

Thin wrapper around ace_scoring.sources and ace_scoring.product.
Produces the 2_scraped_sources.json format expected by the ACE autograder.
"""

from datetime import datetime

from ace_scoring.product import extract_products, map_products_to_sources
from ace_scoring.sources import sources_from_tool_history


def build_autograder_input(
    task_id: int | str,
    query: str,
    response_text: str,
    criteria: list[dict],
    tool_history: dict,
    domain: str = "Shopping",
    shop_vs_product: str | None = None,
) -> dict:
    """Convert tool history into autograder-compatible JSON (2_scraped_sources.json)."""
    sources = sources_from_tool_history(tool_history)
    product_names = extract_products(response_text, query)
    product_source_map = map_products_to_sources(product_names, sources)
    formatted_criteria = _format_criteria(criteria)

    result = {
        "task_id": task_id,
        "query": query,
        "responseText": response_text,
        "provider": "custom",
        "productSourceMap": product_source_map,
        "criteria": formatted_criteria,
        "sources": sources,
        "failed_grounded_sites": [],
        "metadata": {
            "total_sources": len(sources),
            "scraped_at": datetime.now().isoformat(),
            "failed_scrapes": 0,
        },
        "pipeline_timing": {
            "total_seconds": 0.0,
            "scraping_seconds": 0.0,
            "processing_seconds": 0.0,
        },
    }

    if shop_vs_product and domain == "Shopping":
        result["shop_vs_product"] = shop_vs_product

    return result


def _format_criteria(criteria: list[dict]) -> list[dict]:
    """Format criteria from parquet schema to autograder schema.

    Parquet fields -> Autograder fields:
        criterion_id    -> criterion_id
        description     -> description
        criteria_type   -> type
        hurdle_tag      -> hurdle_tag
        grounding_check -> grounded_status
    """
    formatted = []
    for i, c in enumerate(criteria):
        formatted.append({
            "criterion_id": c.get("criterion_id", str(i + 1)),
            "id": i + 1,
            "description": c["description"],
            "type": c.get("criteria_type", "standard"),
            "hurdle_tag": c.get("hurdle_tag", "Not"),
            "grounded_status": c.get("grounding_check", "Not Grounded"),
        })
    return formatted

"""Product/recommendation extraction and source mapping.

Single implementation replacing duplicated logic in:
    - ace_grounding_wrapper._extract_product_names()
    - ace_reward._extract_products()
    - apex-evals grounding-pipeline.extract_recommendations()
"""

import json

from ace_scoring import prompts
from ace_scoring.llm import call_judge, parse_json


def extract_products(response_text: str, query: str) -> list[str]:
    """Extract product/recommendation names from model response."""
    prompt = prompts.EXTRACT_PRODUCTS.format(query=query, response=response_text)
    try:
        text = call_judge(prompt)
        names = parse_json(text)
        return [str(n) for n in names] if isinstance(names, list) else []
    except Exception as e:
        print(f"[ace_scoring] Product extraction error: {e}")
        return []


def map_products_to_sources(
    product_names: list[str],
    sources: list[dict],
) -> list[dict]:
    """Map product names to their supporting source indices.

    Returns list of {"product_name": str, "source_indices": [int, ...]}.
    """
    if not product_names or not sources:
        return [{"product_name": name, "source_indices": []} for name in product_names]

    source_descriptions = []
    for i, src in enumerate(sources):
        title = src.get("source_title", "")
        link = src.get("source_link", "")
        content_preview = src.get("webpage_content", {}).get("markdown", "")[:200]
        source_descriptions.append(f"  Source {i}: [{title}] {link}\n    {content_preview}")

    prompt = prompts.MAP_PRODUCTS_TO_SOURCES.format(
        product_names=json.dumps(product_names),
        sources_text="\n".join(source_descriptions),
    )
    try:
        text = call_judge(prompt)
        mapping = parse_json(text)
        return mapping if isinstance(mapping, list) else []
    except Exception as e:
        print(f"[ace_scoring] Product-source mapping error: {e}")
        return [{"product_name": name, "source_indices": []} for name in product_names]

"""ACE scoring engine.

Single implementation of the ACE hierarchical grading methodology.
Used by both eval pipeline and RL reward function.

Per criterion (Paper Figure 4):
    Route 1: "Provides link(s)"        -> link verification (0 or +1)
    Route 2: Non-grounded              -> Stage 1 only (0 or +1)
    Route 3: Grounded                  -> Stage 1 + Stage 2 (0, +1, or -1)

Task aggregation:
    total_score = sum(criterion_scores)                    (raw, for leaderboard)
    total_hurdle_score = total_score if all hurdles pass    (gated)
    RL reward = total_score / num_criteria                  (normalized)

References:
    apex-evals/ace/harness/autograder.py
    ACE paper Section 4, Figure 4
"""

import json
import re

from ace_scoring import prompts
from ace_scoring.jina import fetch_page
from ace_scoring.llm import call_judge, parse_json
from ace_scoring.sources import build_source_text_for_grounding
from ace_scoring.types import CriterionResult, Stage1Result, Stage2Result, TaskResult

# Shared URL regex — excludes trailing ), ], ", ' that often surround URLs in text
_URL_RE = re.compile(r'https?://[^\s)<>\[\]"\']+')


def _is_grounded(criterion: dict) -> bool:
    """Check if criterion requires grounding verification."""
    val = criterion.get(
        "grounding_check", criterion.get("grounded_status", "Not Grounded")
    )
    return val in ("Grounded", "Check required")


def _get_ctype(criterion: dict) -> str:
    return criterion.get("criteria_type", criterion.get("type", ""))


def _get_cid(criterion: dict) -> str:
    return criterion.get("criterion_id", str(criterion.get("id", "")))


def _get_hurdle(criterion: dict) -> str:
    return criterion.get("hurdle_tag", "Not")


def _get_grounding(criterion: dict) -> str:
    return criterion.get(
        "grounding_check", criterion.get("grounded_status", "Not Grounded")
    )


# ---------------------------------------------------------------------------
# Route 1: Link criteria
# ---------------------------------------------------------------------------


def _verify_link_with_llm(
    product_name: str,
    url: str,
    page_content: str,
    criterion_description: str,
    domain: str = "shopping",
    shop_vs_product: str = "Product",
) -> dict:
    """Use LLM to classify whether a scraped page is a valid link for the product.

    Matches the official ACE autograder approach: Firecrawl+Gemini → Jina+GPT.
    """
    domain_lower = domain.lower()

    if domain_lower == "gaming":
        prompt = prompts.VERIFY_GAMING_LINK.format(
            product_name=product_name,
            url=url,
            criterion_description=criterion_description,
            page_content=page_content[:10000],
        )
    else:
        # Shopping / default: purchase page verification
        if shop_vs_product == "Shop":
            validation_criteria = f"""VALID store/vendor page:
- Direct link to the vendor's store or product category page
- Shows products from {product_name} (the vendor/brand)
- User can browse and potentially purchase from this vendor
- Includes vendor branding, product listings, or shop page

INVALID pages:
- Unrelated vendor or store
- Generic marketplace search results (unless {product_name} is the seller)
- Review/comparison pages without vendor link
- Dead links or error pages"""
            mode_text = "store/vendor"
        else:
            validation_criteria = f"""VALID product purchasing page:
- Direct product page for {product_name}
- Has "Add to Cart", "Buy Now", or similar purchase button
- User can directly purchase THIS specific product

INVALID pages:
- Product list or search results (even if they link to purchase pages)
- Review/comparison pages (without direct purchase)
- General shop homepage
- Category pages with multiple products"""
            mode_text = "product purchasing"

        prompt = prompts.VERIFY_PURCHASE_LINK.format(
            product_name=product_name,
            page_content=page_content[:10000],
            validation_criteria=validation_criteria,
            mode_text=mode_text,
        )

    try:
        text = call_judge(prompt)
        data = parse_json(text)
        return {
            "is_valid": data.get("is_valid", data.get("is_purchase_page", False)),
            "page_type": data.get("page_type", "unknown"),
            "reason": data.get("reason", ""),
        }
    except Exception as e:
        return {"is_valid": False, "page_type": "error", "reason": f"LLM error: {e}"}


def grade_link_criterion(
    criterion: dict,
    response_text: str,
    products: list[str],
    domain: str = "shopping",
    shop_vs_product: str = "Product",
) -> CriterionResult:
    """Grade "Provides link(s)" criterion. Score -1 or +1.

    Matches official ACE autograder: extract links per product, scrape each URL,
    then LLM-classify whether the page is a valid purchase/product page.
    """
    cid = _get_cid(criterion)
    ctype = _get_ctype(criterion)
    hurdle = _get_hurdle(criterion)
    grounding = _get_grounding(criterion)

    if not products:
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=-1,
            stage_reached="link_verification",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=Stage1Result(
                all_pass=False, reasoning="No recommendations identified"
            ),
            reasoning="Failed: No recommendations identified",
        )

    urls_in_response = _URL_RE.findall(response_text)
    if not urls_in_response:
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=-1,
            stage_reached="link_verification",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=Stage1Result(
                all_pass=False, reasoning="No URLs found in response"
            ),
            reasoning="Failed: No links in response",
        )

    # Stage 1: Extract links per product via LLM
    extract_prompt = prompts.EXTRACT_LINKS_FOR_PRODUCT.format(
        products=json.dumps(products),
        response=response_text,
    )
    try:
        text = call_judge(extract_prompt)
        product_links = parse_json(text)
    except Exception as e:
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=-1,
            stage_reached="link_verification",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=Stage1Result(
                all_pass=False, reasoning=f"Link extraction error: {e}"
            ),
            reasoning=f"Error: {e}",
        )

    # Check all products have links
    products_missing_links = [p for p in products if not product_links.get(p)]
    if products_missing_links:
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=-1,
            stage_reached="link_verification",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=Stage1Result(
                all_pass=False,
                reasoning=f"Missing links for: {', '.join(products_missing_links)}",
            ),
            reasoning=f"Failed: Products missing links: {', '.join(products_missing_links)}",
        )

    # Stage 2: Scrape each URL and LLM-verify it's a valid page for the product
    all_pass = True
    verification_results = []

    for product in products:
        urls = product_links.get(product, [])
        product_passes = False
        link_checks = []

        for url in urls[:3]:  # check up to 3 URLs per product
            # Scrape the page via Jina
            page_content, scrape_error = fetch_page(url, max_chars=15000)

            if scrape_error:
                link_checks.append(
                    {"url": url, "is_valid": False, "reason": scrape_error}
                )
                continue

            # LLM-verify the page is valid for this product
            verification = _verify_link_with_llm(
                product_name=product,
                url=url,
                page_content=page_content,
                criterion_description=criterion["description"],
                domain=domain,
                shop_vs_product=shop_vs_product,
            )
            link_checks.append({"url": url, **verification})

            if verification["is_valid"]:
                product_passes = True
                break  # one valid link is enough

        verification_results.append(
            {
                "product_name": product,
                "pass": product_passes,
                "links_checked": link_checks,
            }
        )
        if not product_passes:
            all_pass = False

    score = 1 if all_pass else -1
    reasoning_parts = []
    for vr in verification_results:
        status = "PASS" if vr["pass"] else "FAIL"
        reasons = "; ".join(c.get("reason", "") for c in vr["links_checked"])
        reasoning_parts.append(f"{vr['product_name']}: {status} ({reasons})")

    return CriterionResult(
        criterion_id=cid,
        description=criterion["description"],
        type=ctype,
        score=score,
        stage_reached="link_verification",
        hurdle_tag=hurdle,
        grounding_check=grounding,
        stage_1_result=Stage1Result(all_pass=True, reasoning="All products have links"),
        stage_2_result=Stage2Result(
            all_pass=all_pass,
            reasoning=f"Link verification: {'All valid' if all_pass else 'Some invalid'}",
            product_results=verification_results,
        ),
        reasoning="; ".join(reasoning_parts),
    )


# ---------------------------------------------------------------------------
# Route 2: Non-grounded criteria (Stage 1 only)
# ---------------------------------------------------------------------------


def grade_non_grounded(
    criterion: dict,
    response_text: str,
) -> CriterionResult:
    """Grade non-grounded criterion. Score 0 or +1."""
    cid = _get_cid(criterion)
    ctype = _get_ctype(criterion)
    hurdle = _get_hurdle(criterion)
    grounding = _get_grounding(criterion)

    prompt = prompts.STAGE_1_NON_GROUNDED.format(
        criterion_description=criterion["description"],
        response=response_text,
    )
    try:
        text = call_judge(prompt)
        data = parse_json(text)
        passed = data.get("pass", False)
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=1 if passed else 0,
            stage_reached="response_text_only",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=Stage1Result(
                all_pass=passed,
                reasoning=data.get("reasoning", ""),
                evaluation_type=data.get("evaluation_type", "holistic"),
            ),
            reasoning=data.get("reasoning", ""),
        )
    except Exception as e:
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=0,
            stage_reached="response_text_only",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=Stage1Result(all_pass=False, reasoning=f"Error: {e}"),
            reasoning=f"Error: {e}",
        )


# ---------------------------------------------------------------------------
# Route 3: Grounded criteria (Stage 1 + Stage 2)
# ---------------------------------------------------------------------------


def grade_grounded(
    criterion: dict,
    response_text: str,
    products: list[str],
    sources: list[dict],
    product_source_map: list[dict] | None = None,
) -> CriterionResult:
    """Grade grounded criterion. Score 0 (fail S1), +1 (pass both), -1 (fail S2)."""
    cid = _get_cid(criterion)
    ctype = _get_ctype(criterion)
    hurdle = _get_hurdle(criterion)
    grounding = _get_grounding(criterion)

    # --- Stage 1: Response text check ---
    product_list = (
        ", ".join(products) if products else "(No recommendations identified)"
    )
    prompt = prompts.STAGE_1_GROUNDED.format(
        criterion_description=criterion["description"],
        response=response_text,
        product_list=product_list,
    )

    try:
        text = call_judge(prompt)
        stage1_data = parse_json(text)
    except Exception as e:
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=0,
            stage_reached="response_text",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=Stage1Result(all_pass=False, reasoning=f"Error: {e}"),
            reasoning=f"Stage 1 error: {e}",
        )

    stage1 = Stage1Result(
        all_pass=stage1_data.get("pass", False),
        reasoning=stage1_data.get("reasoning", ""),
        evaluation_type=stage1_data.get("evaluation_type", "per_product_all"),
        products_checked=stage1_data.get("recommendations_checked", []),
        required_pass_count=stage1_data.get("required_pass_count", -1),
    )

    if not stage1.all_pass:
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=0,
            stage_reached="response_text",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=stage1,
            reasoning=f"Failed response text check: {stage1.reasoning}",
        )

    # --- Stage 2: Grounding verification ---
    if not sources:
        # No sources → can't verify grounding → fail Stage 2
        # Matches autograder: "No grounding sources available" → -1
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=-1,
            stage_reached="grounded_sources",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=stage1,
            stage_2_result=Stage2Result(
                all_pass=False,
                reasoning="No grounding sources available for verification",
            ),
            reasoning="Passed Stage 1; failed Stage 2 (no sources)",
        )

    # Determine which products passed Stage 1
    passed_products = (
        [
            p["recommendation_name"]
            for p in stage1.products_checked
            if p.get("meets_criterion", False)
        ]
        if stage1.products_checked
        else products
    )

    if not passed_products:
        # No products to verify → holistic grounding check
        stage2 = _stage2_holistic(criterion, sources)
        return CriterionResult(
            criterion_id=cid,
            description=criterion["description"],
            type=ctype,
            score=1 if stage2.all_pass else -1,
            stage_reached="grounded_sources",
            hurdle_tag=hurdle,
            grounding_check=grounding,
            stage_1_result=stage1,
            stage_2_result=stage2,
            reasoning=f"Stage 1: {stage1.reasoning} | Stage 2: {stage2.reasoning}",
        )

    # Per-product verification
    if product_source_map:
        stage2 = _stage2_per_product_mapped(
            criterion,
            passed_products,
            sources,
            product_source_map,
        )
    else:
        stage2 = _stage2_per_product_batched(
            criterion,
            passed_products,
            sources,
        )

    # Apply evaluation_type logic
    pass_count = sum(1 for r in stage2.product_results if r.get("pass", False))
    total = len(passed_products)
    eval_type = stage1.evaluation_type
    required = stage1.required_pass_count

    if eval_type == "per_product_any":
        needed = required if required != -1 else total
        grounding_pass = pass_count >= needed
    else:
        grounding_pass = pass_count == total

    stage2.all_pass = grounding_pass
    stage2.reasoning = f"Source verification: {pass_count}/{total} passed"

    return CriterionResult(
        criterion_id=cid,
        description=criterion["description"],
        type=ctype,
        score=1 if grounding_pass else -1,
        stage_reached="grounded_sources",
        hurdle_tag=hurdle,
        grounding_check=grounding,
        stage_1_result=stage1,
        stage_2_result=stage2,
        reasoning=f"Stage 1: {stage1.reasoning} | Stage 2: {stage2.reasoning}",
    )


def _stage2_holistic(criterion: dict, sources: list[dict]) -> Stage2Result:
    """Holistic grounding check (no per-product, e.g. Gaming/DIY)."""
    source_text = build_source_text_for_grounding(sources)
    prompt = prompts.STAGE_2_HOLISTIC.format(
        criterion_description=criterion["description"],
        source_text=source_text,
    )
    try:
        text = call_judge(prompt)
        data = parse_json(text)
        return Stage2Result(
            all_pass=data.get("pass", False),
            reasoning=data.get("reason", ""),
        )
    except Exception as e:
        return Stage2Result(all_pass=False, reasoning=f"Error: {e}")


def _stage2_per_product_mapped(
    criterion: dict,
    passed_products: list[str],
    sources: list[dict],
    product_source_map: list[dict],
) -> Stage2Result:
    """Per-product Stage 2 with product_source_map (eval path).

    Checks each product against only its mapped sources,
    matching autograder.check_grounded_sources behavior.
    """
    product_results = []
    for product_name in passed_products:
        source_text = build_source_text_for_grounding(
            sources,
            product_source_map,
            product_name,
        )
        if not source_text:
            product_results.append(
                {
                    "product_name": product_name,
                    "pass": False,
                    "reason": "No sources mapped to this product",
                }
            )
            continue

        prompt = prompts.STAGE_2_PER_PRODUCT.format(
            criterion_description=criterion["description"],
            products_to_check=json.dumps([product_name]),
            source_text=source_text,
        )
        try:
            text = call_judge(prompt)
            results = parse_json(text)
            if isinstance(results, list) and results:
                product_results.append(
                    {
                        "product_name": product_name,
                        "pass": results[0].get("pass", False),
                        "reason": results[0].get("reason", ""),
                    }
                )
            else:
                product_results.append(
                    {
                        "product_name": product_name,
                        "pass": False,
                        "reason": "Invalid response from judge",
                    }
                )
        except Exception as e:
            product_results.append(
                {
                    "product_name": product_name,
                    "pass": False,
                    "reason": f"Error: {e}",
                }
            )

    return Stage2Result(all_pass=False, reasoning="", product_results=product_results)


def _stage2_per_product_batched(
    criterion: dict,
    passed_products: list[str],
    sources: list[dict],
) -> Stage2Result:
    """Per-product Stage 2 without product_source_map (RL path).

    Batches all products in one call with all sources for efficiency.
    """
    source_text = build_source_text_for_grounding(sources)
    prompt = prompts.STAGE_2_PER_PRODUCT.format(
        criterion_description=criterion["description"],
        products_to_check=json.dumps(passed_products),
        source_text=source_text,
    )
    try:
        text = call_judge(prompt)
        results = parse_json(text)
        if isinstance(results, list):
            product_results = [
                {
                    "product_name": r.get("recommendation_name", ""),
                    "pass": r.get("pass", False),
                    "reason": r.get("reason", ""),
                }
                for r in results
            ]
        else:
            product_results = [
                {"product_name": p, "pass": False, "reason": "Invalid response"}
                for p in passed_products
            ]
    except Exception as e:
        product_results = [
            {"product_name": p, "pass": False, "reason": f"Error: {e}"}
            for p in passed_products
        ]

    return Stage2Result(all_pass=False, reasoning="", product_results=product_results)


# ---------------------------------------------------------------------------
# Criterion router
# ---------------------------------------------------------------------------


def grade_criterion(
    criterion: dict,
    response_text: str,
    products: list[str],
    sources: list[dict],
    product_source_map: list[dict] | None = None,
    domain: str = "shopping",
    shop_vs_product: str = "Product",
) -> CriterionResult:
    """Route and grade a single criterion. Returns CriterionResult."""
    ctype = _get_ctype(criterion)

    if ctype == "Provides link(s)":
        return grade_link_criterion(
            criterion,
            response_text,
            products,
            domain=domain,
            shop_vs_product=shop_vs_product,
        )

    if not _is_grounded(criterion):
        return grade_non_grounded(criterion, response_text)

    return grade_grounded(
        criterion, response_text, products, sources, product_source_map
    )


# ---------------------------------------------------------------------------
# Task-level scoring
# ---------------------------------------------------------------------------


def grade_task(
    task_id: str,
    response_text: str,
    criteria: list[dict],
    sources: list[dict],
    product_source_map: list[dict] | None = None,
    products: list[str] | None = None,
    query: str = "",
    domain: str = "unknown",
    shop_vs_product: str = "Product",
) -> TaskResult:
    """Grade all criteria for a task. Returns full structured TaskResult.

    This is THE scoring function. Both eval and RL call this.

    Args:
        task_id: ACE task ID.
        response_text: Model's final response.
        criteria: List of criterion dicts from parquet/dataset.
        sources: Source dicts (from tool_history, Jina, or pre-built JSON).
        product_source_map: Optional mapping of products to source indices.
            When provided (eval path), Stage 2 checks per-product sources.
            When None (RL path), Stage 2 batches all products with all sources.
        products: Pre-extracted product names. If None, extracts them.
        query: Original user query (needed for product extraction).
        domain: ACE domain (shopping, food, gaming, diy).
        shop_vs_product: 'Product' or 'Shop' (Shopping domain link verification mode).
    """
    domain = domain.lower()

    if not criteria:
        return TaskResult(task_id=task_id, num_criteria=0)

    # Extract products if not provided
    if products is None:
        from ace_scoring.product import extract_products

        has_grounded = any(_is_grounded(c) for c in criteria)
        if has_grounded or domain in ("shopping", "gaming"):
            products = extract_products(response_text, query)
        else:
            products = []

    # Grade each criterion
    results: list[CriterionResult] = []
    for c in criteria:
        result = grade_criterion(
            c,
            response_text,
            products,
            sources,
            product_source_map,
            domain=domain,
            shop_vs_product=shop_vs_product,
        )
        results.append(result)

    # Build scores_with_types: [[score, type, hurdle_tag], ...]
    scores_with_types = []
    for r in results:
        scores_with_types.append([r.score, r.type, r.hurdle_tag])

    scores = [r.score for r in results]
    total_score = sum(scores)

    # Hurdle gate (matches autograder lines 1061-1071)
    # If no hurdle criteria, no gate applied → hurdle_score = total_score.
    # If any hurdle criterion scores ≤ 0, gate fails → hurdle_score = 0.
    hurdle_scores = [
        s[0] for s in scores_with_types if len(s) >= 3 and s[2] == "Hurdle"
    ]
    if hurdle_scores and any(s <= 0 for s in hurdle_scores):
        total_hurdle_score = 0
    else:
        total_hurdle_score = total_score

    summary = {
        "pass_count": sum(1 for s in scores if s == 1),
        "fail_response_count": sum(1 for s in scores if s == 0),
        "fail_source_count": sum(1 for s in scores if s == -1),
        "total": len(scores),
        "hurdle_count": len(hurdle_scores),
        "hurdle_pass_count": sum(1 for s in hurdle_scores if s == 1),
    }

    return TaskResult(
        task_id=task_id,
        criteria_scores=scores_with_types,
        total_score=total_score,
        total_hurdle_score=total_hurdle_score,
        num_criteria=len(criteria),
        summary=summary,
        detailed_results=results,
        products=products,
    )

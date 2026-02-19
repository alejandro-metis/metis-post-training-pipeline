"""All LLM judge prompt templates for ACE scoring.

Single source of truth. Ported from apex-evals/ace/harness/autograder.py
and ace_reward.py. See ACE paper Section 4 / Figure 4.
"""

# ---------------------------------------------------------------------------
# Stage 1: Response text check — grounded criteria
# From autograder.check_response_text
# ---------------------------------------------------------------------------

STAGE_1_GROUNDED = """You are evaluating a response against a specific criterion.

Criterion to evaluate: {criterion_description}

Response Text:
{response}

**CANONICAL RECOMMENDATION LIST (use EXACTLY these) if applicable:**
{product_list}

CRITICAL EVALUATION RULES:

1. **ZERO BACKGROUND KNOWLEDGE**: You must COMPLETELY IGNORE any knowledge you have about these recommendations/products. Pretend you have NEVER heard of them before.

2. **ONLY EXPLICIT STATEMENTS COUNT**: A criterion is met for a recommendation ONLY if the response DIRECTLY states the required information or it is obviously inferrable.
   - "supports 2+ players" → response MUST explicitly say "multiplayer", "2 players", etc.
   - If something is IMPLIED or requires INFERENCE → does NOT count

3. **USE THE CANONICAL LIST**: Evaluate EXACTLY the recommendations listed above. Do NOT add, remove, or rename.

4. Multi-purpose products are fine:
   Criterion: "Recommends only MMA gloves"
   Product: "Gloves for MMA and Boxing" → meets_criterion = TRUE

5. **DETERMINE QUANTIFIER**:
   - "at least N" or "one of" → required_pass_count: N, evaluation_type: per_product_any
   - "ALL", "only", "each", or no quantifier → required_pass_count: -1, evaluation_type: per_product_all
   - "Provides X items" (quantity check) → evaluation_type: holistic
   - DEFAULT: If no explicit quantifier, assume ALL must pass → required_pass_count: -1

   **IMPORTANT:** "one or more of the following FEATURES" refers to features PER product, not number of products. This is still per_product_all.
   **IMPORTANT:** "Recommends a [description]" means ALL recommendations must match. The "a" is grammar, NOT a quantifier.

6. **NO VACUOUS TRUTH**: If the response contains NO recommendations:
   - Criteria that CHECK features → FAIL
   - Criteria that REQUIRE providing something → FAIL

7. **CONDITIONAL CRITERIA**: If criterion starts with "If", treat "available in X", "comes in X" as EQUIVALENT to "recommends X". Stock caveats do NOT cause failure.

8. **PER-PRODUCT STRICT CHECKING**: For each recommendation, look at ONLY the text describing that specific recommendation.

Return ONLY valid JSON:
{{
  "pass": true/false,
  "reasoning": "Clear explanation",
  "evaluation_type": "holistic|per_product_all|per_product_any",
  "required_pass_count": -1,
  "recommendations_checked": [{{"recommendation_name": "EXACT name", "meets_criterion": true/false}}]
}}"""


# ---------------------------------------------------------------------------
# Stage 1: Non-grounded criteria (text-only, simpler prompt)
# From autograder.grade_non_grounding_criterion
# ---------------------------------------------------------------------------

STAGE_1_NON_GROUNDED = """You are evaluating a response against a specific criterion.

Criterion to evaluate: {criterion_description}

Response Text:
{response}

CRITICAL EVALUATION RULES:

1. **ZERO BACKGROUND KNOWLEDGE**: Completely ignore any knowledge you have about these products. Pretend you have never heard of them.

2. **ONLY EXPLICIT STATEMENTS COUNT**: A criterion is met ONLY if the response DIRECTLY states the required information or it is obviously inferrable.

3. Multi-purpose products are fine:
   Criterion: "Recommends only MMA gloves"
   Product: "Gloves for MMA and Boxing" → TRUE

4. **DETERMINE SCOPE**:
   - HOLISTIC: Applies to the overall response (e.g., "recommends exactly 3 items")
   - PER_RECOMMENDATION: Applies to each individual recommendation

5. Criteria with "only" or "all": Pass only if EVERY recommendation meets it.
   Criteria about quantity/count: Evaluate the overall response.

6. **CONDITIONAL CRITERIA**: If criterion starts with "If", treat "available in X", "comes in X" as EQUIVALENT to "recommends X". Stock caveats do NOT cause failure.

7. **NO VACUOUS TRUTH**: If the response has no recommendations, criteria checking features or requiring information → FAIL.

Be reasonable and logical.

Return ONLY valid JSON:
{{"pass": true/false, "reasoning": "Clear explanation"}}"""


# ---------------------------------------------------------------------------
# Stage 2: Grounding verification — per-product
# From autograder.check_grounded_sources
# ---------------------------------------------------------------------------

STAGE_2_PER_PRODUCT = """Your job is to check that our grading of a model's response is correct, based on source material.

We gave a model a prompt asking for recommendations. It responded, and we graded whether the response meets the following criterion: {criterion_description}

The following recommendations passed this criterion in the response text. For EACH one, check whether it actually passes given the source material. This catches hallucination — the model may have made claims that aren't backed by any source.

Recommendations to verify:
{products_to_check}

Source content:
{source_text}

A recommendation PASSES if the criterion is true in AT LEAST ONE source (does NOT need all sources).

Return ONLY valid JSON — an array with one entry per recommendation:
[{{"recommendation_name": "exact name", "pass": true/false, "reason": "brief explanation"}}]"""


# ---------------------------------------------------------------------------
# Stage 2: Grounding verification — holistic
# From autograder.check_grounded_sources (holistic path)
# ---------------------------------------------------------------------------

STAGE_2_HOLISTIC = """You are verifying if a criterion is supported by grounding sources.

Criterion to verify: {criterion_description}

Grounding Source Content:
{source_text}

Based ONLY on the source content above, is the criterion's claim supported or verifiable?
- If sources contain information that supports the criterion: pass = true
- If sources do not mention or support the criterion: pass = false

Return ONLY valid JSON:
{{"pass": true/false, "reason": "Brief explanation citing which source(s) support or contradict"}}"""


# ---------------------------------------------------------------------------
# Product extraction
# From grounding-pipeline.py extract_recommendations
# ---------------------------------------------------------------------------

EXTRACT_PRODUCTS = """Extract the specific product, service, or recommendation names from this response.

User's question: {query}

Response:
{response}

Rules:
- Only extract names of items the response directly recommends
- Do NOT include items mentioned as "not recommended" or "for comparison"
- A "meal plan" or "recipe" is ONE recommendation, not individual ingredients
- A "DIY project" guide is ONE recommendation, not individual steps/materials
- Use the exact names as they appear in the response

Return ONLY valid JSON — an array of strings:
["Product Name 1", "Product Name 2", ...]"""


# ---------------------------------------------------------------------------
# Product-to-source mapping
# From grounding-pipeline.py create_recommendation_source_map
# ---------------------------------------------------------------------------

MAP_PRODUCTS_TO_SOURCES = """Map each product/recommendation to the sources that support it.

Products: {product_names}

Available sources (0-indexed):
{sources_text}

For each product, return the indices of sources that are relevant to it
(contain information about that product — pricing, features, reviews, purchase page, etc.).

Return ONLY valid JSON — an array of objects:
[{{"product_name": "...", "source_indices": [0, 2, 5]}}]"""


# ---------------------------------------------------------------------------
# Link extraction per product
# From autograder.extract_links_for_product
# ---------------------------------------------------------------------------

EXTRACT_LINKS_FOR_PRODUCT = """For each product below, extract URLs (http:// or https://) from the response that are specifically associated with that product.

Products: {products}

Response text:
{response}

Return ONLY valid JSON — an object mapping product name to its URLs:
{{"Product Name": ["url1", "url2"]}}"""


# ---------------------------------------------------------------------------
# Link verification — product purchase page
# From autograder.helpers.purchase_page_verifier.verify_purchase_link
# ---------------------------------------------------------------------------

VERIFY_PURCHASE_LINK = """You are classifying a webpage to determine if it's a valid {mode_text} page.

Product/Vendor name: {product_name}

Page content:
{page_content}

{validation_criteria}

Return ONLY valid JSON:
{{
  "is_valid": true/false,
  "page_type": "product_page|vendor_page|list|search_results|review|general|error",
  "reason": "Brief explanation why valid or invalid"
}}"""


# ---------------------------------------------------------------------------
# Link verification — gaming/general criterion
# From autograder.helpers.purchase_page_verifier.verify_gaming_link
# ---------------------------------------------------------------------------

VERIFY_GAMING_LINK = """You are verifying if a webpage meets a link requirement for a criterion.

Game/Item: {product_name}
URL: {url}
Criterion: {criterion_description}

Page content:
{page_content}

Based on the criterion description, does this webpage meet the link requirement?

EXAMPLES:
- If criterion says "official Nintendo links" → URL must be from nintendo.com or official Nintendo domain
- If criterion says "Steam store links" → URL must be from steampowered.com or store.steampowered.com
- If criterion says "purchase links" → Must be a page where the item can be purchased
- If criterion says "official links" → Must be from the official/authoritative source

Verify both:
1. URL domain matches the requirement
2. Page content is relevant to the game/item

Return ONLY valid JSON:
{{
  "is_valid": true/false,
  "page_type": "official|store|forum|video|wiki|third_party|other",
  "reason": "Brief explanation. Mention the domain and content type."
}}"""

"""
ACE reward function for verl GRPO training.

Uses OpenAI gpt-4o-mini as LLM judge to evaluate model responses
against ACE criteria. Compatible with verl's reward function signature.
"""

import json
import os

from openai import OpenAI

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TEMPERATURE = 0.0

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def compute_reward(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict
) -> float:
    """Evaluate a model response against ACE criteria using LLM judge.

    Args:
        data_source: Dataset identifier (e.g. "mercor/ACE")
        solution_str: The model's generated response
        ground_truth: JSON-stringified list of criteria
        extra_info: Metadata dict with domain, task_id, etc.

    Returns:
        Float reward in [0.0, 1.0]. 0.0 if any hurdle criterion fails.
    """
    criteria = json.loads(ground_truth)

    if not criteria:
        return 0.0

    criteria_block = "\n".join(
        f"{i+1}. [ID: {c['criterion_id']}] [Hurdle: {c['hurdle_tag']}] {c['description']}"
        for i, c in enumerate(criteria)
    )

    prompt = f"""You are evaluating an AI assistant's response against a set of criteria.

For each criterion, determine if the response PASSES or FAILS based ONLY on what is explicitly stated in the response text. Do not use any background knowledge.

RESPONSE:
{solution_str}

CRITERIA:
{criteria_block}

Rules:
- A criterion passes ONLY if the response directly and explicitly addresses it
- Implicit or inferred information does NOT count
- Be strict but reasonable

Return ONLY valid JSON — an array of objects, one per criterion, in order:
[{{"criterion_id": "...", "pass": true/false}}]"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        results = json.loads(text)

        # Build lookup for results
        result_map = {str(r["criterion_id"]): r["pass"] for r in results}

        passed = 0
        hurdle_failed = False

        for c in criteria:
            cid = str(c["criterion_id"])
            did_pass = result_map.get(cid, False)

            if did_pass:
                passed += 1
            elif c["hurdle_tag"] == "Hurdle":
                hurdle_failed = True

        if hurdle_failed:
            return 0.0

        return passed / len(criteria)

    except Exception as e:
        print(f"[ace_reward] Error in compute_reward: {e}")
        return 0.0

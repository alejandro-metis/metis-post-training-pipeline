"""ACE reward function for verl GRPO training.

Thin wrapper around ace_scoring that returns float for verl's interface.
For custom RL reward shaping, use compute_reward_structured() to get the
full TaskResult with per-criterion scores, then shape however you want.

verl interface: compute_reward(data_source, solution_str, ground_truth, extra_info) -> float
"""

import json

from ace_scoring.scorer import grade_task
from ace_scoring.sources import sources_from_tool_history
from ace_scoring.types import TaskResult


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    """Evaluate a model response against ACE criteria.

    Returns float reward in [-1.0, 1.0].
    Normalized: total_score / num_criteria.
    Hurdle gate: any hurdle <= 0 -> 0.0.
    """
    result = compute_reward_structured(
        data_source, solution_str, ground_truth, extra_info
    )
    return result.to_reward(normalize=True)


def compute_reward_structured(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> TaskResult:
    """Same as compute_reward but returns full structured TaskResult.

    Use this for custom RL reward shaping:
        result = compute_reward_structured(...)

        # Access per-criterion scores
        for cr in result.detailed_results:
            cr.score        # -1, 0, or +1
            cr.type         # criterion type
            cr.hurdle_tag   # "Hurdle" or "Not"
            cr.stage_reached  # where grading stopped
            cr.stage_1_result # full Stage 1 output
            cr.stage_2_result # full Stage 2 output (grounding)

        # Custom shaping examples:
        # reward = result.to_reward(normalize=False)  # raw sum
        # grounding_penalty = result.summary["fail_source_count"] * -0.5
        # hurdle_bonus = 0.5 if result.summary["hurdle_pass_count"] == result.summary["hurdle_count"] else 0.0
    """
    criteria = json.loads(ground_truth)
    if not criteria:
        return TaskResult(task_id=extra_info.get("task_id", "unknown"), num_criteria=0)

    tool_history = extra_info.get("tool_history")
    sources = sources_from_tool_history(tool_history) if tool_history else []

    try:
        return grade_task(
            task_id=extra_info.get("task_id", "unknown"),
            response_text=solution_str,
            criteria=criteria,
            sources=sources,
            product_source_map=None,
            query=extra_info.get("prompt", ""),
            domain=extra_info.get("domain", "unknown"),
            shop_vs_product=extra_info.get("shop_vs_product", "Product"),
        )
    except Exception as e:
        print(
            f"[ace_reward] grade_task failed for task {extra_info.get('task_id', '?')}: {e}"
        )
        return TaskResult(
            task_id=extra_info.get("task_id", "unknown"), num_criteria=len(criteria)
        )

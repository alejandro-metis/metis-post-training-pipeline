"""Structured result types for ACE scoring.

Matches the official autograder's output format (3_autograder_results.json)
while exposing per-criterion detail for RL reward shaping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Stage1Result:
    """Stage 1: Response text check result."""

    all_pass: bool
    reasoning: str
    evaluation_type: str = "per_product_all"
    products_checked: list[dict] = field(default_factory=list)
    required_pass_count: int = -1


@dataclass
class Stage2Result:
    """Stage 2: Grounding verification result."""

    all_pass: bool
    reasoning: str
    product_results: list[dict] = field(default_factory=list)


@dataclass
class CriterionResult:
    """Per-criterion grading result. Matches autograder detailed_results entries."""

    criterion_id: str
    description: str
    type: str
    score: int  # -1, 0, or +1
    stage_reached: (
        str  # response_text | grounded_sources | response_text_only | link_verification
    )
    hurdle_tag: str
    grounding_check: str
    stage_1_result: Optional[Stage1Result] = None
    stage_2_result: Optional[Stage2Result] = None
    reasoning: str = ""


@dataclass
class TaskResult:
    """Task-level scoring result. Matches 3_autograder_results.json schema."""

    task_id: str
    criteria_scores: list[list] = field(
        default_factory=list
    )  # [[score, type, hurdle_tag], ...]
    total_score: int = 0  # raw sum (leaderboard)
    total_hurdle_score: int = 0  # 0 if any hurdle fails, else total_score
    num_criteria: int = 0
    summary: dict = field(default_factory=dict)
    detailed_results: list[CriterionResult] = field(default_factory=list)
    products: list[str] = field(default_factory=list)

    def to_reward(self, normalize: bool = True) -> float:
        """Convert to scalar reward for RL training.

        Args:
            normalize: If True, divide by num_criteria -> [-1.0, 1.0].
                       If False, return raw total_score as float.
        """
        if self._has_hurdles() and self.total_hurdle_score == 0:
            return 0.0
        if normalize:
            return (
                self.total_score / self.num_criteria if self.num_criteria > 0 else 0.0
            )
        return float(self.total_score)

    def _has_hurdles(self) -> bool:
        return any(len(cs) >= 3 and cs[2] == "Hurdle" for cs in self.criteria_scores)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict matching 3_autograder_results.json."""
        return asdict(self)


def format_criteria_for_autograder(criteria: list[dict]) -> list[dict]:
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
        formatted.append(
            {
                "criterion_id": c.get("criterion_id", str(i + 1)),
                "id": i + 1,
                "description": c["description"],
                "type": c.get("criteria_type", "standard"),
                "hurdle_tag": c.get("hurdle_tag", "Not"),
                "grounded_status": c.get("grounding_check", "Not Grounded"),
            }
        )
    return formatted

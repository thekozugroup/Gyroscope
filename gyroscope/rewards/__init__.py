"""Phase 4 — Reward design: turn a GoldenDocument into GRPO reward functions."""

from gyroscope.rewards.codegen import emit_rewards_module
from gyroscope.rewards.designer import design_rewards
from gyroscope.rewards.judges import LLMJudge, heuristic_judge
from gyroscope.rewards.pipeline import RewardsPipeline
from gyroscope.rewards.spec import RewardBundle, RewardKind, RewardSpec

__all__ = [
    "LLMJudge",
    "RewardBundle",
    "RewardKind",
    "RewardSpec",
    "RewardsPipeline",
    "design_rewards",
    "emit_rewards_module",
    "heuristic_judge",
]

"""Pipeline configuration shared by all phases."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    curator_model: str = "claude-opus-4-7"
    swarm_model: str = "claude-sonnet-4-6"
    critic_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-haiku-4-5-20251001"

    max_tokens: int = 4096
    temperature_curator: float = 0.2
    temperature_swarm: float = 0.9
    temperature_critic: float = 0.0

    max_concurrent: int = 16
    """Max in-flight requests across the pipeline."""

    cache_prompts: bool = True
    """Use Anthropic prompt caching on stable system/golden-doc content."""


class CurationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_target_tokens: int = 1200
    chunk_overlap_tokens: int = 100
    dedup_threshold: float = 0.85
    """Jaccard similarity above which chunks are treated as duplicates."""
    max_principles: int = 60
    max_procedures: int = 40
    max_knowledge_items: int = 400


class SFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_trajectories: int = 1000
    output_format: Literal["sharegpt", "chatml", "alpaca"] = "sharegpt"
    n_personas: int = 8
    difficulty_mix: dict[str, float] = Field(
        default_factory=lambda: {"easy": 0.2, "medium": 0.5, "hard": 0.25, "adversarial": 0.05}
    )
    critic_min_score: float = 0.7
    """Trajectories below this score are dropped or repaired."""
    max_repair_attempts: int = 2
    dedup_threshold: float = 0.9
    eval_holdout_fraction: float = 0.05


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward_budget: int = 12
    """Approx number of reward functions to emit."""
    include_kinds: list[str] = Field(
        default_factory=lambda: [
            "format",
            "lexical",
            "principle",
            "procedure",
            "safety",
            "citation",
            "length",
        ]
    )
    judge_model: str | None = None  # falls back to LLMConfig.judge_model


class GyroscopeConfig(BaseModel):
    """Top-level config object passed to every pipeline."""

    model_config = ConfigDict(extra="forbid")

    input_paths: list[Path] = Field(default_factory=list)
    output_dir: Path = Path("./runs/default")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)

    seed: int = 7
    log_level: str = "INFO"

    api_key: str | None = Field(default=None, repr=False)

    def resolved_api_key(self) -> str:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Pass api_key= or export the env var."
            )
        return key

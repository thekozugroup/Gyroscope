"""Phase 4 entry point: run designer + codegen end-to-end."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gyroscope.rewards.codegen import emit_rewards_module
from gyroscope.rewards.designer import design_rewards

if TYPE_CHECKING:
    from gyroscope.core.config import RewardConfig
    from gyroscope.core.llm import LLMClient
    from gyroscope.core.models import GoldenDocument

logger = logging.getLogger(__name__)

__all__ = ["RewardsPipeline"]


@dataclass(slots=True)
class RewardsPipeline:
    """Run the reward design + codegen pipeline."""

    async def run(
        self,
        golden: GoldenDocument,
        output_dir: Path | str,
        client: LLMClient | None = None,
        *,
        config: RewardConfig,
    ) -> Path:
        """Design rewards from ``golden`` and emit them under ``output_dir``.

        Returns the path to the generated ``rewards.py`` file. The ``client``
        argument is accepted for forward compatibility with future
        LLM-driven designers; the deterministic designer ignores it.
        """
        output_dir = Path(output_dir)
        bundle = await design_rewards(golden, client=client, config=config)
        if not bundle.specs:
            logger.warning(
                "Reward designer produced 0 specs from this golden document. "
                "Common causes: empty principles, no procedures, no anti-patterns, "
                "or the configured include_kinds is too narrow."
            )
        rewards_path = emit_rewards_module(bundle, golden, output_dir)
        logger.info(
            "RewardsPipeline: wrote %d reward functions to %s",
            len(bundle.specs),
            rewards_path,
        )
        return rewards_path

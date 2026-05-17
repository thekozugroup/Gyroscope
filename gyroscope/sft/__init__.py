"""Phase 3 — SFT Swarm: convert a GoldenDocument into a multi-turn SFT dataset."""

from gyroscope.sft.formats import to_alpaca, to_chatml, to_sharegpt
from gyroscope.sft.personas import generate_personas
from gyroscope.sft.pipeline import SFTPipeline
from gyroscope.sft.scenarios import generate_scenarios
from gyroscope.sft.swarm import run_swarm
from gyroscope.sft.trajectory import build_trajectory

__all__ = [
    "SFTPipeline",
    "build_trajectory",
    "generate_personas",
    "generate_scenarios",
    "run_swarm",
    "to_alpaca",
    "to_chatml",
    "to_sharegpt",
]

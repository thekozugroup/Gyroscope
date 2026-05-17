"""Shared retry policy constants for both the async LLMClient and the
sync LLMJudge.

Centralising these here means a 429 burst sees the same backoff / attempt
budget regardless of whether the caller is the curation extractor, the
SFT swarm, or a GRPO trainer hook invoking the sync judge. Tuning one
knob now tunes both sides.
"""

from __future__ import annotations

# Maximum number of attempts (including the initial try).
MAX_ATTEMPTS: int = 5

# Base delay (seconds) used as the floor for jitter sampling.
BACKOFF_BASE: float = 1.0

# Cap on any individual sleep — bound on a worst-case retry storm.
BACKOFF_CAP: float = 60.0

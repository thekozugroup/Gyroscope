"""Held-out evaluation set generation.

The eval split is produced by the SFT swarm (`gyroscope.sft.swarm.run_swarm`
returns `(train, eval)`); this module exists to (a) write it to disk in the
configured format and (b) provide leakage checks.
"""

from gyroscope.eval.pipeline import EvalPipeline, leakage_check

__all__ = ["EvalPipeline", "leakage_check"]

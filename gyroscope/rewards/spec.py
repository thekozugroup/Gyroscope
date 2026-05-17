"""Reward specification data contracts.

Re-exports the canonical :class:`RewardSpec` and :class:`RewardKind` from
:mod:`gyroscope.core.models` and adds :class:`RewardBundle`, which groups a
list of specs along with the role they enforce and a schema version stamp.

The bundle is the on-disk artefact (``reward_spec.yaml``) produced by the
designer and consumed by the code generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from gyroscope.core.io import read_yaml, write_yaml
from gyroscope.core.models import RewardKind, RewardSpec

__all__ = ["RewardBundle", "RewardKind", "RewardSpec"]


class RewardBundle(BaseModel):
    """A collection of :class:`RewardSpec` objects with provenance.

    Attributes:
        specs: ordered list of reward specs (priority order roughly matches the
            list order so a downstream caller can drop the tail to fit a budget).
        golden_role: ``Identity.role`` of the GoldenDocument the specs derive from.
        version: schema version of the bundle, useful for forward compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    specs: list[RewardSpec] = Field(default_factory=list)
    golden_role: str = Field(..., description="Role string from the source GoldenDocument.")
    version: str = Field(default="1", description="Bundle schema version.")

    # ------------------------------------------------------------------
    # YAML round-trip
    # ------------------------------------------------------------------

    def to_yaml(self, path: Path | str) -> Path:
        """Write the bundle to ``path`` as YAML and return the resolved path."""
        payload: dict[str, Any] = self.model_dump(mode="json")
        write_yaml(path, payload)
        return Path(path)

    @classmethod
    def from_yaml(cls, path: Path | str) -> RewardBundle:
        """Load a bundle from a YAML file."""
        raw = read_yaml(path)
        if raw is None:
            raise ValueError(f"Empty YAML file: {path!s}")
        return cls.model_validate(raw)

    # Convenience for round-tripping through arbitrary YAML strings (tests).
    def to_yaml_string(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

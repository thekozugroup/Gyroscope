"""Quality metrics + scoring used by the critique loop.

The critique system grades pipeline outputs along five axes:
- coverage      — does the artefact represent the source BoK breadth?
- faithfulness  — is content grounded in (or compatible with) the source?
- diversity     — no degenerate repetition / dataset collapse?
- trainability  — format correct, token budgets sane, schemas valid?
- reward_soundness — are reward functions gameable-resistant, calibrated?

Each axis returns a 0..100 score with a short note. The pipeline iterates
until all axes >= the configured threshold.
"""

from gyroscope.quality.metrics import (
    AxisScore,
    QualityReport,
    coverage_score,
    diversity_score,
    faithfulness_score,
    reward_soundness_score,
    trainability_score,
)
from gyroscope.quality.report import (
    render_report_html,
    render_report_markdown,
    write_report_artefacts,
)

__all__ = [
    "AxisScore",
    "QualityReport",
    "coverage_score",
    "diversity_score",
    "faithfulness_score",
    "render_report_html",
    "render_report_markdown",
    "reward_soundness_score",
    "trainability_score",
    "write_report_artefacts",
]

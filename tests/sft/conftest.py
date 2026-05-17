"""Shared fixtures for the SFT test suite."""

from __future__ import annotations

from typing import Any

import pytest

from gyroscope.core.models import (
    AntiPattern,
    GoldenDocument,
    Identity,
    KnowledgeItem,
    Principle,
    Procedure,
    ProcedureStep,
    Trajectory,
    TrajectoryMessage,
    VocabularyTerm,
)


def make_golden(
    *,
    n_procedures: int = 3,
    n_principles: int = 4,
) -> GoldenDocument:
    return GoldenDocument(
        identity=Identity(
            role="Test Role",
            description="A test agent for unit tests.",
            mission="Answer faithfully.",
        ),
        principles=[
            Principle(id=f"PRN-{i:04d}", statement=f"Principle {i}.")
            for i in range(1, n_principles + 1)
        ],
        procedures=[
            Procedure(
                id=f"PRC-{i:04d}",
                name=f"Procedure {i}",
                purpose=f"Demonstrate procedure {i}.",
                steps=[
                    ProcedureStep(order=1, action="Step one"),
                    ProcedureStep(order=2, action="Step two"),
                ],
            )
            for i in range(1, n_procedures + 1)
        ],
        knowledge=[KnowledgeItem(id="KNW-0001", statement="Fact one.")],
        vocabulary=[VocabularyTerm(term="WAT", definition="What a term.")],
        anti_patterns=[
            AntiPattern(
                id="ANT-0001",
                description="Hallucinate.",
                why_bad="Misleads.",
                correction="Cite sources.",
            )
        ],
    )


def make_trajectory(
    *,
    id_: str = "TRJ-SCN-0001",
    scenario_id: str = "SCN-0001",
    system: str = "You are an expert.",
    user_text: str = "Hi there.",
    assistant_text: str = "Hello!",
    quality_score: float | None = 0.9,
    tags: dict[str, Any] | None = None,
) -> Trajectory:
    return Trajectory(
        id=id_,
        scenario_id=scenario_id,
        system=system,
        messages=[
            TrajectoryMessage(role="user", content=user_text),
            TrajectoryMessage(role="assistant", content=assistant_text),
        ],
        tags=tags
        or {
            "procedure_ids": ["PRC-0001"],
            "principle_ids": ["PRN-0001"],
            "persona": "PER-0001",
            "difficulty": "medium",
        },
        quality_score=quality_score,
        critic_notes="ok",
    )


class FakeLLM:
    """In-memory LLM stub with configurable responses.

    Methods accept the same kwargs as `LLMClient` so it can be substituted for it
    in tests via monkeypatch.
    """

    def __init__(
        self,
        *,
        complete_responses: list[str] | None = None,
        json_responses: list[Any] | None = None,
        json_array_responses: list[list[Any]] | None = None,
        messages_responses: list[str] | None = None,
    ) -> None:
        self.complete_responses = list(complete_responses or [])
        self.json_responses = list(json_responses or [])
        self.json_array_responses = list(json_array_responses or [])
        self.messages_responses = list(messages_responses or [])
        self.complete_calls: list[dict[str, Any]] = []
        self.json_calls: list[dict[str, Any]] = []
        self.json_array_calls: list[dict[str, Any]] = []
        self.messages_calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> str:
        self.complete_calls.append(kwargs)
        if not self.complete_responses:
            return "OK"
        return self.complete_responses.pop(0)

    async def complete_messages(self, **kwargs: Any) -> str:
        self.messages_calls.append(kwargs)
        if not self.messages_responses:
            return "OK"
        return self.messages_responses.pop(0)

    async def complete_json(self, **kwargs: Any) -> Any:
        self.json_calls.append(kwargs)
        if not self.json_responses:
            return {}
        return self.json_responses.pop(0)

    async def complete_json_array(self, **kwargs: Any) -> list[Any]:
        self.json_array_calls.append(kwargs)
        if not self.json_array_responses:
            return []
        return self.json_array_responses.pop(0)


@pytest.fixture
def golden() -> GoldenDocument:
    return make_golden()


@pytest.fixture
def trajectory() -> Trajectory:
    return make_trajectory()

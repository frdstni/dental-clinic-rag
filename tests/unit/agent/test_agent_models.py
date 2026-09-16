import pytest
from pydantic import ValidationError

from dental_rag.agent.models import (
    AgentDecision,
    AgentIntent,
)


def test_agent_decision_creation() -> None:
    decision = AgentDecision(
        intent=AgentIntent.CLINIC,
        confidence=0.95,
        reasoning="Clinic related question",
    )

    assert decision.intent == AgentIntent.CLINIC
    assert decision.confidence == 0.95
    assert decision.reasoning == "Clinic related question"


@pytest.mark.parametrize(
    "intent",
    [
        AgentIntent.CLINIC,
        AgentIntent.GENERAL_DENTAL,
    ],
)
def test_agent_decision_accepts_supported_intents(
    intent: AgentIntent,
) -> None:
    decision = AgentDecision(
        intent=intent,
        confidence=0.8,
        reasoning="Valid reasoning",
    )

    assert decision.intent == intent


@pytest.mark.parametrize(
    "confidence",
    [
        -0.1,
        -1,
        1.1,
        2,
    ],
)
def test_agent_decision_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            intent=AgentIntent.CLINIC,
            confidence=confidence,
            reasoning="Invalid confidence",
        )


@pytest.mark.parametrize(
    "reasoning",
    [
        "",
        " ",
    ],
)
def test_agent_decision_rejects_empty_reasoning(
    reasoning: str,
) -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            intent=AgentIntent.CLINIC,
            confidence=0.9,
            reasoning=reasoning,
        )


def test_agent_decision_requires_intent() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            confidence=0.9,
            reasoning="Missing intent",
        )


def test_agent_decision_requires_confidence() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            intent=AgentIntent.CLINIC,
            reasoning="Missing confidence",
        )


def test_agent_decision_requires_reasoning() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            intent=AgentIntent.CLINIC,
            confidence=0.9,
        )


def test_agent_decision_is_immutable() -> None:
    decision = AgentDecision(
        intent=AgentIntent.CLINIC,
        confidence=0.9,
        reasoning="Test reasoning",
    )

    with pytest.raises(ValidationError):
        decision.confidence = 0.5
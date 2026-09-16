from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class AgentIntent(StrEnum):
    """
    Supported routing destinations for agent workflow.
    """

    CLINIC = "clinic"

    GENERAL_DENTAL = "general_dental"


class AgentDecision(BaseModel):
    """
    Decision returned by the agent router.
    """

    model_config = {
        "frozen": True,
    }

    intent: AgentIntent

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reasoning: str

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError(
                "Reasoning cannot be empty",
            )

        return value
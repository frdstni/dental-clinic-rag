from dental_rag.agent.clinic import ClinicAgent
from dental_rag.agent.general_dental import (
    GeneralDentalAgent,
)
from dental_rag.agent.models import AgentIntent
from dental_rag.agent.router import AgentRouter
from dental_rag.agent.state import AgentState


class AgentNodes:
    """
    LangGraph node implementations.
    """

    def __init__(
        self,
        router: AgentRouter,
        clinic_agent: ClinicAgent,
        general_dental_agent: GeneralDentalAgent,
    ) -> None:
        self.router = router
        self.clinic_agent = clinic_agent
        self.general_dental_agent = (
            general_dental_agent
        )

    def route_query(
        self,
        state: AgentState,
    ) -> AgentState:
        decision = self.router.route(
            state["query"],
        )

        return {
            **state,
            "intent": decision.intent,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }

    def clinic_node(
        self,
        state: AgentState,
    ) -> AgentState:
        return self.clinic_agent.run(
            state,
        )

    def general_dental_node(
        self,
        state: AgentState,
    ) -> AgentState:
        return self.general_dental_agent.run(
            state,
        )


def route_condition(
    state: AgentState,
) -> str:
    """
    Decide next graph node based on intent.
    """

    intent = state.get(
        "intent",
    )

    if intent == AgentIntent.CLINIC:
        return "clinic"

    if intent == AgentIntent.GENERAL_DENTAL:
        return "general_dental"

    raise ValueError(
        "Unknown agent intent",
    )
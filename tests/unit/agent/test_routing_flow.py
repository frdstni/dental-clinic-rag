import pytest

from dental_rag.agent.clinic import ClinicAgent
from dental_rag.agent.general_dental import (
    GeneralDentalAgent,
)
from dental_rag.agent.models import AgentIntent
from dental_rag.agent.nodes import AgentNodes
from dental_rag.agent.router import AgentRouter
from dental_rag.agent.workflow import create_agent_graph


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response

    def generate(
        self,
        prompt: str,
    ) -> str:
        return self.response


class FakeRagService:
    def retrieve_with_quality(
        self,
        query: str,
    ):
        return type(
            "Context",
            (),
            {
                "results": (),
            },
        )()


class FakeSearchProvider:
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list:
        return []


def build_workflow(
    response: str,
):
    router = AgentRouter(
        llm=FakeLLM(response),
    )

    clinic_agent = ClinicAgent(
        rag_service=FakeRagService(),
    )

    general_dental_agent = GeneralDentalAgent(
        search_provider=FakeSearchProvider(),
    )

    nodes = AgentNodes(
        router=router,
        clinic_agent=clinic_agent,
        general_dental_agent=general_dental_agent,
    )

    return create_agent_graph(
        nodes,
    )


def test_clinic_query_goes_to_clinic_node() -> None:
    workflow = build_workflow(
        "clinic",
    )

    result = workflow.invoke(
        {
            "query": "What are your clinic hours?",
        },
    )

    assert result["intent"] == AgentIntent.CLINIC


def test_general_query_goes_to_general_node() -> None:
    workflow = build_workflow(
        "general_dental",
    )

    result = workflow.invoke(
        {
            "query": "What causes tooth pain?",
        },
    )

    assert (
        result["intent"]
        == AgentIntent.GENERAL_DENTAL
    )

    assert (
        "web_results"
        in result
    )


@pytest.mark.parametrize(
    "response",
    [
        "",
        "unknown",
        "invalid",
    ],
)
def test_invalid_intent_fails(
    response: str,
) -> None:
    workflow = build_workflow(
        response,
    )

    with pytest.raises(
        ValueError,
    ):
        workflow.invoke(
            {
                "query": "Dental question",
            },
        )
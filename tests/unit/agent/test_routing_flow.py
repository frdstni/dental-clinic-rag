import pytest

from dental_rag.agent.answer import AnswerGenerator
from dental_rag.agent.clinic import ClinicAgent
from dental_rag.agent.general_dental import (
    GeneralDentalAgent,
)
from dental_rag.agent.models import AgentIntent
from dental_rag.agent.nodes import AgentNodes
from dental_rag.agent.router import AgentRouter
from dental_rag.agent.workflow import create_agent_graph
from dental_rag.web_search.models import WebSearchResult


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


class FakeAnswerLLM:
    def generate(
        self,
        prompt: str,
    ) -> str:
        return "answer"


class FakeRagService:
    def retrieve_with_quality(
        self,
        query: str,
    ):
        result = type(
            "Result",
            (),
            {
                "payload": {
                    "content": "Clinic information",
                },
            },
        )()

        return type(
            "Context",
            (),
            {
                "results": (
                    result,
                ),
            },
        )()


class FakeSearchProvider:
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title="Dental",
                content="Dental information",
                url="https://example.com",
            ),
        ]


def build_workflow(
    response: str,
):
    router = AgentRouter(
        llm=FakeLLM(response),
    )

    nodes = AgentNodes(
        router=router,
        clinic_agent=ClinicAgent(
            rag_service=FakeRagService(),
        ),
        general_dental_agent=GeneralDentalAgent(
            search_provider=FakeSearchProvider(),
        ),
        answer_generator=AnswerGenerator(
            llm=FakeAnswerLLM(),
        ),
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
            "query": "Clinic hours?",
        },
    )

    assert result["intent"] == AgentIntent.CLINIC
    assert result["answer"] == "answer"


def test_general_query_goes_to_general_node() -> None:
    workflow = build_workflow(
        "general_dental",
    )

    result = workflow.invoke(
        {
            "query": "Why tooth pain happens?",
        },
    )

    assert (
        result["intent"]
        == AgentIntent.GENERAL_DENTAL
    )

    assert (
        result["answer"]
        == "answer"
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
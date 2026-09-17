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
        response: str = "clinic",
    ) -> None:
        self.response = response

    def generate(
        self,
        prompt: str,
    ) -> str:
        return self.response


class FailingLLM:
    def generate(
        self,
        prompt: str,
    ) -> str:
        raise RuntimeError(
            "LLM failure",
        )


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


class FakeAnswerLLM:
    def generate(
        self,
        prompt: str,
    ) -> str:
        return "generated answer"


def build_workflow(
    llm,
):
    router = AgentRouter(
        llm=llm,
    )

    clinic_agent = ClinicAgent(
        rag_service=FakeRagService(),
    )

    general_dental_agent = GeneralDentalAgent(
        search_provider=FakeSearchProvider(),
    )

    answer_generator = AnswerGenerator(
        llm=FakeAnswerLLM(),
    )

    nodes = AgentNodes(
        router=router,
        clinic_agent=clinic_agent,
        general_dental_agent=general_dental_agent,
        answer_generator=answer_generator,
    )

    return create_agent_graph(
        nodes,
    )


def test_workflow_routes_clinic_query() -> None:
    workflow = build_workflow(
        FakeLLM("clinic"),
    )

    result = workflow.invoke(
        {
            "query": "Do you provide implants?",
        },
    )

    assert result["intent"] == AgentIntent.CLINIC
    assert result["answer"] == "generated answer"


def test_workflow_routes_general_dental_query() -> None:
    workflow = build_workflow(
        FakeLLM("general_dental"),
    )

    result = workflow.invoke(
        {
            "query": "What causes tooth sensitivity?",
        },
    )

    assert (
        result["intent"]
        == AgentIntent.GENERAL_DENTAL
    )

    assert result["answer"] == "generated answer"


def test_workflow_rejects_missing_query() -> None:
    workflow = build_workflow(
        FakeLLM(),
    )

    with pytest.raises(
        KeyError,
    ):
        workflow.invoke({})


@pytest.mark.parametrize(
    "response",
    [
        "",
        "unknown",
        "clinic extra text",
        "123",
    ],
)
def test_workflow_handles_invalid_router_response(
    response: str,
) -> None:
    workflow = build_workflow(
        FakeLLM(response),
    )

    with pytest.raises(
        ValueError,
    ):
        workflow.invoke(
            {
                "query": "Dental question",
            },
        )


def test_workflow_propagates_llm_failure() -> None:
    workflow = build_workflow(
        FailingLLM(),
    )

    with pytest.raises(
        RuntimeError,
    ):
        workflow.invoke(
            {
                "query": "Dental question",
            },
        )


def test_workflow_preserves_state() -> None:
    workflow = build_workflow(
        FakeLLM("clinic"),
    )

    result = workflow.invoke(
        {
            "query": "Clinic hours?",
            "confidence": 0.5,
        },
    )

    assert result["confidence"] == 1.0
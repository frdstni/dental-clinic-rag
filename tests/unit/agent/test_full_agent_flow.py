import pytest

from dental_rag.agent.answer import AnswerGenerator
from dental_rag.agent.clinic import ClinicAgent
from dental_rag.agent.general_dental import (
    GeneralDentalAgent,
)
from dental_rag.agent.models import AgentIntent
from dental_rag.agent.nodes import AgentNodes
from dental_rag.agent.router import AgentRouter
from dental_rag.agent.workflow import (
    create_agent_graph,
)
from dental_rag.web_search.models import (
    WebSearchResult,
)


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
        return "final answer"


class FakeRag:
    def retrieve_with_quality(
        self,
        query: str,
    ):
        return type(
            "Context",
            (),
            {
                "results": (
                    type(
                        "Result",
                        (),
                        {
                            "payload": {
                                "content": (
                                    "Clinic information"
                                ),
                            },
                        },
                    )(),
                ),
            },
        )()


class FakeSearch:
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title="Dental",
                content="Dental info",
                url="https://example.com",
            ),
        ]


def build_graph(
    intent: str,
):
    router = AgentRouter(
        llm=FakeLLM(intent),
    )

    nodes = AgentNodes(
        router=router,
        clinic_agent=ClinicAgent(
            FakeRag(),
        ),
        general_dental_agent=GeneralDentalAgent(
            FakeSearch(),
        ),
        answer_generator=AnswerGenerator(
            FakeAnswerLLM(),
        ),
    )

    return create_agent_graph(
        nodes,
    )


def test_clinic_route() -> None:
    graph = build_graph(
        "clinic",
    )

    result = graph.invoke(
        {
            "query": "Clinic hours",
        },
    )

    assert result["intent"] == AgentIntent.CLINIC
    assert result["answer"] == "final answer"


def test_general_dental_route() -> None:
    graph = build_graph(
        "general_dental",
    )

    result = graph.invoke(
        {
            "query": "Why tooth pain happens?",
        },
    )

    assert (
        result["intent"]
        == AgentIntent.GENERAL_DENTAL
    )

    assert len(
        result["web_results"],
    ) == 1

    assert result["answer"] == "final answer"


@pytest.mark.parametrize(
    "intent",
    [
        "",
        "invalid",
    ],
)
def test_invalid_route(
    intent: str,
) -> None:
    graph = build_graph(
        intent,
    )

    with pytest.raises(
        ValueError,
    ):
        graph.invoke(
            {
                "query": "question",
            },
        )
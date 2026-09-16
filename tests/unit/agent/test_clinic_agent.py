import pytest

from dental_rag.agent.clinic import ClinicAgent
from dental_rag.retrieval.models import (
    RetrievalResult,
)


class FakeRagService:
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results

    def retrieve_with_quality(
        self,
        query: str,
    ):
        return type(
            "Context",
            (),
            {
                "results": tuple(
                    self.results,
                ),
            },
        )()


class FailingRagService:
    def retrieve_with_quality(
        self,
        query: str,
    ):
        raise RuntimeError(
            "RAG failure",
        )


def test_clinic_agent_returns_context() -> None:
    rag = FakeRagService(
        [
            RetrievalResult(
                id="1",
                score=0.9,
                payload={
                    "content": "Implant service available",
                },
            ),
        ],
    )

    agent = ClinicAgent(
        rag_service=rag,
    )

    result = agent.run(
        {
            "query": "Do you provide implants?",
        },
    )

    assert result["context"] == [
        "Implant service available",
    ]


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n",
    ],
)
def test_clinic_agent_rejects_empty_query(
    query: str,
) -> None:
    agent = ClinicAgent(
        rag_service=FakeRagService([]),
    )

    with pytest.raises(ValueError):
        agent.run(
            {
                "query": query,
            },
        )


def test_clinic_agent_propagates_rag_failure() -> None:
    agent = ClinicAgent(
        rag_service=FailingRagService(),
    )

    with pytest.raises(RuntimeError):
        agent.run(
            {
                "query": "clinic hours",
            },
        )


def test_clinic_agent_ignores_non_string_content() -> None:
    rag = FakeRagService(
        [
            RetrievalResult(
                id="1",
                score=0.9,
                payload={
                    "content": 123,
                },
            ),
        ],
    )

    agent = ClinicAgent(
        rag_service=rag,
    )

    result = agent.run(
        {
            "query": "test",
        },
    )

    assert result["context"] == []


def test_clinic_agent_preserves_state() -> None:
    rag = FakeRagService(
        [],
    )

    agent = ClinicAgent(
        rag_service=rag,
    )

    result = agent.run(
        {
            "query": "test",
            "confidence": 0.9,
        },
    )

    assert result["confidence"] == 0.9
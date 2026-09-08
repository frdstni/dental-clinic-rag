from typing import Any

import pytest

from dental_rag.application.rag_service import RagService


class FakeRetriever:
    def __init__(
        self,
        result: list[dict[str, Any]] | None = None,
    ) -> None:
        self.received_query: str | None = None
        self.received_limit: int | None = None
        self.result = result or []

    def retrieve(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.received_query = query
        self.received_limit = limit

        return self.result


class ErrorRetriever:
    def retrieve(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        raise ValueError("invalid query")


def test_rag_service_delegates_retrieval() -> None:
    retriever = FakeRetriever(
        result=[
            {
                "content": "implant information",
                "score": 0.9,
            }
        ]
    )

    service = RagService(
        retriever=retriever,
    )

    result = service.retrieve_context(
        query="implant",
        limit=3,
    )

    assert result == [
        {
            "content": "implant information",
            "score": 0.9,
        }
    ]

    assert retriever.received_query == "implant"
    assert retriever.received_limit == 3


def test_rag_service_uses_default_limit() -> None:
    retriever = FakeRetriever()

    service = RagService(
        retriever=retriever,
    )

    service.retrieve_context(
        query="implant",
    )

    assert retriever.received_limit == 5


def test_rag_service_returns_empty_results() -> None:
    retriever = FakeRetriever()

    service = RagService(
        retriever=retriever,
    )

    result = service.retrieve_context(
        query="unknown",
    )

    assert result == []


def test_rag_service_propagates_retriever_errors() -> None:
    service = RagService(
        retriever=ErrorRetriever(),
    )

    with pytest.raises(ValueError):
        service.retrieve_context(
            query="",
        )
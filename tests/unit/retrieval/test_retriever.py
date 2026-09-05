from typing import Any

import pytest

from dental_rag.retrieval.retriever import Retriever


class FakeEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [0.1, 0.2, 0.3]
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.received_vector: list[float] | None = None
        self.received_limit: int | None = None

    def search(
        self,
        vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.received_vector = vector
        self.received_limit = limit

        return [
            {
                "content": "Dental implant information",
                "score": 0.95,
            }
        ]


def test_retriever_embeds_query_and_searches() -> None:
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="implant cost",
        limit=3,
    )

    assert results == [
        {
            "content": "Dental implant information",
            "score": 0.95,
        }
    ]

    assert vector_store.received_vector == [
        0.1,
        0.2,
        0.3,
    ]

    assert vector_store.received_limit == 3

def test_retriever_rejects_empty_query() -> None:
    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            query="",
        )

def test_retriever_returns_empty_result_when_no_match() -> None:
    class EmptyVectorStore:
        def search(
            self,
            vector: list[float],
            limit: int,
        ) -> list[dict[str, object]]:
            return []

    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=EmptyVectorStore(),
    )

    result = retriever.retrieve(
        query="unknown question",
    )

    assert result == []

def test_retriever_rejects_invalid_limit() -> None:
    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            query="implant",
            limit=0,
        )
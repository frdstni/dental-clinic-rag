import pytest

from dental_rag.retrieval.models import (
    RetrievalResult,
)
from dental_rag.retrieval.retriever import (
    Retriever,
)


class FakeEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [
                0.1,
                0.2,
                0.3,
            ]
        ]


class FakeVectorStore:
    def search(
        self,
        vector: list[float],
        limit: int,
    ) -> list[dict[str, object]]:
        return [
            {
                "id": "doc-1",
                "score": 0.85,
                "payload": {
                    "text": "implant information",
                },
            }
        ]


@pytest.fixture
def retriever() -> Retriever:
    return Retriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
    )


def test_retriever_embeds_query_and_searches(
    retriever: Retriever,
) -> None:
    results = retriever.retrieve(
        "implant",
    )

    assert len(results) == 1

    assert isinstance(
        results[0],
        RetrievalResult,
    )

    assert results[0].id == "doc-1"

    assert results[0].score == 0.85

    assert results[0].payload == {
        "text": "implant information",
    }


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_retriever_rejects_empty_query(
    retriever: Retriever,
    query: str,
) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve(query)


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        -5,
    ],
)
def test_retriever_rejects_invalid_limit(
    retriever: Retriever,
    limit: int,
) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve(
            "implant",
            limit=limit,
        )


def test_retriever_returns_empty_results(
    retriever: Retriever,
) -> None:
    retriever.vector_store.search = (
        lambda vector, limit: []
    )

    results = retriever.retrieve(
        "implant",
    )

    assert results == []
import pytest

from dental_rag.retrieval.mmr import (
    MMRSelector,
)
from dental_rag.retrieval.mmr_models import (
    MMRDocument,
)


def create_document(
    doc_id: str,
    embedding: list[float],
) -> MMRDocument:
    return MMRDocument(
        id=doc_id,
        content=f"content {doc_id}",
        embedding=embedding,
    )


def test_mmr_selects_relevant_documents() -> None:
    selector = MMRSelector(
        lambda_value=0.5,
    )

    documents = [
        create_document(
            "1",
            [1.0, 0.0],
        ),
        create_document(
            "2",
            [0.0, 1.0],
        ),
    ]

    result = selector.select(
        query_embedding=[1.0, 0.0],
        documents=documents,
        limit=1,
    )

    assert result[0].id == "1"


def test_mmr_respects_limit() -> None:
    selector = MMRSelector()

    documents = [
        create_document(
            "1",
            [1.0, 0.0],
        ),
        create_document(
            "2",
            [0.0, 1.0],
        ),
    ]

    result = selector.select(
        query_embedding=[1.0, 0.0],
        documents=documents,
        limit=1,
    )

    assert len(result) == 1


def test_mmr_returns_empty_for_empty_documents() -> None:
    selector = MMRSelector()

    assert selector.select(
        query_embedding=[1.0, 0.0],
        documents=[],
    ) == []


def test_mmr_rejects_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        MMRSelector(
            lambda_value=1.5,
        )


def test_mmr_rejects_invalid_limit() -> None:
    selector = MMRSelector()

    with pytest.raises(ValueError):
        selector.select(
            query_embedding=[1.0, 0.0],
            documents=[],
            limit=0,
        )


def test_mmr_rejects_different_embedding_dimensions() -> None:
    selector = MMRSelector()

    documents = [
        create_document(
            "1",
            [1.0, 0.0],
        )
    ]

    with pytest.raises(ValueError):
        selector.select(
            query_embedding=[1.0, 0.0, 1.0],
            documents=documents,
        )


def test_mmr_handles_zero_embeddings() -> None:
    selector = MMRSelector()

    documents = [
        create_document(
            "1",
            [0.0, 0.0],
        )
    ]

    result = selector.select(
        query_embedding=[0.0, 0.0],
        documents=documents,
    )

    assert len(result) == 1
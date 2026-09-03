from unittest.mock import patch

import pytest

from dental_rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


def test_create_collection_calls_qdrant_correctly() -> None:
    with patch(
        "dental_rag.vector_store.qdrant_store.QdrantClient"
    ) as mock_qdrant:

        store = QdrantVectorStore(
            collection_name="clinic_documents"
        )

        store.create_collection(
            vector_size=1536
        )

    mock_qdrant.return_value.create_collection.assert_called_once()

    call_kwargs = (
        mock_qdrant.return_value
        .create_collection
        .call_args
        .kwargs
    )

    assert call_kwargs["collection_name"] == (
        "clinic_documents"
    )

    assert call_kwargs["vectors_config"].size == 1536

def test_add_vectors_calls_upsert_with_correct_points() -> None:
    with patch(
        "dental_rag.vector_store.qdrant_store.QdrantClient"
    ) as mock_qdrant:

        store = QdrantVectorStore(
            collection_name="clinic_documents"
        )

        store.add(
            vectors=[
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            payloads=[
                {"text": "clinic hours"},
                {"text": "dental services"},
            ],
        )

    mock_qdrant.return_value.upsert.assert_called_once()

    call_kwargs = (
        mock_qdrant.return_value
        .upsert
        .call_args
        .kwargs
    )

    assert call_kwargs["collection_name"] == (
        "clinic_documents"
    )

    points = call_kwargs["points"]

    assert len(points) == 2

    assert points[0].id == 0
    assert points[0].vector == [0.1, 0.2]
    assert points[0].payload == {
        "text": "clinic hours"
    }

    assert points[1].id == 1

def test_add_vectors_fails_when_vectors_and_payloads_length_differ() -> None:
    with patch(
        "dental_rag.vector_store.qdrant_store.QdrantClient"
    ):
        store = QdrantVectorStore(
            collection_name="clinic_documents"
        )

        with pytest.raises(ValueError):
            store.add(
                vectors=[
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
                payloads=[
                    {"text": "only one"},
                ],
            )
def test_add_empty_vectors_does_not_call_qdrant() -> None:
    with patch(
        "dental_rag.vector_store.qdrant_store.QdrantClient"
    ) as mock_qdrant:

        store = QdrantVectorStore(
            collection_name="clinic_documents"
        )

        store.add(
            vectors=[],
            payloads=[],
        )

    mock_qdrant.return_value.upsert.assert_not_called()


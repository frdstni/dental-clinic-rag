from pathlib import Path

from qdrant_client import QdrantClient

from dental_rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


def test_qdrant_store_adds_vectors_in_local_storage(
    tmp_path: Path,
) -> None:
    collection_name = "test_collection"

    store = QdrantVectorStore(
        collection_name=collection_name
    )

    store.client = QdrantClient(
        path=str(tmp_path)
    )

    store.create_collection(
        vector_size=2
    )

    store.add(
        vectors=[
            [0.1, 0.2],
        ],
        payloads=[
            {
                "text": "clinic hours",
            }
        ],
    )

    points = store.client.retrieve(
        collection_name=collection_name,
        ids=[0],
    )

    assert len(points) == 1

    assert points[0].payload == {
        "text": "clinic hours"
    }
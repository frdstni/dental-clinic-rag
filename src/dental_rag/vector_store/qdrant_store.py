from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from dental_rag.vector_store.base import VectorStore


class QdrantVectorStore(VectorStore):
    """
    Qdrant implementation of vector storage.
    """

    def __init__(
        self,
        collection_name: str,
    ) -> None:
        self.client = QdrantClient(
            path="qdrant_data"
        )

        self.collection_name = collection_name

    def create_collection(
        self,
        vector_size: int,
    ) -> None:
         self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
        ),
    )

    def add(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        
        if not vectors:
            return

        points = [
             PointStruct(
                 id=index,
                 vector=vector,
                 payload=payload,
             )
            for index, (vector, payload)
            in enumerate(zip(vectors, payloads, strict=True))
        ]

        self.client.upsert(
             collection_name=self.collection_name,
             points=points,
        )
    def search(
        self,
        vector: list[float],
       limit: int,
    ) -> list[dict[str, Any]]:
        results = self.client.query_points(
              collection_name=self.collection_name,
              query=vector,
              limit=limit,
        )

        return [
           {
            "id": result.id,
            "score": result.score,
            "payload": result.payload or {},
           }
           for result in results.points
    ]
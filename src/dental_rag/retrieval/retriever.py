from typing import Any

from dental_rag.embeddings.base import EmbeddingModel
from dental_rag.vector_store.base import VectorStore


class Retriever:
    """
    Retrieves relevant documents from vector store.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:

         if not query.strip():
             raise ValueError("Query cannot be empty")

         if limit <= 0:
             raise ValueError(
                  "Limit must be greater than zero"
            )

         query_vector = self.embedding_model.embed(
             [query]
        )[0]

         return self.vector_store.search(
            vector=query_vector,
            limit=limit,
        )
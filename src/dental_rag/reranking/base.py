from typing import Protocol

from dental_rag.retrieval.models import (
    RetrievalResult,
)


class Reranker(Protocol):
    """
    Interface for reranking retrieved documents.
    """

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        limit: int = 5,
    ) -> list[RetrievalResult]:
        ...
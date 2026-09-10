from typing import Protocol

from dental_rag.retrieval.models import (
    RetrievalResult,
)


class RetrievalBackend(Protocol):
    """
    Interface for retrieval implementations.
    """

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        ...
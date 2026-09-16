from dental_rag.reranking.base import (
    Reranker,
)
from dental_rag.retrieval.base import (
    RetrievalBackend,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)


class RetrievalPipeline:
    """
    Executes retrieval and optional reranking.
    """

    def __init__(
        self,
        retriever: RetrievalBackend,
        reranker: Reranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty",
            )

        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero",
            )

        results = self.retriever.retrieve(
            query=query,
            limit=limit,
        )

        if self.reranker is None:
            return results[:limit]

        return self.reranker.rerank(
            query=query,
            results=results,
            limit=limit,
        )
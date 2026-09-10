from dental_rag.retrieval.fusion import (
    ReciprocalRankFusion,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)
from dental_rag.retrieval.retriever import (
    Retriever,
)
from dental_rag.retrieval.sparse_retriever import (
    SparseRetriever,
)


class HybridRetriever:
    """
    Combines dense and sparse retrieval.
    """

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: SparseRetriever,
        fusion: ReciprocalRankFusion,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero"
            )

        dense_results = self.dense_retriever.retrieve(
            query=query,
            limit=limit,
        )

        sparse_results = self.sparse_retriever.retrieve(
            query=query,
            limit=limit,
        )

        return self.fusion.fuse(
            result_lists=[
                dense_results,
                sparse_results,
            ],
            limit=limit,
        )
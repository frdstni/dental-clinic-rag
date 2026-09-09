from dental_rag.application.models import (
    RetrievalContext,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)
from dental_rag.retrieval.quality_checker import (
    RetrievalQualityChecker,
)
from dental_rag.retrieval.retriever import (
    Retriever,
)


class RagService:
    def __init__(
        self,
        retriever: Retriever,
        quality_checker: RetrievalQualityChecker,
    ) -> None:
        self.retriever = retriever
        self.quality_checker = quality_checker

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        return self.retriever.retrieve(
            query=query,
            limit=limit,
        )

    def retrieve_with_quality(
        self,
        query: str,
        limit: int = 5,
    ) -> RetrievalContext:
        results = self.retrieve_context(
            query=query,
            limit=limit,
        )

        quality = self.quality_checker.check(
            results,
        )

        return RetrievalContext(
            results=tuple(results),
            quality=quality,
        )
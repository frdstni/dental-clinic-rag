from dental_rag.application.models import (
    RetrievalContext,
)
from dental_rag.query_processing.refiners.base import (
    QueryRefiner,
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
        refiner: QueryRefiner | None = None,
    ) -> None:
        self.retriever = retriever
        self.quality_checker = quality_checker
        self.refiner = refiner

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

    def retrieve_with_refinement(
        self,
        query: str,
        limit: int = 5,
    ) -> RetrievalContext:
        initial_results = self.retrieve_context(
            query=query,
            limit=limit,
        )

        initial_quality = self.quality_checker.check(
            initial_results,
        )

        if initial_quality.passed:
            return RetrievalContext(
                results=tuple(initial_results),
                quality=initial_quality,
            )

        if self.refiner is None:
            raise ValueError(
                "refiner is required when retrieval quality fails"
            )

        refined_query = self.refiner.refine(
            query,
        )

        if not refined_query.strip():
            raise ValueError(
                "refined query cannot be empty"
            )

        refined_results = self.retrieve_context(
            query=refined_query,
            limit=limit,
        )

        refined_quality = self.quality_checker.check(
            refined_results,
        )

        return RetrievalContext(
            results=tuple(refined_results),
            quality=refined_quality,
        )
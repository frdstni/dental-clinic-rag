from dental_rag.application.models import (
    RetrievalContext,
)
from dental_rag.query_processing.pipeline import (
    QueryPipeline,
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
        query_pipeline: QueryPipeline | None = None,
        refiner: QueryRefiner | None = None,
    ) -> None:
        self.retriever = retriever
        self.quality_checker = quality_checker
        self.query_pipeline = query_pipeline
        self.refiner = refiner

    def _process_query(
        self,
        query: str,
    ) -> tuple[str, ...]:
        if self.query_pipeline is None:
            return (query,)

        return self.query_pipeline.process(
            query,
        )

    def _deduplicate_results(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        seen_ids: set[str] = set()
        unique_results: list[RetrievalResult] = []

        for result in results:
            if result.id in seen_ids:
                continue

            seen_ids.add(result.id)
            unique_results.append(result)

        return unique_results

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        queries = self._process_query(
            query,
        )

        results: list[RetrievalResult] = []

        for processed_query in queries:
            results.extend(
                self.retriever.retrieve(
                    query=processed_query,
                    limit=limit,
                )
            )

        return self._deduplicate_results(
            results,
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
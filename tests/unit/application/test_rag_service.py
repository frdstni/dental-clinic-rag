from collections.abc import Sequence

import pytest

from dental_rag.application.rag_service import (
    RagService,
)
from dental_rag.retrieval.models import (
    RetrievalQuality,
    RetrievalResult,
)


class FakeRetriever:
    def __init__(
        self,
        results: list[list[RetrievalResult]]
        | None = None,
    ) -> None:
        self.received_queries: list[str] = []
        self.received_limits: list[int] = []

        self.results = (
            results
            if results is not None
            else [[]]
        )

    def retrieve(
        self,
        query: str,
        limit: int,
    ) -> list[RetrievalResult]:
        self.received_queries.append(query)
        self.received_limits.append(limit)

        return self.results.pop(0)


class ErrorRetriever:
    def retrieve(
        self,
        query: str,
        limit: int,
    ) -> list[RetrievalResult]:
        raise ValueError(
            "invalid query"
        )


class FakeQualityChecker:
    def __init__(
        self,
        quality: RetrievalQuality,
    ) -> None:
        self.quality = quality
        self.received_results: (
            Sequence[RetrievalResult] | None
        ) = None

    def check(
        self,
        results: Sequence[RetrievalResult],
    ) -> RetrievalQuality:
        self.received_results = results

        return self.quality


class SequentialQualityChecker:
    def __init__(
        self,
        qualities: list[RetrievalQuality],
    ) -> None:
        self.qualities = qualities
        self.calls = 0

    def check(
        self,
        results: Sequence[RetrievalResult],
    ) -> RetrievalQuality:
        quality = self.qualities[self.calls]

        self.calls += 1

        return quality


class ErrorQualityChecker:
    def check(
        self,
        results: Sequence[RetrievalResult],
    ) -> RetrievalQuality:
        raise RuntimeError(
            "quality check failed"
        )


class FakeRefiner:
    def __init__(
        self,
        refined_query: str = "refined query",
    ) -> None:
        self.received_query: str | None = None
        self.refined_query = refined_query

    def refine(
        self,
        query: str,
    ) -> str:
        self.received_query = query

        return self.refined_query


class ErrorRefiner:
    def refine(
        self,
        query: str,
    ) -> str:
        raise RuntimeError(
            "refinement failed"
        )
class FakeQueryPipeline:
    def __init__(
        self,
        processed_queries: tuple[str, ...],
    ) -> None:
        self.processed_queries = processed_queries
        self.received_queries: list[str] = []

    def process(
        self,
        query: str,
    ) -> tuple[str, ...]:
        self.received_queries.append(query)

        return self.processed_queries


def make_result(
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        id="doc-1",
        score=score,
        payload={
            "content": "implant information",
        },
    )


def make_quality(
    passed: bool = True,
) -> RetrievalQuality:
    return RetrievalQuality(
        passed=passed,
        reason=(
            "sufficient retrieval score"
            if passed
            else "low retrieval score"
        ),
    )


def test_rag_service_delegates_retrieval() -> None:
    expected_result = make_result()

    retriever = FakeRetriever(
        results=[
            [
                expected_result,
            ]
        ],
    )

    service = RagService(
        retriever=retriever,
        quality_checker=FakeQualityChecker(
            quality=make_quality(),
        ),
    )

    result = service.retrieve_context(
        query="implant",
        limit=3,
    )

    assert result == [
        expected_result,
    ]

    assert retriever.received_queries == [
        "implant",
    ]

    assert retriever.received_limits == [
        3,
    ]


def test_rag_service_uses_default_limit() -> None:
    retriever = FakeRetriever()

    service = RagService(
        retriever=retriever,
        quality_checker=FakeQualityChecker(
            quality=make_quality(),
        ),
    )

    service.retrieve_context(
        query="implant",
    )

    assert retriever.received_limits == [
        5,
    ]


def test_rag_service_returns_empty_results() -> None:
    retriever = FakeRetriever()

    service = RagService(
        retriever=retriever,
        quality_checker=FakeQualityChecker(
            quality=make_quality(),
        ),
    )

    result = service.retrieve_context(
        query="unknown",
    )

    assert result == []


def test_rag_service_propagates_retriever_errors() -> None:
    service = RagService(
        retriever=ErrorRetriever(),
        quality_checker=FakeQualityChecker(
            quality=make_quality(),
        ),
    )

    with pytest.raises(ValueError):
        service.retrieve_context(
            query="",
        )


def test_retrieve_with_quality_returns_context() -> None:
    retrieved_result = make_result()

    retriever = FakeRetriever(
        results=[
            [
                retrieved_result,
            ]
        ],
    )

    expected_quality = make_quality()

    quality_checker = FakeQualityChecker(
        quality=expected_quality,
    )

    service = RagService(
        retriever=retriever,
        quality_checker=quality_checker,
    )

    context = service.retrieve_with_quality(
        query="implant",
        limit=3,
    )

    assert context.results == (
        retrieved_result,
    )

    assert context.quality == expected_quality

    assert quality_checker.received_results == [
        retrieved_result,
    ]


def test_retrieve_with_quality_delegates_query_and_limit() -> None:
    retriever = FakeRetriever(
        results=[
            [
                make_result(),
            ]
        ],
    )

    service = RagService(
        retriever=retriever,
        quality_checker=FakeQualityChecker(
            quality=make_quality(),
        ),
    )

    service.retrieve_with_quality(
        query="dental implant",
        limit=7,
    )

    assert retriever.received_queries == [
        "dental implant",
    ]

    assert retriever.received_limits == [
        7,
    ]


def test_retrieve_with_quality_checks_empty_results() -> None:
    retriever = FakeRetriever()

    quality_checker = FakeQualityChecker(
        quality=make_quality(
            passed=False,
        ),
    )

    service = RagService(
        retriever=retriever,
        quality_checker=quality_checker,
    )

    context = service.retrieve_with_quality(
        query="unknown",
    )

    assert context.results == ()

    assert context.quality.passed is False

    assert quality_checker.received_results == []


def test_retrieve_with_quality_propagates_checker_errors() -> None:
    service = RagService(
        retriever=FakeRetriever(
            results=[
                [
                    make_result(),
                ]
            ],
        ),
        quality_checker=ErrorQualityChecker(),
    )

    with pytest.raises(
        RuntimeError,
        match="quality check failed",
    ):
        service.retrieve_with_quality(
            query="implant",
        )


def test_retrieve_with_refinement_skips_refiner_when_quality_passes() -> None:
    refiner = FakeRefiner()

    service = RagService(
        retriever=FakeRetriever(
            results=[
                [
                    make_result(),
                ]
            ],
        ),
        quality_checker=FakeQualityChecker(
            quality=make_quality(
                passed=True,
            ),
        ),
        refiner=refiner,
    )

    context = service.retrieve_with_refinement(
        query="implant",
    )

    assert context.quality.passed is True

    assert refiner.received_query is None


def test_retrieve_with_refinement_uses_refined_query() -> None:
    retriever = FakeRetriever(
        results=[
            [],
            [
                make_result(),
            ],
        ],
    )

    refiner = FakeRefiner(
        refined_query="implant complications",
    )

    quality_checker = SequentialQualityChecker(
        qualities=[
            make_quality(
                passed=False,
            ),
            make_quality(
                passed=True,
            ),
        ],
    )

    service = RagService(
        retriever=retriever,
        quality_checker=quality_checker,
        refiner=refiner,
    )

    context = service.retrieve_with_refinement(
        query="implant",
    )

    assert context.quality.passed is True

    assert retriever.received_queries == [
        "implant",
        "implant complications",
    ]

    assert refiner.received_query == "implant"


def test_retrieve_with_refinement_propagates_refiner_error() -> None:
    service = RagService(
        retriever=FakeRetriever(
            results=[
                []
            ],
        ),
        quality_checker=FakeQualityChecker(
            quality=make_quality(
                passed=False,
            ),
        ),
        refiner=ErrorRefiner(),
    )

    with pytest.raises(
        RuntimeError,
        match="refinement failed",
    ):
        service.retrieve_with_refinement(
            query="implant",
        )


def test_retrieve_with_refinement_requires_refiner_on_failed_quality() -> None:
    service = RagService(
        retriever=FakeRetriever(
            results=[
                []
            ],
        ),
        quality_checker=FakeQualityChecker(
            quality=make_quality(
                passed=False,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="refiner is required",
    ):
        service.retrieve_with_refinement(
            query="implant",
        )


def test_retrieve_with_refinement_rejects_empty_refined_query() -> None:
    service = RagService(
        retriever=FakeRetriever(
            results=[
                []
            ],
        ),
        quality_checker=FakeQualityChecker(
            quality=make_quality(
                passed=False,
            ),
        ),
        refiner=FakeRefiner(
            refined_query="   ",
        ),
    )

    with pytest.raises(
        ValueError,
        match="refined query cannot be empty",
    ):
        service.retrieve_with_refinement(
            query="implant",
        )
def test_rag_service_uses_query_pipeline() -> None:
    first_result = make_result()

    retriever = FakeRetriever(
        results=[
            [
                first_result,
            ],
            [
                first_result,
            ],
        ],
    )

    query_pipeline = FakeQueryPipeline(
        processed_queries=(
            "implant",
            "implant risks",
        ),
    )

    service = RagService(
        retriever=retriever,
        quality_checker=FakeQualityChecker(
            quality=make_quality(),
        ),
        query_pipeline=query_pipeline,
    )

    result = service.retrieve_context(
        query="implant",
    )

    assert query_pipeline.received_queries == [
        "implant",
    ]

    assert retriever.received_queries == [
        "implant",
        "implant risks",
    ]

    assert result == [
        first_result,
    ]
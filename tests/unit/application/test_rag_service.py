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
        result: list[RetrievalResult] | None = None,
    ) -> None:
        self.received_query: str | None = None
        self.received_limit: int | None = None
        self.result = (
            result
            if result is not None
            else []
        )

    def retrieve(
        self,
        query: str,
        limit: int,
    ) -> list[RetrievalResult]:
        self.received_query = query
        self.received_limit = limit

        return self.result


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


class ErrorQualityChecker:
    def check(
        self,
        results: Sequence[RetrievalResult],
    ) -> RetrievalQuality:
        raise RuntimeError(
            "quality check failed"
        )


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
        result=[
            expected_result,
        ]
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

    assert retriever.received_query == "implant"
    assert retriever.received_limit == 3


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

    assert retriever.received_limit == 5


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
        result=[
            retrieved_result,
        ]
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
        result=[
            make_result(),
        ]
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

    assert (
        retriever.received_query
        == "dental implant"
    )
    assert retriever.received_limit == 7


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
            result=[
                make_result(),
            ]
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
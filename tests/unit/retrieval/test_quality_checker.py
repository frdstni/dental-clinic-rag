import pytest

from dental_rag.retrieval.models import (
    RetrievalResult,
)
from dental_rag.retrieval.quality_checker import (
    RetrievalQualityChecker,
)


def make_result(
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        id="doc-1",
        score=score,
        payload={},
    )


def test_high_score_results_pass() -> None:
    checker = RetrievalQualityChecker()

    result = checker.check(
        [
            make_result(0.8),
        ]
    )

    assert result.passed is True


def test_low_score_results_fail() -> None:
    checker = RetrievalQualityChecker()

    result = checker.check(
        [
            make_result(0.2),
        ]
    )

    assert result.passed is False


def test_empty_results_fail() -> None:
    checker = RetrievalQualityChecker()

    result = checker.check([])

    assert result.passed is False


def test_best_score_is_used() -> None:
    checker = RetrievalQualityChecker()

    result = checker.check(
        [
            make_result(0.2),
            make_result(0.8),
            make_result(0.3),
        ]
    )

    assert result.passed is True


def test_score_equal_to_threshold_passes() -> None:
    checker = RetrievalQualityChecker(
        minimum_score=0.5,
    )

    result = checker.check(
        [
            make_result(0.5),
        ]
    )

    assert result.passed is True


def test_zero_score_fails() -> None:
    checker = RetrievalQualityChecker()

    result = checker.check(
        [
            make_result(0),
        ]
    )

    assert result.passed is False


def test_custom_threshold() -> None:
    checker = RetrievalQualityChecker(
        minimum_score=0.9,
    )

    result = checker.check(
        [
            make_result(0.85),
        ]
    )

    assert result.passed is False


def test_negative_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        RetrievalQualityChecker(
            minimum_score=-0.1,
        )
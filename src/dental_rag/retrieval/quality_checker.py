from collections.abc import Sequence

from dental_rag.retrieval.models import (
    RetrievalQuality,
    RetrievalResult,
)


class RetrievalQualityChecker:
    """
    Evaluates the quality of retrieved results.
    """

    def __init__(
        self,
        minimum_score: float = 0.5,
    ) -> None:
        if minimum_score < 0:
            raise ValueError(
                "minimum score cannot be negative"
            )

        self.minimum_score = minimum_score

    def check(
        self,
        results: Sequence[RetrievalResult],
    ) -> RetrievalQuality:
        if not results:
            return RetrievalQuality(
                passed=False,
                reason="no retrieval results",
            )

        best_score = max(
            result.score
            for result in results
        )

        if best_score < self.minimum_score:
            return RetrievalQuality(
                passed=False,
                reason="low retrieval score",
            )

        return RetrievalQuality(
            passed=True,
            reason="sufficient retrieval score",
        )
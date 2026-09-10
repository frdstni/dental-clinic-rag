from collections import defaultdict

from dental_rag.retrieval.models import (
    RetrievalResult,
)


class ReciprocalRankFusion:
    """
    Combines ranked retrieval results using RRF.
    """

    def __init__(
        self,
        k: int = 60,
    ) -> None:
        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        self.k = k

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        scores: dict[str, float] = defaultdict(float)
        results_by_id: dict[str, RetrievalResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                scores[result.id] += 1 / (
                    self.k + rank
                )

                results_by_id[result.id] = result

        ranked_ids = sorted(
            scores,
            key=lambda item: scores[item],
            reverse=True,
        )

        return [
            RetrievalResult(
                id=result_id,
                score=scores[result_id],
                payload=results_by_id[result_id].payload,
            )
            for result_id in ranked_ids[:limit]
        ]
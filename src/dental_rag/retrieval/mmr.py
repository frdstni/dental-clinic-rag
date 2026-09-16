import math

from dental_rag.retrieval.mmr_models import (
    MMRDocument,
)


class MMRSelector:
    """
    Selects diverse documents using Maximal Marginal Relevance.
    """

    def __init__(
        self,
        lambda_value: float = 0.5,
    ) -> None:
        if not 0 <= lambda_value <= 1:
            raise ValueError(
                "lambda_value must be between 0 and 1",
            )

        self.lambda_value = lambda_value

    def select(
        self,
        query_embedding: list[float],
        documents: list[MMRDocument],
        limit: int = 5,
    ) -> list[MMRDocument]:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero",
            )

        if not documents:
            return []

        selected: list[MMRDocument] = []
        remaining = documents.copy()

        while remaining and len(selected) < limit:
            if not selected:
                best = max(
                    remaining,
                    key=lambda doc: self._cosine_similarity(
                        query_embedding,
                        doc.embedding,
                    ),
                )
            else:
                best = max(
                    remaining,
                    key=lambda doc: self._mmr_score(
                        query_embedding,
                        doc,
                        selected,
                    ),
                )

            selected.append(best)
            remaining.remove(best)

        return selected

    def _mmr_score(
        self,
        query_embedding: list[float],
        document: MMRDocument,
        selected: list[MMRDocument],
    ) -> float:
        relevance = self._cosine_similarity(
            query_embedding,
            document.embedding,
        )

        diversity = max(
            self._cosine_similarity(
                document.embedding,
                selected_doc.embedding,
            )
            for selected_doc in selected
        )

        return (
            self.lambda_value * relevance
            - (1 - self.lambda_value) * diversity
        )

    @staticmethod
    def _cosine_similarity(
        first: list[float],
        second: list[float],
    ) -> float:
        if len(first) != len(second):
            raise ValueError(
                "embeddings must have the same dimension",
            )

        denominator = (
            math.sqrt(sum(value * value for value in first))
            * math.sqrt(sum(value * value for value in second))
        )

        if denominator == 0:
            return 0.0

        return sum(
            first[index] * second[index]
            for index in range(len(first))
        ) / denominator
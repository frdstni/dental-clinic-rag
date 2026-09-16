from sentence_transformers import CrossEncoder

from dental_rag.retrieval.models import (
    RetrievalResult,
)


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross encoder model.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model = CrossEncoder(
            model_name,
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError(
                "query cannot be empty",
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero",
            )

        if not results:
            return []

        pairs: list[tuple[str, str]] = []

        for result in results:
            content = result.payload.get(
                "content",
                "",
            )

            if not isinstance(content, str):
                raise TypeError(
                    "payload content must be a string",
                )

            pairs.append(
                (
                    query,
                    content,
                )
            )

        scores = self.model.predict(
            pairs,
        )

        reranked_results = [
            RetrievalResult(
                id=result.id,
                score=float(score),
                payload=result.payload,
            )
            for result, score in zip(
                results,
                scores,
                strict=True,
            )
        ]

        return sorted(
            reranked_results,
            key=lambda result: result.score,
            reverse=True,
        )[:limit]
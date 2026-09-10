
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from dental_rag.retrieval.models import (
    RetrievalResult,
)


class SparseRetriever:
    """
    Sparse retrieval using BM25.
    """

    def __init__(
        self,
        documents: list[dict[str, object]],
    ) -> None:
        if not documents:
            raise ValueError(
                "documents cannot be empty"
            )

        self.documents = documents

        tokenized_documents = [
            self._tokenize(
                str(document["content"])
            )
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents,
        )

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero"
            )

        query_tokens = self._tokenize(
            query,
        )

        scores = self.bm25.get_scores(
            query_tokens,
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[RetrievalResult] = []

        for index in ranked_indexes[:limit]:
            document = self.documents[index]

            results.append(
                RetrievalResult(
                    id=str(document["id"]),
                    score=float(scores[index]),
                    payload=document,
                )
            )

        return results
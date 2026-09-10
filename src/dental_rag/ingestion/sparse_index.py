from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from dental_rag.domain.models import (
    DocumentChunk,
)


class SparseIndex:
    """
    In-memory BM25 index built from document chunks.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "chunks cannot be empty"
            )

        self.chunks = chunks

        tokenized_documents = [
            self._tokenize(
                chunk.content
            )
            for chunk in chunks
        ]

        self.index = BM25Okapi(
            tokenized_documents,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        scores = self.index.get_scores(
            self._tokenize(query),
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            (
                self.chunks[index],
                float(scores[index]),
            )
            for index in ranked_indexes[:limit]
        ]

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        return text.lower().split()
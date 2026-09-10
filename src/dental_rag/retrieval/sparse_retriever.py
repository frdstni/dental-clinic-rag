
from dental_rag.domain.models import (
    DocumentChunk,
)
from dental_rag.ingestion.sparse_index import (
    SparseIndex,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)


class SparseRetriever:
    """
    Retrieval adapter over SparseIndex.
    """

    def __init__(
        self,
        index: SparseIndex,
    ) -> None:
        self.index = index

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        results = self.index.search(
            query=query,
            limit=limit,
        )

        return [
            RetrievalResult(
                id=self._create_id(chunk),
                score=score,
                payload={
                    "content": chunk.content,
                    "source": chunk.metadata.file_name,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, score in results
        ]

    def _create_id(
         self,
         chunk: DocumentChunk,
    ) -> str:
        return (
            f"{chunk.metadata.file_name}:"
            f"{chunk.chunk_index}"
        )
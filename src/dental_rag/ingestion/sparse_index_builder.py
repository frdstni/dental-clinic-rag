from dental_rag.domain.models import (
    DocumentChunk,
)
from dental_rag.ingestion.sparse_index import (
    SparseIndex,
)


class SparseIndexBuilder:
    """
    Builds sparse indexes from document chunks.
    """

    def build(
        self,
        chunks: list[DocumentChunk],
    ) -> SparseIndex:
        return SparseIndex(
            chunks=chunks,
        )
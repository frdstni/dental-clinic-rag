from dental_rag.ingestion.sparse_index import (
    SparseIndex,
)


class RetrievalResources:
    """
    Runtime resources required by retrieval layer.
    """

    def __init__(
        self,
        sparse_index: SparseIndex,
    ) -> None:
        self.sparse_index = sparse_index
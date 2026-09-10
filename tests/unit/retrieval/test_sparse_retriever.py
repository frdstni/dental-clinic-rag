from pathlib import Path

import pytest

from dental_rag.domain.models import (
    DocumentChunk,
    DocumentMetadata,
)
from dental_rag.ingestion.sparse_index import (
    SparseIndex,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)
from dental_rag.retrieval.sparse_retriever import (
    SparseRetriever,
)


def create_chunk(
    content: str,
    index: int,
) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        metadata=DocumentMetadata(
           source_path=Path("implant.txt"),
        ),
        chunk_index=index,
        start_sentence_index=0,
        end_sentence_index_exclusive=1,
    )


def create_retriever() -> SparseRetriever:
    chunks = [
        create_chunk(
            "dental implant surgery information",
            0,
        ),
        create_chunk(
            "tooth whitening procedure",
            1,
        ),
    ]

    index = SparseIndex(
        chunks=chunks,
    )

    return SparseRetriever(
        index=index,
    )


def test_sparse_retriever_returns_relevant_documents() -> None:
    retriever = create_retriever()

    results = retriever.retrieve(
        query="dental implant",
        limit=1,
    )

    assert len(results) == 1

    assert isinstance(
        results[0],
        RetrievalResult,
    )


def test_sparse_retriever_rejects_empty_query() -> None:
    retriever = create_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_sparse_retriever_rejects_invalid_limit() -> None:
    retriever = create_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve(
            "implant",
            limit=0,
        )
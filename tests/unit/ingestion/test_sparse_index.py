from pathlib import Path

import pytest

from dental_rag.domain.models import (
    DocumentChunk,
    DocumentMetadata,
)
from dental_rag.ingestion.sparse_index import (
    SparseIndex,
)


def create_chunk(
    content: str,
    index: int,
) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        metadata=DocumentMetadata(
            source_path=Path("test.txt"),
        ),
        chunk_index=index,
        start_sentence_index=0,
        end_sentence_index_exclusive=1,
    )


def test_sparse_index_returns_ranked_chunks() -> None:
    chunks = [
        create_chunk(
            "dental implant surgery",
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

    results = index.search(
        query="dental implant",
        limit=1,
    )

    assert len(results) == 1

    chunk, score = results[0]

    assert chunk.chunk_index == 0
    assert isinstance(score, float)


def test_sparse_index_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError):
        SparseIndex(
            chunks=[],
        )


def test_sparse_index_rejects_empty_query() -> None:
    index = SparseIndex(
        chunks=[
            create_chunk(
                "dental implant",
                0,
            )
        ],
    )

    with pytest.raises(ValueError):
        index.search(
            query="",
        )


def test_sparse_index_rejects_invalid_limit() -> None:
    index = SparseIndex(
        chunks=[
            create_chunk(
                "dental implant",
                0,
            )
        ],
    )

    with pytest.raises(ValueError):
        index.search(
            query="implant",
            limit=0,
        )
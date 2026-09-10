import pytest

from dental_rag.retrieval.fusion import (
    ReciprocalRankFusion,
)
from dental_rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)


class FakeDenseRetriever:
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results

    def retrieve(
        self,
        query: str,
        limit: int,
    ) -> list[RetrievalResult]:
        return self.results[:limit]


class FakeSparseRetriever:
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results

    def retrieve(
        self,
        query: str,
        limit: int,
    ) -> list[RetrievalResult]:
        return self.results[:limit]


def make_result(
    document_id: str,
) -> RetrievalResult:
    return RetrievalResult(
        id=document_id,
        score=1.0,
        payload={
            "content": document_id,
        },
    )


def create_hybrid(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
) -> HybridRetriever:
    return HybridRetriever(
        dense_retriever=FakeDenseRetriever(
            dense_results,
        ),
        sparse_retriever=FakeSparseRetriever(
            sparse_results,
        ),
        fusion=ReciprocalRankFusion(),
    )


def test_hybrid_combines_dense_and_sparse_results() -> None:
    retriever = create_hybrid(
        dense_results=[
            make_result("doc-1"),
        ],
        sparse_results=[
            make_result("doc-2"),
        ],
    )

    results = retriever.retrieve(
        query="implant",
        limit=5,
    )

    assert {
        result.id
        for result in results
    } == {
        "doc-1",
        "doc-2",
    }


def test_hybrid_handles_empty_dense_results() -> None:
    retriever = create_hybrid(
        dense_results=[],
        sparse_results=[
            make_result("doc-1"),
        ],
    )

    results = retriever.retrieve(
        query="implant",
    )

    assert len(results) == 1
    assert results[0].id == "doc-1"


def test_hybrid_handles_empty_sparse_results() -> None:
    retriever = create_hybrid(
        dense_results=[
            make_result("doc-1"),
        ],
        sparse_results=[],
    )

    results = retriever.retrieve(
        query="implant",
    )

    assert len(results) == 1
    assert results[0].id == "doc-1"


def test_hybrid_handles_no_results() -> None:
    retriever = create_hybrid(
        dense_results=[],
        sparse_results=[],
    )

    results = retriever.retrieve(
        query="unknown",
    )

    assert results == []


def test_hybrid_deduplicates_documents() -> None:
    retriever = create_hybrid(
        dense_results=[
            make_result("doc-1"),
        ],
        sparse_results=[
            make_result("doc-1"),
        ],
    )

    results = retriever.retrieve(
        query="implant",
    )

    assert len(results) == 1
    assert results[0].id == "doc-1"


def test_hybrid_rejects_blank_query() -> None:
    retriever = create_hybrid(
        dense_results=[],
        sparse_results=[],
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_hybrid_rejects_invalid_limit() -> None:
    retriever = create_hybrid(
        dense_results=[],
        sparse_results=[],
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "implant",
            limit=0,
        )
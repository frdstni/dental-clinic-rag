import pytest

from dental_rag.retrieval.models import RetrievalResult
from dental_rag.retrieval.sparse_retriever import SparseRetriever


def test_sparse_retriever_returns_relevant_documents() -> None:
    documents = [
        {
            "id": "doc-1",
            "content": "dental implant surgery information",
            "source": "implant.txt",
        },
        {
            "id": "doc-2",
            "content": "tooth whitening procedure",
            "source": "whitening.txt",
        },
    ]

    retriever = SparseRetriever(
        documents=documents,
    )

    results = retriever.retrieve(
        query="dental implant",
        limit=1,
    )

    assert len(results) == 1

    assert isinstance(
        results[0],
        RetrievalResult,
    )

    assert results[0].id == "doc-1"


def test_sparse_retriever_rejects_empty_query() -> None:
    retriever = SparseRetriever(
        documents=[
            {
                "id": "doc-1",
                "content": "dental implant",
            }
        ],
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_sparse_retriever_rejects_invalid_limit() -> None:
    retriever = SparseRetriever(
        documents=[
            {
                "id": "doc-1",
                "content": "dental implant",
            }
        ],
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "implant",
            limit=0,
        )
from unittest.mock import MagicMock

import pytest

from dental_rag.retrieval.models import (
    RetrievalResult,
)
from dental_rag.retrieval.pipeline import (
    RetrievalPipeline,
)


def create_result(
    result_id: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        score=score,
        payload={
            "content": "test content",
        },
    )


def test_retrieval_pipeline_without_reranker() -> None:
    retriever = MagicMock()

    retriever.retrieve.return_value = [
        create_result("1", 0.9),
        create_result("2", 0.8),
    ]

    pipeline = RetrievalPipeline(
        retriever=retriever,
    )

    results = pipeline.retrieve(
        query="implant",
        limit=1,
    )

    assert len(results) == 1

    retriever.retrieve.assert_called_once_with(
        query="implant",
        limit=1,
    )


def test_retrieval_pipeline_with_reranker() -> None:
    retriever = MagicMock()
    reranker = MagicMock()

    candidates = [
        create_result("1", 0.5),
        create_result("2", 0.4),
    ]

    reranked = [
        create_result("2", 0.95),
    ]

    retriever.retrieve.return_value = candidates
    reranker.rerank.return_value = reranked

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
    )

    results = pipeline.retrieve(
        query="implant",
        limit=5,
    )

    assert results == reranked

    reranker.rerank.assert_called_once_with(
        query="implant",
        results=candidates,
        limit=5,
    )


def test_retrieval_pipeline_rejects_empty_query() -> None:
    pipeline = RetrievalPipeline(
        retriever=MagicMock(),
    )

    with pytest.raises(ValueError):
        pipeline.retrieve(
            query="",
        )


def test_retrieval_pipeline_rejects_invalid_limit() -> None:
    pipeline = RetrievalPipeline(
        retriever=MagicMock(),
    )

    with pytest.raises(ValueError):
        pipeline.retrieve(
            query="test",
            limit=0,
        )


def test_retrieval_pipeline_handles_empty_results() -> None:
    retriever = MagicMock()

    retriever.retrieve.return_value = []

    pipeline = RetrievalPipeline(
        retriever=retriever,
    )

    assert pipeline.retrieve(
        query="test",
    ) == []
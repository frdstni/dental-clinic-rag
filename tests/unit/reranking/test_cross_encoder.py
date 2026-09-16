from unittest.mock import MagicMock, patch

import pytest

from dental_rag.reranking.cross_encoder import (
    CrossEncoderReranker,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)


def create_result(
    content: str,
    score: float = 0.5,
) -> RetrievalResult:
    return RetrievalResult(
        id="1",
        score=score,
        payload={
            "content": content,
        },
    )


def test_cross_encoder_reranker_reranks_results() -> None:
    with patch(
        "dental_rag.reranking.cross_encoder.CrossEncoder",
    ) as mock_encoder:
        model = MagicMock()

        model.predict.return_value = [
            0.9,
            0.2,
        ]

        mock_encoder.return_value = model

        reranker = CrossEncoderReranker(
            model_name="test-model",
        )

        results = [
            create_result(
                "implant information",
            ),
            create_result(
                "tooth cleaning",
            ),
        ]

        reranked = reranker.rerank(
            query="implant",
            results=results,
        )

    assert reranked[0].score == 0.9
    assert reranked[1].score == 0.2


def test_cross_encoder_reranker_respects_limit() -> None:
    with patch(
        "dental_rag.reranking.cross_encoder.CrossEncoder",
    ) as mock_encoder:
        model = MagicMock()

        model.predict.return_value = [
            0.9,
            0.8,
        ]

        mock_encoder.return_value = model

        reranker = CrossEncoderReranker(
            model_name="test-model",
        )

        results = [
            create_result("a"),
            create_result("b"),
        ]

        output = reranker.rerank(
            query="test",
            results=results,
            limit=1,
        )

    assert len(output) == 1


def test_cross_encoder_reranker_rejects_empty_query() -> None:
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker,
    )

    with pytest.raises(ValueError):
        reranker.rerank(
            query="",
            results=[],
        )


def test_cross_encoder_reranker_rejects_invalid_limit() -> None:
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker,
    )

    with pytest.raises(ValueError):
        reranker.rerank(
            query="test",
            results=[],
            limit=0,
        )


def test_cross_encoder_reranker_returns_empty_for_empty_results() -> None:
    with patch(
        "dental_rag.reranking.cross_encoder.CrossEncoder",
    ):
        reranker = CrossEncoderReranker(
            model_name="test-model",
        )

    assert reranker.rerank(
        query="test",
        results=[],
    ) == []


def test_cross_encoder_reranker_rejects_non_string_content() -> None:
    with patch(
        "dental_rag.reranking.cross_encoder.CrossEncoder",
    ):
        reranker = CrossEncoderReranker(
            model_name="test-model",
        )

    result = RetrievalResult(
        id="1",
        score=0.5,
        payload={
            "content": 123,
        },
    )

    with pytest.raises(TypeError):
        reranker.rerank(
            query="test",
            results=[result],
        )
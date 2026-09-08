from dataclasses import FrozenInstanceError

import pytest

from dental_rag.retrieval.models import (
    RetrievalResult,
)


def test_retrieval_result_creation() -> None:
    result = RetrievalResult(
        id="doc-1",
        score=0.85,
        payload={
            "text": "implant information",
        },
    )

    assert result.id == "doc-1"
    assert result.score == 0.85
    assert result.payload == {
        "text": "implant information",
    }


def test_retrieval_result_rejects_blank_id() -> None:
    with pytest.raises(ValueError):
        RetrievalResult(
            id="",
            score=0.8,
            payload={},
        )


@pytest.mark.parametrize(
    "score",
    [
        -1,
        -0.1,
    ],
)
def test_retrieval_result_rejects_negative_score(
    score: float,
) -> None:
    with pytest.raises(ValueError):
        RetrievalResult(
            id="doc-1",
            score=score,
            payload={},
        )


def test_retrieval_result_allows_zero_score() -> None:
    result = RetrievalResult(
        id="doc-1",
        score=0,
        payload={},
    )

    assert result.score == 0


def test_retrieval_result_is_immutable() -> None:
    result = RetrievalResult(
        id="doc-1",
        score=0.8,
        payload={},
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.score = 0.5
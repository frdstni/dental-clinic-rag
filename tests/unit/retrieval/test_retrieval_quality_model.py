from dataclasses import FrozenInstanceError

import pytest

from dental_rag.retrieval.models import (
    RetrievalQuality,
)


def test_retrieval_quality_creation() -> None:
    quality = RetrievalQuality(
        passed=True,
        reason="sufficient score",
    )

    assert quality.passed is True
    assert quality.reason == "sufficient score"


def test_quality_rejects_blank_reason() -> None:
    with pytest.raises(ValueError):
        RetrievalQuality(
            passed=False,
            reason="",
        )


def test_quality_is_immutable() -> None:
    quality = RetrievalQuality(
        passed=True,
        reason="ok",
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        quality.passed = False
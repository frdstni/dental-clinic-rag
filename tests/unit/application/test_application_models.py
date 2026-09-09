from dental_rag.application.models import (
    RetrievalContext,
)
from dental_rag.retrieval.models import (
    RetrievalQuality,
    RetrievalResult,
)


def test_retrieval_context_creation() -> None:
    context = RetrievalContext(
        results=(
            RetrievalResult(
                id="doc-1",
                score=0.8,
                payload={
                    "text": "implant",
                },
            ),
        ),
        quality=RetrievalQuality(
            passed=True,
            reason="sufficient score",
        ),
    )

    assert len(context.results) == 1

    assert context.results[0].id == "doc-1"

    assert context.quality.passed is True

    assert (
        context.quality.reason
        == "sufficient score"
    )
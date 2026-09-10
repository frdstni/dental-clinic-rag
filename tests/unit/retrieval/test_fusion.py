from dental_rag.retrieval.fusion import (
    ReciprocalRankFusion,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)


def test_rrf_fuses_results_from_multiple_retrievers() -> None:
    dense_results = [
        RetrievalResult(
            id="doc-1",
            score=0.9,
            payload={
                "content": "implant",
            },
        ),
        RetrievalResult(
            id="doc-2",
            score=0.8,
            payload={
                "content": "whitening",
            },
        ),
    ]

    sparse_results = [
        RetrievalResult(
            id="doc-2",
            score=12.0,
            payload={
                "content": "whitening",
            },
        ),
        RetrievalResult(
            id="doc-3",
            score=10.0,
            payload={
                "content": "gum disease",
            },
        ),
    ]

    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        result_lists=[
            dense_results,
            sparse_results,
        ],
        limit=3,
    )

    assert len(results) == 3

    assert results[0].id == "doc-2"

    assert {
        result.id
        for result in results
    } == {
        "doc-1",
        "doc-2",
        "doc-3",
    }


def test_rrf_rejects_invalid_limit() -> None:
    fusion = ReciprocalRankFusion()

    try:
        fusion.fuse(
            result_lists=[],
            limit=0,
        )
    except ValueError:
        assert True
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_rrf_rejects_invalid_k() -> None:
    try:
        ReciprocalRankFusion(
            k=0,
        )
    except ValueError:
        assert True
    else:
        raise AssertionError(
            "Expected ValueError"
        )
import pytest

from dental_rag.query_processing.analyzer import (
    QueryAnalyzer,
)
from dental_rag.query_processing.decomposer import (
    QueryDecomposer,
)
from dental_rag.query_processing.expansion import (
    QueryExpander,
)
from dental_rag.query_processing.pipeline import (
    QueryPipeline,
)


@pytest.fixture
def pipeline() -> QueryPipeline:
    return QueryPipeline(
        analyzer=QueryAnalyzer(),
        expander=QueryExpander(),
        decomposer=QueryDecomposer(),
    )


def test_short_query_uses_expansion(
    pipeline: QueryPipeline,
) -> None:
    result = pipeline.process(
        "implant",
    )

    assert result == (
        "implant",
        "implant dental treatment",
        "implant risks complications",
    )


def test_normal_query_returns_original_query(
    pipeline: QueryPipeline,
) -> None:
    query = "implant treatment options"

    result = pipeline.process(
        query,
    )

    assert result == (query,)


def test_long_query_uses_decomposition(
    pipeline: QueryPipeline,
) -> None:
    result = pipeline.process(
        ("implant causes and treatment and prevention and recovery and complications"),
    )

    assert result == (
        "implant causes",
        "treatment",
        "prevention",
        "recovery",
        "complications",
    )


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_empty_query_is_rejected(
    pipeline: QueryPipeline,
    query: str,
) -> None:
    with pytest.raises(ValueError):
        pipeline.process(query)


def test_pipeline_preserves_original_query_for_normal_flow(
    pipeline: QueryPipeline,
) -> None:
    query = "dental implant procedure"

    result = pipeline.process(
        query,
    )

    assert result == (query,)

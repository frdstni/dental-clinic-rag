import pytest

from dental_rag.query_processing.analyzer import (
    QueryAnalyzer,
)
from dental_rag.query_processing.models import (
    QueryAction,
    QueryType,
)


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


def test_short_query_requires_expansion(
    analyzer: QueryAnalyzer,
) -> None:
    result = analyzer.analyze(
        "implant cost",
    )

    assert result.query_type == QueryType.SHORT
    assert result.action == QueryAction.EXPAND


def test_three_words_query_is_normal(
    analyzer: QueryAnalyzer,
) -> None:
    result = analyzer.analyze(
        "implant cost today",
    )

    assert result.query_type == QueryType.NORMAL
    assert result.action == QueryAction.NONE


def test_eight_words_query_is_normal(
    analyzer: QueryAnalyzer,
) -> None:
    result = analyzer.analyze(
        "one two three four five six seven eight",
    )

    assert result.query_type == QueryType.NORMAL


def test_nine_words_query_requires_decomposition(
    analyzer: QueryAnalyzer,
) -> None:
    result = analyzer.analyze(
        "one two three four five six seven eight nine",
    )

    assert result.query_type == QueryType.LONG
    assert result.action == QueryAction.DECOMPOSE


def test_long_dental_query_requires_decomposition(
    analyzer: QueryAnalyzer,
) -> None:
    result = analyzer.analyze(
        (
            "What are the causes of dental "
            "implant failure and how can "
            "they be prevented effectively "
            "for patients?"
        ),
    )

    assert result.query_type == QueryType.LONG
    assert result.action == QueryAction.DECOMPOSE


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n",
        "\t",
        " \n\t ",
    ],
)
def test_empty_query_is_rejected(
    analyzer: QueryAnalyzer,
    query: str,
) -> None:
    with pytest.raises(ValueError):
        analyzer.analyze(query)


def test_original_query_is_preserved(
    analyzer: QueryAnalyzer,
) -> None:
    query = "   implant cost   "

    result = analyzer.analyze(query)

    assert result.original_query == query
import pytest

from dental_rag.query_processing.expansion import (
    QueryExpander,
)
from dental_rag.query_processing.models import (
    ExpandedQuery,
)


@pytest.fixture
def expander() -> QueryExpander:
    return QueryExpander()


def test_short_query_is_expanded(
    expander: QueryExpander,
) -> None:
    result = expander.expand(
        "implant",
    )

    assert isinstance(
        result,
        ExpandedQuery,
    )

    assert result.original_query == "implant"

    assert len(
        result.expanded_queries,
    ) == 3


def test_long_query_is_not_expanded(
    expander: QueryExpander,
) -> None:
    query = "implant treatment options"

    result = expander.expand(
        query,
    )

    assert result.expanded_queries == (
        query,
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
    expander: QueryExpander,
    query: str,
) -> None:
    with pytest.raises(ValueError):
        expander.expand(query)


def test_original_query_is_preserved(
    expander: QueryExpander,
) -> None:
    query = "implant"

    result = expander.expand(
        query,
    )

    assert result.original_query == query

def test_two_word_query_is_expanded(
    expander: QueryExpander,
) -> None:
    result = expander.expand(
        "implant cost",
    )

    assert len(
        result.expanded_queries,
    ) == 3


def test_three_word_query_is_not_expanded(
    expander: QueryExpander,
) -> None:
    query = "implant cost today"

    result = expander.expand(
        query,
    )

    assert result.expanded_queries == (
        query,
    )


def test_original_query_is_first_expansion(
    expander: QueryExpander,
) -> None:
    result = expander.expand(
        "implant",
    )

    assert (
        result.expanded_queries[0]
        == result.original_query
    )


def test_expansion_results_are_not_empty(
    expander: QueryExpander,
) -> None:
    result = expander.expand(
        "implant",
    )

    for expanded in result.expanded_queries:
        assert expanded.strip()


def test_query_with_spaces_is_preserved(
    expander: QueryExpander,
) -> None:
    query = "  implant  "

    result = expander.expand(
        query,
    )

    assert result.original_query == query
import pytest

from dental_rag.query_processing.decomposer import (
    QueryDecomposer,
)


@pytest.fixture
def decomposer() -> QueryDecomposer:
    return QueryDecomposer()


def test_query_is_decomposed_by_and(
    decomposer: QueryDecomposer,
) -> None:
    result = decomposer.decompose(
        "implant causes and treatment",
    )

    assert result.sub_queries == (
        "implant causes",
        "treatment",
    )


def test_query_with_commas_is_decomposed(
    decomposer: QueryDecomposer,
) -> None:
    result = decomposer.decompose(
        "cost, procedure and risks",
    )

    assert result.sub_queries == (
        "cost",
        "procedure",
        "risks",
    )


def test_simple_query_returns_original_query(
    decomposer: QueryDecomposer,
) -> None:
    query = "implant treatment"

    result = decomposer.decompose(
        query,
    )

    assert result.sub_queries == (
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
    decomposer: QueryDecomposer,
    query: str,
) -> None:
    with pytest.raises(ValueError):
        decomposer.decompose(query)


def test_original_query_is_preserved(
    decomposer: QueryDecomposer,
) -> None:
    query = (
        "implant causes and treatment"
    )

    result = decomposer.decompose(
        query,
    )

    assert result.original_query == query

def test_multiple_and_tokens_are_handled(
    decomposer: QueryDecomposer,
) -> None:
    result = decomposer.decompose(
        "implant and and treatment",
    )

    assert result.sub_queries == (
        "implant",
        "treatment",
    )


def test_only_separator_is_rejected(
    decomposer: QueryDecomposer,
) -> None:
    with pytest.raises(ValueError):
        decomposer.decompose(
            "and",
        )


def test_multiple_parts_are_created(
    decomposer: QueryDecomposer,
) -> None:
    result = decomposer.decompose(
        "cost and procedure and risks",
    )

    assert result.sub_queries == (
        "cost",
        "procedure",
        "risks",
    )


def test_extra_spaces_are_ignored(
    decomposer: QueryDecomposer,
) -> None:
    result = decomposer.decompose(
        "implant   and   treatment",
    )

    assert result.sub_queries == (
        "implant",
        "treatment",
    )


def test_uppercase_separator_is_supported(
    decomposer: QueryDecomposer,
) -> None:
    result = decomposer.decompose(
        "implant AND treatment",
    )

    assert result.sub_queries == (
        "implant",
        "treatment",
    )
from dataclasses import FrozenInstanceError

import pytest

from dental_rag.query_processing.models import (
    QueryAction,
    QueryAnalysis,
    QueryType,
)


def test_query_analysis_creation() -> None:
    analysis = QueryAnalysis(
        original_query="implant",
        query_type=QueryType.SHORT,
        action=QueryAction.EXPAND,
    )

    assert analysis.original_query == "implant"
    assert analysis.query_type == QueryType.SHORT
    assert analysis.action == QueryAction.EXPAND


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "\t",
        "\n",
        " \n\t ",
    ],
)
def test_query_analysis_rejects_blank_query(
    query: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="query cannot be blank",
    ):
        QueryAnalysis(
            original_query=query,
            query_type=QueryType.SHORT,
            action=QueryAction.EXPAND,
        )


@pytest.mark.parametrize(
    ("query_type", "action"),
    [
        (QueryType.SHORT, QueryAction.DECOMPOSE),
        (QueryType.SHORT, QueryAction.NONE),
        (QueryType.NORMAL, QueryAction.EXPAND),
        (QueryType.NORMAL, QueryAction.DECOMPOSE),
        (QueryType.LONG, QueryAction.EXPAND),
        (QueryType.LONG, QueryAction.NONE),
    ],
)
def test_query_analysis_rejects_incompatible_action(
    query_type: QueryType,
    action: QueryAction,
) -> None:
    with pytest.raises(ValueError):
        QueryAnalysis(
            original_query="valid query",
            query_type=query_type,
            action=action,
        )


@pytest.mark.parametrize(
    ("query_type", "action"),
    [
        (QueryType.SHORT, QueryAction.EXPAND),
        (QueryType.NORMAL, QueryAction.NONE),
        (QueryType.LONG, QueryAction.DECOMPOSE),
    ],
)
def test_query_analysis_accepts_valid_type_action_pairs(
    query_type: QueryType,
    action: QueryAction,
) -> None:
    analysis = QueryAnalysis(
        original_query="valid query",
        query_type=query_type,
        action=action,
    )

    assert analysis.query_type is query_type
    assert analysis.action is action


def test_query_analysis_is_immutable() -> None:
    analysis = QueryAnalysis(
        original_query="implant",
        query_type=QueryType.SHORT,
        action=QueryAction.EXPAND,
    )

    with pytest.raises(FrozenInstanceError):
        analysis.original_query = "changed"


def test_query_enum_values_are_stable() -> None:
    assert QueryType.SHORT.value == "short"
    assert QueryType.NORMAL.value == "normal"
    assert QueryType.LONG.value == "long"

    assert QueryAction.NONE.value == "none"
    assert QueryAction.EXPAND.value == "expand"
    assert QueryAction.DECOMPOSE.value == "decompose"
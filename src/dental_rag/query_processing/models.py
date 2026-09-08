from dataclasses import dataclass
from enum import StrEnum


class QueryType(StrEnum):
    SHORT = "short"
    NORMAL = "normal"
    LONG = "long"


class QueryAction(StrEnum):
    NONE = "none"
    EXPAND = "expand"
    DECOMPOSE = "decompose"


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    original_query: str
    query_type: QueryType
    action: QueryAction

    def __post_init__(self) -> None:
        if not self.original_query.strip():
            raise ValueError(
                "query cannot be blank"
            )

        expected_action = {
            QueryType.SHORT: QueryAction.EXPAND,
            QueryType.NORMAL: QueryAction.NONE,
            QueryType.LONG: QueryAction.DECOMPOSE,
        }[self.query_type]

        if self.action is not expected_action:
            raise ValueError(
                f"{self.query_type.value} query must use "
                f"{expected_action.value} action"
            )

@dataclass(frozen=True, slots=True)
class ExpandedQuery:
    original_query: str
    expanded_queries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.original_query.strip():
            raise ValueError(
                "original query cannot be blank"
            )

        if not self.expanded_queries:
            raise ValueError(
                "expanded queries cannot be empty"
            )

        for query in self.expanded_queries:
            if not query.strip():
                raise ValueError(
                    "expanded query cannot be blank"
                )

@dataclass(frozen=True, slots=True)
class DecomposedQuery:
    original_query: str
    sub_queries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.original_query.strip():
            raise ValueError(
                "original query cannot be blank"
            )

        if not self.sub_queries:
            raise ValueError(
                "sub queries cannot be empty"
            )

        for query in self.sub_queries:
            if not query.strip():
                raise ValueError(
                    "sub query cannot be blank"
                )
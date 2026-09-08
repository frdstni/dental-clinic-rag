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
import pytest

from dental_rag.agent.general_dental import (
    GeneralDentalAgent,
)
from dental_rag.web_search.models import (
    WebSearchResult,
)


class FakeSearchProvider:
    def __init__(
        self,
        results: list[WebSearchResult],
    ) -> None:
        self.results = results

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[WebSearchResult]:
        return self.results


class FailingSearchProvider:
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[WebSearchResult]:
        raise RuntimeError(
            "Search failure",
        )


def test_general_dental_agent_returns_results() -> None:
    provider = FakeSearchProvider(
        [
            WebSearchResult(
                title="Tooth pain",
                content="Dental information",
                url="https://example.com",
            ),
        ],
    )

    agent = GeneralDentalAgent(
        search_provider=provider,
    )

    result = agent.run(
        {
            "query": "Why does tooth pain happen?",
        },
    )

    assert len(
        result["web_results"],
    ) == 1


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n",
    ],
)
def test_general_dental_agent_rejects_empty_query(
    query: str,
) -> None:
    agent = GeneralDentalAgent(
        search_provider=FakeSearchProvider(
            [],
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        agent.run(
            {
                "query": query,
            },
        )


def test_general_dental_agent_handles_search_failure() -> None:
    agent = GeneralDentalAgent(
        search_provider=FailingSearchProvider(),
    )

    with pytest.raises(
        RuntimeError,
    ):
        agent.run(
            {
                "query": "Dental question",
            },
        )


def test_general_dental_agent_handles_empty_results() -> None:
    agent = GeneralDentalAgent(
        search_provider=FakeSearchProvider(
            [],
        ),
    )

    result = agent.run(
        {
            "query": "Dental question",
        },
    )

    assert result["web_results"] == []


def test_general_dental_agent_preserves_state() -> None:
    agent = GeneralDentalAgent(
        search_provider=FakeSearchProvider(
            [],
        ),
    )

    result = agent.run(
        {
            "query": "Dental question",
            "confidence": 0.8,
        },
    )

    assert result["confidence"] == 0.8
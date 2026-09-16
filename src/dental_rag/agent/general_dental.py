from dental_rag.agent.state import AgentState
from dental_rag.web_search.base import WebSearchProvider


class GeneralDentalAgent:
    """
    Agent responsible for general dentistry queries.
    """

    def __init__(
        self,
        search_provider: WebSearchProvider,
    ) -> None:
        self.search_provider = search_provider

    def run(
        self,
        state: AgentState,
    ) -> AgentState:
        query = state.get(
            "query",
        )

        if query is None or not query.strip():
            raise ValueError(
                "query cannot be empty",
            )

        results = self.search_provider.search(
            query=query,
        )

        return {
            **state,
            "web_results": results,
        }
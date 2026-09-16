from typing import TypedDict

from dental_rag.agent.models import AgentIntent
from dental_rag.web_search.models import WebSearchResult


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    query: str

    intent: AgentIntent

    confidence: float

    reasoning: str

    context: list[str]

    web_results: list[WebSearchResult]

    answer: str
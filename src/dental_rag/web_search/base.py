from typing import Protocol

from dental_rag.web_search.models import (
    WebSearchResult,
)


class WebSearchProvider(Protocol):
    """
    Interface for web search providers.
    """

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[WebSearchResult]:
        """
        Search web content.
        """
        ...
from typing import Protocol


class QueryRefiner(Protocol):
    """
    Interface for query refinement strategies.
    """

    def refine(
        self,
        query: str,
    ) -> str:
        """
        Refine a query.
        """
        ...


from dental_rag.query_processing.models import (
    ExpandedQuery,
)


class QueryExpander:
    """
    Expands short queries into richer search queries.
    """

    def expand(
        self,
        query: str,
    ) -> ExpandedQuery:
        if not query.strip():
            raise ValueError(
                "query cannot be blank"
            )

        words = query.split()

        if len(words) > 2:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=(
                    query,
                ),
            )

        return ExpandedQuery(
            original_query=query,
            expanded_queries=(
                query,
                f"{query} dental treatment",
                f"{query} risks complications",
            ),
        )
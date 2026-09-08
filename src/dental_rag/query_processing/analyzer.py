from dental_rag.query_processing.models import (
    QueryAction,
    QueryAnalysis,
    QueryType,
)


class QueryAnalyzer:
    """
    Analyzes user queries and determines
    the required query processing strategy.
    """

    def analyze(
        self,
        query: str,
    ) -> QueryAnalysis:
        if not query.strip():
            raise ValueError(
                "query cannot be blank"
            )

        word_count = len(
            query.split()
        )

        if word_count <= 2:
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType.SHORT,
                action=QueryAction.EXPAND,
            )

        if word_count > 8:
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType.LONG,
                action=QueryAction.DECOMPOSE,
            )

        return QueryAnalysis(
            original_query=query,
            query_type=QueryType.NORMAL,
            action=QueryAction.NONE,
        )
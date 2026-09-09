from dental_rag.query_processing.analyzer import (
    QueryAnalyzer,
)
from dental_rag.query_processing.decomposer import (
    QueryDecomposer,
)
from dental_rag.query_processing.expansion import (
    QueryExpander,
)
from dental_rag.query_processing.models import (
    QueryAction,
)


class QueryPipeline:
    """
    Orchestrates query analysis and transformation.
    """

    def __init__(
        self,
        analyzer: QueryAnalyzer,
        expander: QueryExpander,
        decomposer: QueryDecomposer,
    ) -> None:
        self._analyzer = analyzer
        self._expander = expander
        self._decomposer = decomposer

    def process(
        self,
        query: str,
    ) -> tuple[str, ...]:
        """
        Process a query and return retrieval-ready queries.
        """

        analysis = self._analyzer.analyze(
            query,
        )

        if analysis.action is QueryAction.EXPAND:
            expanded = self._expander.expand(
                query,
            )

            return expanded.expanded_queries

        if analysis.action is QueryAction.DECOMPOSE:
            decomposed = self._decomposer.decompose(
                query,
            )

            return decomposed.sub_queries

        return (
            query,
        )
from dental_rag.query_processing.models import (
    DecomposedQuery,
)


class QueryDecomposer:
    """
    Decomposes complex queries into smaller sub queries.
    """

    def decompose(
        self,
        query: str,
    ) -> DecomposedQuery:
        if not query.strip():
            raise ValueError(
                "query cannot be blank"
            )

        normalized_query = (
            query
            .lower()
            .replace(",", " and ")
        )

        parts = [
            part.strip()
            for part in normalized_query.split(" and ")
            if part.strip()
            and part.strip() != "and"
        ]

        if not parts:
            raise ValueError(
                "query must contain valid parts"
            )

        if len(parts) == 1:
            parts = [
                query,
            ]

        cleaned_parts = tuple(
            part.removeprefix("and ").strip()
            for part in parts
            if part.strip()
        )

        return DecomposedQuery(
            original_query=query,
            sub_queries=cleaned_parts,
        )
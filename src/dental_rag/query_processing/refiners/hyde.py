from dental_rag.llm.base import LLM


class HyDERefiner:
    """
    Hypothetical Document Embeddings query refiner.
    """

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        self._llm = llm

    def refine(
        self,
        query: str,
    ) -> str:
        """
        Generate a hypothetical document for a query.
        """

        if not query.strip():
            raise ValueError(
                "query cannot be blank"
            )

        prompt = (
            "Generate a hypothetical dental "
            "document that answers this query:\n"
            f"{query}"
        )

        response = self._llm.generate(
            prompt,
        )

        if not response.strip():
            raise ValueError(
                "LLM returned empty response"
            )

        return response
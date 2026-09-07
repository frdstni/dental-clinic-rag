from typing import Any

from dental_rag.retrieval.retriever import Retriever


class RagService:
    def __init__(
        self,
        retriever: Retriever,
    ) -> None:
        self.retriever = retriever

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self.retriever.retrieve(
            query=query,
            limit=limit,
        )
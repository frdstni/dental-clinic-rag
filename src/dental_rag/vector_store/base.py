from typing import Protocol


class VectorStore(Protocol):
    """
    Interface for vector database operations.
    """

    def create_collection(
        self,
        vector_size: int,
    ) -> None:
        ...

    def add(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        ...
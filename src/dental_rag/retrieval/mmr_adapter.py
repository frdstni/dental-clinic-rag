from dental_rag.retrieval.mmr_models import (
    MMRDocument,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)


class MMRAdapter:
    """
    Converts retrieval results to MMR documents
    and maps selected documents back.
    """

    @staticmethod
    def to_documents(
        results: list[RetrievalResult],
        embeddings: list[list[float]],
    ) -> list[MMRDocument]:
        if len(results) != len(embeddings):
            raise ValueError(
                "results and embeddings length must match",
            )

        documents: list[MMRDocument] = []

        for result, embedding in zip(
            results,
            embeddings,
            strict=True,
        ):
            content = result.payload.get(
                "content",
                "",
            )

            if not isinstance(content, str):
                raise TypeError(
                    "content must be a string",
                )

            documents.append(
                MMRDocument(
                    id=result.id,
                    content=content,
                    embedding=embedding,
                )
            )

        return documents

    @staticmethod
    def select_results(
        selected: list[MMRDocument],
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        result_map = {
            result.id: result
            for result in results
        }

        return [
            result_map[document.id]
            for document in selected
            if document.id in result_map
        ]
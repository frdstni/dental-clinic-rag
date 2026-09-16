from dental_rag.embeddings.base import (
    EmbeddingModel,
)
from dental_rag.reranking.base import (
    Reranker,
)
from dental_rag.retrieval.base import (
    RetrievalBackend,
)
from dental_rag.retrieval.mmr import (
    MMRSelector,
)
from dental_rag.retrieval.mmr_adapter import (
    MMRAdapter,
)
from dental_rag.retrieval.models import (
    RetrievalResult,
)


class RetrievalPipeline:
    """
    Executes retrieval, reranking and optional MMR selection.
    """

    def __init__(
        self,
        retriever: RetrievalBackend,
        reranker: Reranker | None = None,
        embedding_model: EmbeddingModel | None = None,
        mmr_selector: MMRSelector | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.embedding_model = embedding_model
        self.mmr_selector = mmr_selector

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty",
            )

        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero",
            )

        results = self.retriever.retrieve(
            query=query,
            limit=limit,
        )

        if self.reranker is not None:
            results = self.reranker.rerank(
                query=query,
                results=results,
                limit=limit,
            )

        if (
            self.mmr_selector is None
            or self.embedding_model is None
        ):
            return results[:limit]

        query_embedding = self.embedding_model.embed(
            [query],
        )[0]

        document_embeddings = self.embedding_model.embed(
            [
                str(
                    result.payload.get(
                        "content",
                        "",
                    )
                )
                for result in results
            ],
        )

        documents = MMRAdapter.to_documents(
            results=results,
            embeddings=document_embeddings,
        )

        selected = self.mmr_selector.select(
            query_embedding=query_embedding,
            documents=documents,
            limit=limit,
        )

        return MMRAdapter.select_results(
            selected=selected,
            results=results,
        )
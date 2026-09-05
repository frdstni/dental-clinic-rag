from typing import Protocol

from dental_rag.domain.models import (
    DocumentChunk,
    SourceDocument,
)
from dental_rag.embeddings.base import EmbeddingModel
from dental_rag.vector_store.base import VectorStore


class Chunker(Protocol):
    def chunk(
        self,
        document: SourceDocument,
    ) -> list[DocumentChunk]:
        ...


class IngestionPipeline:
    def __init__(
        self,
        chunker: Chunker,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def run(
        self,
        document: SourceDocument,
    ) -> list[DocumentChunk]:
        chunks = self.chunker.chunk(document)

        chunk_texts = [
            chunk.content
            for chunk in chunks
        ]

        embeddings = self.embedding_model.embed(
            chunk_texts
        )

        payloads = [
            {
                "content": chunk.content,
                "source": chunk.metadata.file_name,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        if embeddings:
            self.vector_store.add(
                vectors=embeddings,
                payloads=payloads,
            )

        return chunks
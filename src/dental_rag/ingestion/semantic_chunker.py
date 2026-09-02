import math
from collections.abc import Sequence
from itertools import pairwise

from dental_rag.domain.models import (
    DocumentChunk,
    DocumentMetadata,
    SourceDocument,
)
from dental_rag.embeddings.base import EmbeddingModel
from dental_rag.ingestion.sentence_splitter import (
    split_sentences,
)


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b, strict=True)
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a)
    )
    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b)
    )

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        raise ValueError("cosine similarity is undefined for zero vectors")

    return dot_product / (magnitude_a * magnitude_b)
def cosine_distance(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    return 1.0 - cosine_similarity(vector_a, vector_b)

def calculate_adjacent_distances(
    embeddings: Sequence[Sequence[float]],
) -> list[float]:
    return [
        cosine_distance(current, next_embedding)
        for current, next_embedding in pairwise(embeddings)
    ]

def calculate_percentile_threshold(
    distances: Sequence[float],
    percentile: float,
) -> float:
    if not distances:
        raise ValueError("distances must not be empty")

    if not 0 <= percentile <= 100:
        raise ValueError(
            "percentile must be between 0 and 100"
        )

    sorted_distances = sorted(distances)

    if len(sorted_distances) == 1:
        return sorted_distances[0]

    position = (
        (len(sorted_distances) - 1)
        * percentile
        / 100
    )

    lower_index = int(position)
    upper_index = lower_index + 1

    if upper_index >= len(sorted_distances):
        return sorted_distances[lower_index]

    lower_value = sorted_distances[lower_index]
    upper_value = sorted_distances[upper_index]

    fraction = position - lower_index

    return (
        lower_value
        + (upper_value - lower_value)
        * fraction
    )

def detect_breakpoints(
    distances: Sequence[float],
    threshold: float,
) -> list[int]:
    if not distances:
        raise ValueError("distances must not be empty")

    if threshold < 0:
        raise ValueError(
            "threshold must not be negative"
        )

    breakpoints: list[int] = []

    for index, distance in enumerate(distances):
        if distance > threshold:
            breakpoints.append(index)

    return breakpoints

def create_chunks(
    sentences: Sequence[str],
    breakpoints: Sequence[int],
    metadata: DocumentMetadata,
) -> list[DocumentChunk]:
    for breakpoint in breakpoints:
        if breakpoint < 0 or breakpoint >= len(sentences):
            raise ValueError(
                "breakpoint index is out of range"
            )
    chunks: list[DocumentChunk] = []

    start_index = 0

    breakpoint_set = set(breakpoints)

    for index in range(len(sentences)):
        if index in breakpoint_set:
            chunks.append(
                DocumentChunk(
                    content=" ".join(
                        sentences[start_index:index + 1]
                    ),
                    metadata=metadata,
                    chunk_index=len(chunks),
                    start_sentence_index=start_index,
                    end_sentence_index_exclusive=index + 1,
                )
            )

            start_index = index + 1

    if start_index < len(sentences):
        chunks.append(
            DocumentChunk(
                content=" ".join(
                    sentences[start_index:]
                ),
                metadata=metadata,
                chunk_index=len(chunks),
                start_sentence_index=start_index,
                end_sentence_index_exclusive=len(sentences),
            )
        )

    return chunks
class SemanticChunker:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        breakpoint_percentile: float,
    ) -> None:
        self.embedding_model = embedding_model
        self.breakpoint_percentile = breakpoint_percentile
    def chunk(
        self,
        document: SourceDocument,
    ) -> list[DocumentChunk]:
        sentences = split_sentences(
            document.content
        )
        if len(sentences) == 1:
            return [
              DocumentChunk(
                  content=sentences[0],
                  metadata=document.metadata,
                  chunk_index=0,
                  start_sentence_index=0,
                  end_sentence_index_exclusive=1, )
                  ]

        embeddings = self.embedding_model.embed(
            sentences
        )
        if len(sentences) != len(embeddings):
            raise ValueError(
                 "embedding count must match sentence count")

        distances = calculate_adjacent_distances(
            embeddings
        )

        threshold = calculate_percentile_threshold(
            distances,
            self.breakpoint_percentile,
        )

        breakpoints = detect_breakpoints(
            distances,
            threshold,
        )

        return create_chunks(
            sentences=sentences,
            breakpoints=breakpoints,
            metadata=document.metadata,
        )
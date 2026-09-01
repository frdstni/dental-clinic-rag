from pathlib import Path

import pytest

from dental_rag.domain.models import (
    DocumentChunk,
    DocumentMetadata,
    SourceDocument,
)
from dental_rag.ingestion.semantic_chunker import (
    SemanticChunker,
    calculate_adjacent_distances,
    calculate_percentile_threshold,
    cosine_distance,
    cosine_similarity,
    create_chunks,
    detect_breakpoints,
)


class FakeEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [float(index), 1.0]
            for index, _ in enumerate(texts)
        ]


def test_cosine_similarity_returns_one_for_identical_vectors() -> None:
    similarity = cosine_similarity([1.0, 0.0], [1.0, 0.0])

    assert similarity == 1.0


def test_cosine_similarity_returns_zero_for_orthogonal_vectors() -> None:
    similarity = cosine_similarity([1.0, 0.0], [0.0, 1.0])

    assert similarity == 0.0

def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])

def test_cosine_distance_returns_zero_for_identical_vectors() -> None:
    distance = cosine_distance([1.0, 0.0], [1.0, 0.0])

    assert distance == 0.0


def test_cosine_distance_returns_one_for_orthogonal_vectors() -> None:
    distance = cosine_distance([1.0, 0.0], [0.0, 1.0])

    assert distance == 1.0

def test_calculate_adjacent_distances_compares_neighboring_embeddings() -> None:
    embeddings = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    distances = calculate_adjacent_distances(embeddings)

    assert distances == [0.0, 1.0]

def test_calculate_percentile_threshold_returns_expected_value() -> None:
    distances = [0.1, 0.2, 0.3]

    threshold = calculate_percentile_threshold(
        distances,
        percentile=50.0,
    )

    assert threshold == 0.2
def test_percentile_threshold_rejects_empty_distances():
    with pytest.raises(ValueError):
        calculate_percentile_threshold(
            distances=[],
            percentile=50,
        )

@pytest.mark.parametrize(
    "percentile",
    [-1, 101],
)
def test_percentile_threshold_rejects_invalid_percentile(
    percentile: float,
):
    with pytest.raises(ValueError):
        calculate_percentile_threshold(
            distances=[0.1, 0.2],
            percentile=percentile,
        )

def test_percentile_threshold_with_single_distance():
    result = calculate_percentile_threshold(
        distances=[0.5],
        percentile=50,
    )

    assert result == 0.5

def test_detect_breakpoints_returns_indexes_above_threshold():
    distances = [0.1, 0.8, 0.2]

    result = detect_breakpoints(
        distances=distances,
        threshold=0.5,
    )

    assert result == [1]

def test_detect_breakpoints_rejects_empty_distances():
    with pytest.raises(ValueError):
        detect_breakpoints(
            distances=[],
            threshold=0.5,
        )

def test_detect_breakpoints_rejects_negative_threshold():
    with pytest.raises(ValueError):
        detect_breakpoints(
            distances=[0.1, 0.2],
            threshold=-0.1,
        )

def test_create_chunks_splits_sentences_at_breakpoints():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    sentences = [
        "Clinic is open on Saturday.",
        "Call us to book an appointment.",
        "Dental implants replace missing teeth.",
        "Aftercare is important after implant treatment.",
    ]

    result = create_chunks(
        sentences=sentences,
        breakpoints=[1],
        metadata=metadata,
    )

    assert len(result) == 2

    assert result[0].content == (
        "Clinic is open on Saturday. "
        "Call us to book an appointment."
    )
    assert result[0].chunk_index == 0
    assert result[0].start_sentence_index == 0
    assert result[0].end_sentence_index_exclusive == 2

    assert result[1].content == (
        "Dental implants replace missing teeth. "
        "Aftercare is important after implant treatment."
    )
    assert result[1].chunk_index == 1
    assert result[1].start_sentence_index == 2
    assert result[1].end_sentence_index_exclusive == 4

def test_create_chunks_without_breakpoints_returns_single_chunk():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    sentences = [
        "Clinic is open.",
        "Call for appointment.",
        "Emergency service is available.",
    ]

    result = create_chunks(
        sentences=sentences,
        breakpoints=[],
        metadata=metadata,
    )

    assert len(result) == 1

    assert result[0].content == (
        "Clinic is open. "
        "Call for appointment. "
        "Emergency service is available."
    )

    assert result[0].chunk_index == 0
    assert result[0].start_sentence_index == 0
    assert result[0].end_sentence_index_exclusive == 3

def test_create_chunks_preserves_metadata():
    metadata = DocumentMetadata(
        source_path=Path("clinic_information.txt"),
    )

    result = create_chunks(
        sentences=[
            "Dental clinic information."
        ],
        breakpoints=[],
        metadata=metadata,
    )

    assert result[0].metadata == metadata

def test_create_chunks_with_multiple_breakpoints():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    sentences = [
        "Sentence zero.",
        "Sentence one.",
        "Sentence two.",
        "Sentence three.",
        "Sentence four.",
    ]

    result = create_chunks(
        sentences=sentences,
        breakpoints=[1,3],
        metadata=metadata,
    )

    assert len(result) == 3

    assert result[0].start_sentence_index == 0
    assert result[0].end_sentence_index_exclusive == 2

    assert result[1].start_sentence_index == 2
    assert result[1].end_sentence_index_exclusive == 4

    assert result[2].start_sentence_index == 4
    assert result[2].end_sentence_index_exclusive == 5 

def test_create_chunks_rejects_invalid_breakpoint():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    sentences = [
        "Sentence zero.",
        "Sentence one.",
    ]

    with pytest.raises(ValueError):
        create_chunks(
            sentences=sentences,
            breakpoints=[5],
            metadata=metadata,
        )
def test_semantic_chunker_creates_chunks():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    document = SourceDocument(
        content=(
            "Clinic is open. "
            "Call for appointment. "
            "Dental implants are available."
        ),
        metadata=metadata,
    )

    chunker = SemanticChunker(
        embedding_model=FakeEmbeddingModel(),
        breakpoint_percentile=50,
    )

    result = chunker.chunk(document)

    assert len(result) >= 1

    assert all(
        isinstance(chunk, DocumentChunk)
        for chunk in result
    )

def test_semantic_chunker_splits_on_breakpoint():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    document = SourceDocument(
        content=(
            "Clinic is open. "
            "Call for appointment. "
            "Dental implants are available."
        ),
        metadata=metadata,
    )

    chunker = SemanticChunker(
        embedding_model=FakeEmbeddingModel(),
        breakpoint_percentile=50,
    )

    result = chunker.chunk(document)

    assert len(result) == 2
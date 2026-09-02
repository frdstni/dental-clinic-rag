from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    source_path: Path

    @property
    def file_name(self) -> str:
        return self.source_path.name

    @property
    def file_extension(self) -> str:
        return self.source_path.suffix.lower()


@dataclass(frozen=True, slots=True)
class SourceDocument:
    content: str
    metadata: DocumentMetadata

@dataclass(frozen=True, slots=True)
class DocumentChunk:
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    start_sentence_index: int
    end_sentence_index_exclusive: int

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("chunk content must not be blank")

        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")

        if self.start_sentence_index < 0:
            raise ValueError(
                "start_sentence_index must be >= 0"
            )

        if (
            self.end_sentence_index_exclusive
            <= self.start_sentence_index
        ):
            raise ValueError(
                "end_sentence_index_exclusive must be greater than start_sentence_index"
            )
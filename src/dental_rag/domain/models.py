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
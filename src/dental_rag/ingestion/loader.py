from pathlib import Path

from dental_rag.domain.models import DocumentMetadata, SourceDocument

SUPPORTED_TEXT_EXTENSIONS = frozenset({".txt", ".md"})


def load_text_document(path: str | Path) -> SourceDocument:
    source_path = Path(path)

    if not source_path.exists():
        raise FileNotFoundError(f"Document not found: {source_path}")

    if not source_path.is_file():
        raise IsADirectoryError(f"Expected a file, got: {source_path}")

    if source_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {source_path.suffix or '<no extension>'}"
        )

    content = source_path.read_text(encoding="utf-8")

    metadata = DocumentMetadata(source_path=source_path)

    return SourceDocument(
        content=content,
        metadata=metadata,
    )
import re
import unicodedata

from dental_rag.domain.models import SourceDocument

_HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[^\S\n]+")
_EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)

    normalized = normalized.lstrip("\ufeff")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in normalized.split("\n"):
        clean_line = _HORIZONTAL_WHITESPACE_PATTERN.sub(" ", line).strip()
        lines.append(clean_line)

    normalized = "\n".join(lines)
    normalized = _EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", normalized)

    return normalized.strip()


def preprocess_document(document: SourceDocument) -> SourceDocument:
    cleaned_content = normalize_text(document.content)

    if not cleaned_content:
        raise ValueError("Document content is empty after preprocessing.")

    return SourceDocument(
        content=cleaned_content,
        metadata=document.metadata,
    )
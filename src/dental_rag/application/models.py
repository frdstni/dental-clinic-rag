from dataclasses import dataclass

from dental_rag.retrieval.models import (
    RetrievalQuality,
    RetrievalResult,
)


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    results: tuple[RetrievalResult, ...]
    quality: RetrievalQuality
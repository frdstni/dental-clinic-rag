from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    id: str
    score: float
    payload: dict[str, object]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError(
                "id cannot be blank"
            )

        if self.score < 0:
            raise ValueError(
                "score cannot be negative"
            )

@dataclass(frozen=True, slots=True)
class RetrievalQuality:
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError(
                "reason cannot be blank"
            )
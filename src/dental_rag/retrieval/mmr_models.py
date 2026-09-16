from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MMRDocument:
    """
    Document representation used for diversity selection.
    """

    id: str
    content: str
    embedding: list[float]
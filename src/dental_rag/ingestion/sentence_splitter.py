import re


def split_sentences(text: str) -> list[str]:
    if not text.strip():
        raise ValueError("text must not be blank")

    return re.split(
        r"(?:\n\s*\n|(?<=[.!?؟])\s+)",
        text.strip(),
    )

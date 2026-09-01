import pytest

from dental_rag.ingestion.sentence_splitter import split_sentences


def test_split_sentences_handles_common_sentence_endings() -> None:
    text = (
        "کلینیک امروز باز است. "
        "آیا خدمات ایمپلنت ارائه می‌شود؟ "
        "بله! برای رزرو تماس بگیرید?"
    )

    sentences = split_sentences(text)

    assert sentences == [
        "کلینیک امروز باز است.",
        "آیا خدمات ایمپلنت ارائه می‌شود؟",
        "بله!",
        "برای رزرو تماس بگیرید?",
    ]
def test_split_sentences_respects_paragraph_boundaries() -> None:
    text = "ساعات کاری کلینیک\n\nخدمات ایمپلنت"

    sentences = split_sentences(text)

    assert sentences == [
        "ساعات کاری کلینیک",
        "خدمات ایمپلنت",
    ]   
def test_split_sentences_returns_single_sentence_unchanged() -> None:
    text = "کلینیک امروز باز است."

    sentences = split_sentences(text)

    assert sentences == ["کلینیک امروز باز است."]

def test_split_sentences_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="text"):
        split_sentences("   \n\t")
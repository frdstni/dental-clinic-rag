from dataclasses import FrozenInstanceError

import pytest

from dental_rag.web_search.models import (
    WebSearchResult,
)


def test_web_search_result_creation() -> None:
    result = WebSearchResult(
        title="Dental health",
        content="Information about dental health",
        url="https://example.com",
    )

    assert result.title == "Dental health"
    assert result.content == "Information about dental health"
    assert result.url == "https://example.com"


@pytest.mark.parametrize(
    "field",
    [
        "title",
        "content",
        "url",
    ],
)
def test_web_search_result_rejects_empty_string_fields(
    field: str,
) -> None:
    values = {
        "title": "Dental",
        "content": "Content",
        "url": "https://example.com",
    }

    values[field] = ""

    with pytest.raises(
        ValueError,
    ):
        WebSearchResult(
            **values,
        )


@pytest.mark.parametrize(
    "field",
    [
        "title",
        "content",
        "url",
    ],
)
def test_web_search_result_rejects_whitespace_fields(
    field: str,
) -> None:
    values = {
        "title": "Dental",
        "content": "Content",
        "url": "https://example.com",
    }

    values[field] = "   "

    with pytest.raises(
        ValueError,
    ):
        WebSearchResult(
            **values,
        )


@pytest.mark.parametrize(
    "field",
    [
        "title",
        "content",
        "url",
    ],
)
def test_web_search_result_rejects_none_fields(
    field: str,
) -> None:
    values = {
        "title": "Dental",
        "content": "Content",
        "url": "https://example.com",
    }

    values[field] = None

    with pytest.raises(
        (TypeError, ValueError),
    ):
        WebSearchResult(
            **values,
        )


def test_web_search_result_is_immutable() -> None:
    result = WebSearchResult(
        title="Dental",
        content="Content",
        url="https://example.com",
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.title = "Changed"


@pytest.mark.parametrize(
    "values",
    [
        {
            "title": 123,
            "content": "Content",
            "url": "https://example.com",
        },
        {
            "title": "Title",
            "content": 123,
            "url": "https://example.com",
        },
        {
            "title": "Title",
            "content": "Content",
            "url": 123,
        },
    ],
)
def test_web_search_result_rejects_invalid_types(
    values: dict[str, object],
) -> None:
    with pytest.raises(
        (TypeError, ValueError, AttributeError),
    ):
        WebSearchResult(
            **values,
        )
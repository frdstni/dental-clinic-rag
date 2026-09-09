from unittest.mock import Mock, patch

import pytest

from dental_rag.config.settings import settings
from dental_rag.llm.openai_llm import OpenAILLM


def make_response(
    content: str | None = "generated response",
) -> Mock:
    mock_response = Mock()

    mock_message = Mock()
    mock_message.content = content

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_response.choices = [
        mock_choice,
    ]

    return mock_response


def test_openai_llm_returns_generated_content() -> None:
    mock_response = make_response(
        "Dental implants replace missing teeth."
    )

    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        llm = OpenAILLM()

        result = llm.generate(
            "Explain dental implants"
        )

    assert (
        result
        == "Dental implants replace missing teeth."
    )


def test_openai_llm_sends_correct_prompt_and_model() -> None:
    mock_response = make_response()

    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        llm = OpenAILLM()

        llm.generate(
            "Explain root canal treatment"
        )

    mock_client.chat.completions.create.assert_called_once_with(
        model=settings.openai_chat_model,
        messages=[
            {
                "role": "user",
                "content": "Explain root canal treatment",
            }
        ],
    )


def test_openai_llm_initializes_client_with_api_key() -> None:
    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        OpenAILLM()

    mock_openai.assert_called_once_with(
        api_key=settings.openai_api_key,
    )


def test_openai_llm_rejects_empty_prompt() -> None:
    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        llm = OpenAILLM()

        with pytest.raises(
            ValueError,
            match="prompt cannot be blank",
        ):
            llm.generate("")

    (
        mock_openai.return_value
        .chat.completions.create
        .assert_not_called()
    )


def test_openai_llm_rejects_whitespace_only_prompt() -> None:
    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        llm = OpenAILLM()

        with pytest.raises(
            ValueError,
            match="prompt cannot be blank",
        ):
            llm.generate("   ")

    (
        mock_openai.return_value
        .chat.completions.create
        .assert_not_called()
    )


def test_openai_llm_rejects_response_without_choices() -> None:
    mock_response = Mock()
    mock_response.choices = []

    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        llm = OpenAILLM()

        with pytest.raises(
            ValueError,
            match="LLM returned no choices",
        ):
            llm.generate(
                "Explain dental implants"
            )


def test_openai_llm_rejects_none_content() -> None:
    mock_response = make_response(
        content=None,
    )

    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        llm = OpenAILLM()

        with pytest.raises(
            ValueError,
            match="LLM returned empty content",
        ):
            llm.generate(
                "Explain dental implants"
            )


def test_openai_llm_rejects_blank_content() -> None:
    mock_response = make_response(
        content="   ",
    )

    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        llm = OpenAILLM()

        with pytest.raises(
            ValueError,
            match="LLM returned empty content",
        ):
            llm.generate(
                "Explain dental implants"
            )


def test_openai_llm_propagates_api_error() -> None:
    with patch(
        "dental_rag.llm.openai_llm.OpenAI"
    ) as mock_openai:
        mock_client = mock_openai.return_value

        mock_client.chat.completions.create.side_effect = (
            RuntimeError(
                "OpenAI API failed"
            )
        )

        llm = OpenAILLM()

        with pytest.raises(
            RuntimeError,
            match="OpenAI API failed",
        ):
            llm.generate(
                "Explain dental implants"
            )
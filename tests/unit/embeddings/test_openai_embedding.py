from unittest.mock import Mock, patch

from dental_rag.embeddings.openai_embedding import (
    OpenAIEmbeddingModel,
)


def test_openai_embedding_returns_embeddings() -> None:
    mock_response = Mock()

    mock_item = Mock()
    mock_item.embedding = [0.1, 0.2, 0.3]

    mock_response.data = [mock_item]

    with patch(
        "dental_rag.embeddings.openai_embedding.OpenAI"
    ) as mock_openai:

        mock_client = mock_openai.return_value

        mock_client.embeddings.create.return_value = mock_response

        embedding_model = OpenAIEmbeddingModel()

        result = embedding_model.embed(
            ["clinic information"]
        )

    assert result == [
        [0.1, 0.2, 0.3]
    ]


def test_openai_embedding_sends_correct_input() -> None:
    mock_response = Mock()

    mock_item = Mock()
    mock_item.embedding = [0.5, 0.6]

    mock_response.data = [mock_item]

    with patch(
        "dental_rag.embeddings.openai_embedding.OpenAI"
    ) as mock_openai:

        mock_client = mock_openai.return_value

        mock_client.embeddings.create.return_value = mock_response

        embedding_model = OpenAIEmbeddingModel()

        embedding_model.embed(
            [
                "opening hours",
                "dental services",
            ]
        )

    mock_client.embeddings.create.assert_called_once()

    call_kwargs = (
        mock_client.embeddings.create.call_args.kwargs
    )

    assert call_kwargs["input"] == [
        "opening hours",
        "dental services",
    ]

def test_openai_embedding_returns_empty_list_for_empty_input() -> None:
    with patch(
        "dental_rag.embeddings.openai_embedding.OpenAI"
    ) as mock_openai:

        embedding_model = OpenAIEmbeddingModel()

        result = embedding_model.embed([])

    assert result == []

    mock_openai.return_value.embeddings.create.assert_not_called()

def test_openai_embedding_propagates_api_error() -> None:
    with patch(
        "dental_rag.embeddings.openai_embedding.OpenAI"
    ) as mock_openai:

        mock_client = mock_openai.return_value

        mock_client.embeddings.create.side_effect = (
            RuntimeError("OpenAI API failed")
        )

        embedding_model = OpenAIEmbeddingModel()

        try:
            embedding_model.embed(
                ["clinic information"]
            )

        except RuntimeError as error:
            assert str(error) == "OpenAI API failed"

        else:
            raise AssertionError(
                "Expected RuntimeError was not raised"
            )
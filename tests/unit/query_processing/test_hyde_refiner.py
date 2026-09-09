from unittest.mock import Mock

import pytest

from dental_rag.query_processing.refiners.hyde import (
    HyDERefiner,
)


@pytest.fixture
def llm() -> Mock:
    return Mock()


@pytest.fixture
def refiner(
    llm: Mock,
) -> HyDERefiner:
    return HyDERefiner(
        llm=llm,
    )


def test_hyde_returns_generated_document(
    refiner: HyDERefiner,
    llm: Mock,
) -> None:
    llm.generate.return_value = (
        "Dental implant complications document."
    )

    result = refiner.refine(
        "implant complications",
    )

    assert (
        result
        == "Dental implant complications document."
    )


def test_hyde_sends_correct_prompt(
    refiner: HyDERefiner,
    llm: Mock,
) -> None:
    llm.generate.return_value = (
        "Generated document"
    )

    refiner.refine(
        "implant complications",
    )

    llm.generate.assert_called_once_with(
        
            "Generate a hypothetical dental "
            "document that answers this query:\n"
            "implant complications"
        
    )


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_hyde_rejects_blank_query(
    refiner: HyDERefiner,
    query: str,
) -> None:
    with pytest.raises(ValueError):
        refiner.refine(query)


def test_hyde_propagates_llm_error(
    refiner: HyDERefiner,
    llm: Mock,
) -> None:
    llm.generate.side_effect = RuntimeError(
        "LLM failed",
    )

    with pytest.raises(
        RuntimeError,
        match="LLM failed",
    ):
        refiner.refine(
            "implant complications",
        )


def test_hyde_rejects_empty_llm_response(
    refiner: HyDERefiner,
    llm: Mock,
) -> None:
    llm.generate.return_value = ""

    with pytest.raises(
        ValueError,
        match="LLM returned empty response",
    ):
        refiner.refine(
            "implant complications",
        )


def test_hyde_rejects_whitespace_llm_response(
    refiner: HyDERefiner,
    llm: Mock,
) -> None:
    llm.generate.return_value = "   "

    with pytest.raises(
        ValueError,
        match="LLM returned empty response",
    ):
        refiner.refine(
            "implant complications",
        )
import pytest

from dental_rag.agent.answer import (
    AnswerGenerator,
)


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response

    def generate(
        self,
        prompt: str,
    ) -> str:
        return self.response


class FailingLLM:
    def generate(
        self,
        prompt: str,
    ) -> str:
        raise RuntimeError(
            "LLM failure",
        )


def test_answer_generation_from_context() -> None:
    generator = AnswerGenerator(
        llm=FakeLLM(
            "Final answer",
        ),
    )

    result = generator.run(
        {
            "query": "What is implant?",
            "context": [
                "Implant replaces missing teeth",
            ],
        },
    )

    assert result["answer"] == "Final answer"


def test_answer_generation_from_web_results() -> None:
    class Result:
        content = "Dental information"

    generator = AnswerGenerator(
        llm=FakeLLM(
            "Web answer",
        ),
    )

    result = generator.run(
        {
            "query": "Dental question",
            "web_results": [
                Result(),
            ],
        },
    )

    assert result["answer"] == "Web answer"


def test_answer_generation_rejects_empty_sources() -> None:
    generator = AnswerGenerator(
        llm=FakeLLM(
            "answer",
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        generator.run(
            {
                "query": "Question",
            },
        )


def test_answer_generation_propagates_llm_failure() -> None:
    generator = AnswerGenerator(
        llm=FailingLLM(),
    )

    with pytest.raises(
        RuntimeError,
    ):
        generator.run(
            {
                "query": "Question",
                "context": [
                    "Information",
                ],
            },
        )


def test_answer_generation_preserves_state() -> None:
    generator = AnswerGenerator(
        llm=FakeLLM(
            "Answer",
        ),
    )

    result = generator.run(
        {
            "query": "Question",
            "context": [
                "Info",
            ],
            "confidence": 0.9,
        },
    )

    assert result["confidence"] == 0.9
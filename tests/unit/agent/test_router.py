import pytest

from dental_rag.agent.models import (
    AgentIntent,
)
from dental_rag.agent.router import (
    AgentRouter,
)


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.last_prompt = ""

    def generate(
        self,
        prompt: str,
    ) -> str:
        self.last_prompt = prompt
        return self.response


@pytest.mark.parametrize(
    (
        "response",
        "expected",
    ),
    [
        (
            "clinic",
            AgentIntent.CLINIC,
        ),
        (
            "general_dental",
            AgentIntent.GENERAL_DENTAL,
        ),
    ],
)
def test_router_returns_correct_intent(
    response: str,
    expected: AgentIntent,
) -> None:
    router = AgentRouter(
        llm=FakeLLM(response),
    )

    decision = router.route(
        "What services do you provide?",
    )

    assert decision.intent == expected


@pytest.mark.parametrize(
    "response",
    [
        "",
        " ",
        "unknown",
        "clinic something else",
    ],
)
def test_router_rejects_invalid_llm_response(
    response: str,
) -> None:
    router = AgentRouter(
        llm=FakeLLM(response),
    )

    with pytest.raises(ValueError):
        router.route(
            "Dental question",
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
def test_router_rejects_empty_query(
    query: str,
) -> None:
    router = AgentRouter(
        llm=FakeLLM("clinic"),
    )

    with pytest.raises(ValueError):
        router.route(query)


def test_router_sends_query_inside_prompt() -> None:
    llm = FakeLLM(
        "clinic",
    )

    router = AgentRouter(
        llm=llm,
    )

    router.route(
        "Does your clinic offer implants?",
    )

    assert (
        "Does your clinic offer implants?"
        in llm.last_prompt
    )
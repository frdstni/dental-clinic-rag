from dental_rag.agent.state import AgentState
from dental_rag.llm.base import LLM


class AnswerGenerator:
    """
    Generate final answers from retrieved sources.
    """

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        self.llm = llm

    def run(
        self,
        state: AgentState,
    ) -> AgentState:
        prompt = self._build_prompt(
            state,
        )

        answer = self.llm.generate(
            prompt,
        )

        return {
            **state,
            "answer": answer,
        }

    def _build_prompt(
        self,
        state: AgentState,
    ) -> str:
        query = state.get(
            "query",
            "",
        )

        context = state.get(
            "context",
            [],
        )

        web_results = state.get(
            "web_results",
            [],
        )

        sources = []

        sources.extend(
            context,
        )

        sources.extend(
            [
                result.content
                for result in web_results
            ],
        )

        if not sources:
            raise ValueError(
                "No sources available for answer generation",
            )

        return f"""
You are a helpful dental assistant.

Answer the user's question using only the provided information.

Question:
{query}

Information:
{sources}

Provide a clear and accurate answer.
""".strip()
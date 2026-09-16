from dental_rag.agent.state import AgentState
from dental_rag.application.rag_service import RagService


class ClinicAgent:
    """
    Agent responsible for clinic-specific questions.
    """

    def __init__(
        self,
        rag_service: RagService,
    ) -> None:
        self.rag_service = rag_service

    def run(
        self,
        state: AgentState,
    ) -> AgentState:
        query = state.get("query")

        if query is None or not query.strip():
            raise ValueError(
                "query cannot be empty",
            )

        context = self.rag_service.retrieve_with_quality(
            query=query,
        )

        documents: list[str] = []

        for result in context.results:
            content = result.payload.get(
                "content",
            )

            if isinstance(
                content,
                str,
            ):
                documents.append(
                    content,
                )

        return {
            **state,
            "context": documents,
        }
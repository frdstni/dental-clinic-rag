from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from dental_rag.agent.nodes import (
    AgentNodes,
    route_condition,
)
from dental_rag.agent.state import AgentState


def create_agent_graph(
    nodes: AgentNodes,
) -> object:
    """
    Create LangGraph workflow.
    """

    graph = StateGraph(
        AgentState,
    )

    graph.add_node(
        "router",
        nodes.route_query,
    )

    graph.add_node(
        "clinic",
        nodes.clinic_node,
    )

    graph.add_node(
        "general_dental",
        nodes.general_dental_node,
    )

    graph.add_node(
        "answer",
        nodes.answer_node,
    )

    graph.add_edge(
        START,
        "router",
    )

    graph.add_conditional_edges(
        "router",
        route_condition,
        {
            "clinic": "clinic",
            "general_dental": "general_dental",
        },
    )

    graph.add_edge(
        "clinic",
        "answer",
    )

    graph.add_edge(
        "general_dental",
        "answer",
    )

    graph.add_edge(
        "answer",
        END,
    )

    return graph.compile()
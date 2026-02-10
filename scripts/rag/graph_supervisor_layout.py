from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class SupervisorState(TypedDict):
    user_input: str
    route: Literal["rag_agent_1", "rag_agent_2", "db_search_agent", "table_agent", "end"]


def user_input_node(state: SupervisorState) -> SupervisorState:
    return state


def supervisor_node(state: SupervisorState) -> SupervisorState:
    return state


def rag_agent_1_node(state: SupervisorState) -> SupervisorState:
    return state


def rag_agent_2_node(state: SupervisorState) -> SupervisorState:
    return state


def db_search_agent_node(state: SupervisorState) -> SupervisorState:
    return state


def table_agent_node(state: SupervisorState) -> SupervisorState:
    return state


def route_from_supervisor(state: SupervisorState) -> str:
    route = state.get("route", "end")
    return route


def build_graph():
    graph = StateGraph(SupervisorState)

    graph.add_node("user_input", user_input_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_agent_1", rag_agent_1_node)
    graph.add_node("rag_agent_2", rag_agent_2_node)
    graph.add_node("db_search_agent", db_search_agent_node)
    graph.add_node("table_agent", table_agent_node)

    graph.add_edge(START, "user_input")
    graph.add_edge("user_input", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "rag_agent_1": "rag_agent_1",
            "rag_agent_2": "rag_agent_2",
            "db_search_agent": "db_search_agent",
            "table_agent": "table_agent",
            "end": END,
        },
    )
    # Multi-agent orchestration: each worker returns control to supervisor.
    graph.add_edge("rag_agent_1", "supervisor")
    graph.add_edge("rag_agent_2", "supervisor")
    graph.add_edge("db_search_agent", "supervisor")
    graph.add_edge("table_agent", "supervisor")

    return graph.compile()


def main() -> int:
    app = build_graph()
    img = app.get_graph().draw_mermaid_png()
    with open("graph_img_supervisor_1.png", "wb") as f:
        f.write(img)
    print("[INFO] saved graph_img_supervisor_1.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

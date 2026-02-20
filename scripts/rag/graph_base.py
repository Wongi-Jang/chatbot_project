import argparse
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from rich.console import Console
from rich.markdown import Markdown

from cli_args import parse_graph_args
from helper import build_vector_store, set_retriever_mode, render_text_with_graphs
from state import GraphState
from node import (
    routing,
    query_gen,
    retrieve,
    rerank,
    relevant,
    supporting,
    scoring,
    answering,
    requery,
    user_input,
    summarize,
)
from conditions import (
    decision,
    is_quit,
    should_retrieve,
    relevant_decision,
    support_decision,
)


class PipelineManager:
    def __init__(self, state_cls, nodes: dict[str, callable]):
        self.state_cls = state_cls
        self.nodes = nodes
        self.edges: list[tuple[object, object]] = []
        self.conditional_edges: list[tuple[object, callable, dict]] = []

    def add_edge(self, start, end) -> None:
        self.edges.append((start, end))

    def add_conditional_edges(self, source, condition, mapping: dict) -> None:
        self.conditional_edges.append((source, condition, mapping))

    def build(self) -> StateGraph:
        graph = StateGraph(self.state_cls)
        for name, fn in self.nodes.items():
            graph.add_node(name, fn)
        for start, end in self.edges:
            graph.add_edge(start, end)
        for source, condition, mapping in self.conditional_edges:
            graph.add_conditional_edges(source, condition, mapping)
        return graph

    def compile(self, checkpointer=None):
        return self.build().compile(checkpointer=checkpointer)

    def _node_id(self, node) -> str:
        if node is START:
            return "START"
        if node is END:
            return "END"
        return str(node)

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for start, end in self.edges:
            lines.append(f"  {self._node_id(start)} --> {self._node_id(end)}")
        for source, _condition, mapping in self.conditional_edges:
            src = self._node_id(source)
            for label, target in mapping.items():
                lines.append(f"  {src} -->|{label}| {self._node_id(target)}")
        return "\n".join(lines)

    def save_mermaid(self, path: str) -> None:
        content = self.to_mermaid()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def save_png(self, path: str) -> None:
        graph = self.build().compile()
        try:
            img_data = graph.get_graph().draw_mermaid_png()
            with open(path, "wb") as f:
                f.write(img_data)
            print("그래프 이미지가 저장되었습니다!")
        except Exception as e:
            print(f"시각화 실패: {e}")


def _extract_answer_text(event: dict) -> str | None:
    answer_event = event.get("answer")
    if isinstance(answer_event, dict):
        answer_text = answer_event.get("answer")
        if isinstance(answer_text, str):
            return answer_text
    return None


def create_graph(
    enable_scoring=True, retriever_mode="basic", is_web=False, checkpointer=None
):
    set_retriever_mode(retriever_mode, force_rebuild=False)

    nodes = {
        "routing": routing,
        "query_gen": query_gen,
        "retrieve": retrieve,
        "rerank": rerank,
        "relevant": relevant,
        "supporting": supporting,
        "answer": answering,
        "requery": requery,
        "summarize": summarize,
    }
    if enable_scoring:
        nodes["scoring"] = scoring

    if not is_web:
        nodes["user_input"] = user_input

    pipeline = PipelineManager(GraphState, nodes)

    if is_web:
        # Web Mode: Request-Response driven
        pipeline.add_edge(START, "routing")
        pipeline.add_edge("summarize", END)
    else:
        # CLI Mode: Infinite Input Loop
        pipeline.add_edge(START, "user_input")
        pipeline.add_conditional_edges(
            "user_input",
            is_quit,
            {
                True: END,
                False: "routing",
            },
        )
        pipeline.add_edge("summarize", "user_input")

    # Common Logic
    pipeline.add_conditional_edges(
        "routing",
        should_retrieve,
        {
            True: "query_gen",
            False: "answer",
        },
    )

    pipeline.add_edge("query_gen", "retrieve")
    pipeline.add_edge("retrieve", "rerank")
    pipeline.add_edge("rerank", "relevant")

    pipeline.add_conditional_edges(
        "relevant",
        relevant_decision,
        {
            "answer": "answer",
            "requery": "requery",
        },
    )

    if enable_scoring:
        pipeline.add_conditional_edges(
            "answer",
            should_retrieve,
            {
                True: "supporting",
                False: "scoring",
            },
        )
        pipeline.add_conditional_edges(
            "supporting",
            support_decision,
            {
                "answer": "answer",
                "scoring": "scoring",
            },
        )
        pipeline.add_conditional_edges(
            "scoring",
            decision,
            {
                "end": "summarize",
                "requery": "requery",
            },
        )
    else:
        pipeline.add_conditional_edges(
            "answer",
            should_retrieve,
            {
                True: "supporting",
                False: "summarize",
            },
        )
        pipeline.add_conditional_edges(
            "supporting",
            support_decision,
            {
                "answer": "answer",
                "scoring": "summarize",
            },
        )
    pipeline.add_edge("requery", "retrieve")

    return pipeline.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    args = parse_graph_args()

    if args.store:
        build_vector_store(True)
        raise SystemExit(0)

    # set_retriever_mode is called inside create_graph, but for main we might want explicitly pass args
    # actually create_graph handles it.

    memory = MemorySaver()
    app = create_graph(
        enable_scoring=args.scoring,
        retriever_mode=args.retriever,
        is_web=False,
        checkpointer=memory,
    )

    # Visualization (re-instantiate pipeline manager logic or abstract it?
    # pipeline.save_png("graph_img.png") - create_graph returns compiled graph directly.
    # To save PNG we need the state graph object before compile or extract it.
    # For now, let's skip saving PNG in main execution or do it via graph getter
    try:
        img_data = app.get_graph().draw_mermaid_png()
        with open("graph_img.png", "wb") as f:
            f.write(img_data)
        print("Graph image saved to graph_img.png")
    except Exception:
        pass
    config = {
        "configurable": {"thread_id": "user_session_1"},
        "recursion_limit": args.max_nodes,
    }

    initial_input = {
        "history": [],
    }

    for event in app.stream(initial_input, config=config):
        if args.answer:
            answer_text = _extract_answer_text(event)
            if answer_text is not None:
                segments = render_text_with_graphs(answer_text)
                if Console and Markdown:
                    for seg in segments:
                        if seg["type"] == "text":
                            Console().print(Markdown(seg["content"]))
                        else:
                            print(seg["content"])
                else:
                    print(answer_text)
            continue
        if not args.summary and "summarize" in event.keys():
            continue
        if "user_input" in event.keys():
            continue
        if args.docs or "retrieve" not in event.keys():
            print(event)

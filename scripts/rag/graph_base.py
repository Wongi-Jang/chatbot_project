import argparse
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from rich.console import Console
from rich.markdown import Markdown


from helper import build_vector_store, render_text_with_graphs
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
    set_retriever_mode,
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
    enable_scoring: bool = False,
    retriever_mode: str = "basic",
    force_rebuild_parent: bool = False,
    is_interactive: bool = True,
) -> object:
    set_retriever_mode(
        retriever_mode,
        force_rebuild_parent=force_rebuild_parent,
    )

    nodes = {
        "routing": routing,
        "query_gen": query_gen,
        "retrieve": retrieve,
        "rerank": rerank,
        "relevant": relevant,
        "supporting": supporting,
        "answer": answering,
        "requery": requery,
        "user_input": user_input,
        "summarize": summarize,
    }
    if enable_scoring:
        nodes["scoring"] = scoring

    pipeline = PipelineManager(GraphState, nodes)

    pipeline.add_edge(START, "user_input")

    pipeline.add_conditional_edges(
        "user_input",
        is_quit,
        {
            True: END,
            False: "routing",
        },
    )
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
    if is_interactive:
        pipeline.add_edge("summarize", "user_input")
    else:
        pipeline.add_edge("summarize", END)

    pipeline.add_edge("requery", "retrieve")

    memory = MemorySaver()
    app = pipeline.compile(checkpointer=memory)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="graph pipeline")
    parser.add_argument("--store", action="store_true", help="initialize Chroma DB")
    parser.add_argument(
        "--scoring",
        action="store_true",
        help="enable scoring node in pipeline",
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="print docs-related events in stream output",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print summarize events in stream output",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="print answer events only",
    )
    parser.add_argument(
        "--retriever",
        choices=["basic", "parent"],
        default="basic",
        help="retrieval backend mode",
    )
    parser.add_argument(
        "--rebuild-parent",
        action="store_true",
        help="rebuild parent retriever index from source PDFs",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="maximum number of nodes to execute per run",
    )
    args = parser.parse_args()

    if args.max_nodes <= 0:
        parser.error("--max-nodes must be a positive integer")

    if args.store:
        build_vector_store(True)
        raise SystemExit(0)

    app = create_graph(
        enable_scoring=args.scoring,
        retriever_mode=args.retriever,
        force_rebuild_parent=args.rebuild_parent,
    )
    try:
        img_data = app.get_graph().draw_mermaid_png()
        with open("graph_img.png", "wb") as f:
            f.write(img_data)
        print("그래프 이미지가 저장되었습니다!")
    except Exception as e:
        print(f"시각화 실패: {e}")
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

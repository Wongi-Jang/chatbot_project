import argparse


def build_graph_parser() -> argparse.ArgumentParser:
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
        "--rebuild",
        action="store_true",
        help="rebuild retriever index from source PDFs",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="maximum number of nodes to execute per run",
    )
    return parser


def parse_graph_args() -> argparse.Namespace:
    parser = build_graph_parser()
    args = parser.parse_args()
    if args.max_nodes <= 0:
        parser.error("--max-nodes must be a positive integer")
    return args

import os
import sys

# Ensure api keys are loaded and setup is complete
from helper import load_api_key
from state import GraphState
from node import query_gen
import json


def test_query_expansion():
    # Setup mock state
    state = GraphState(
        question="BBQ 브랜드 가맹점 운영 시 사용 가능한 지식재산권 종류 및 등록 번호",
        retrieve_types=["franchise", "law"],
        history=[],
        query={},
        docs_raw=[],
        docs="",
        answer="",
        do_retrieve=True,
        db_meta={},
        isrel={},
        query_state={},
        issup="",
        irrelevant_count=0,
        relevant_loop=0,
        supporting_loop=0,
        scoring_loop=0,
        relevant_feedback={},
        supporting_feedback="",
        ispass="",
        no_docs=False,
        target_brands=[],
    )

    print("--- Running query_gen with expanded prompts ---")
    next_state = query_gen(state)

    print("\nResulting Queries from query_gen:")
    print(json.dumps(next_state.get("query"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_query_expansion()

from state import GraphState


MAX_RELEVANT_LOOP = 3
MAX_SUPPORT_LOOP = 3
MAX_SCORING_LOOP = 3


def decision(state: GraphState) -> str:
    if state.get("ispass") == "Pass":
        state["scoring_loop"] = 0
        return "end"
    if state.get("scoring_loop", 0) >= MAX_SCORING_LOOP:
        print("scoring loop overflow")
        state["scoring_loop"] = 0
        return "end"
    return "requery"


def is_quit(state: GraphState) -> bool:
    return state["question"] == "exit" or state["question"] == "quit"


def should_retrieve(state: GraphState) -> bool:
    return bool(state.get("do_retrieve"))


# def relevant_decision(state: GraphState) -> str:
#     if state.get("isrel") == "irrelevant":
#         return "requery"
#     return "answer"


def relevant_decision(state: GraphState) -> str:
    query_state = state.get("query_state") or {}
    if not query_state or not all(query_state.values()):
        return "requery"
    return "answer"


def support_decision(state: GraphState) -> str:
    if state.get("issup") == "no support":
        if state.get("supporting_loop", 0) >= MAX_SUPPORT_LOOP:
            print("supporting loop overflow")
            state["supporting_loop"] = 0
            return "scoring"
        return "requery"
    state["supporting_loop"] = 0
    return "scoring"

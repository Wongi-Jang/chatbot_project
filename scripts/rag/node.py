import json
from collections.abc import Iterable
from pathlib import Path
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

from helper import (
    build_parent_retrievers,
    build_vector_store,
    load_openai,
    load_prompts,
)
from preprocess import docs_to_context, history_to_context
from state import GraphState, ScoreSummary, RelevantSummary, SupportSummary


PROMPTS = load_prompts(str(Path(__file__).with_name("prompts.yaml")))
VECTOR_DBS = build_vector_store(False)
PARENT_RETRIEVERS = None


MAX_IRRELEVANT = 3
NO_DOCS_MESSAGE = "사용자 입력에 관련된 정보가 DB에 없습니다"
RETRIEVER_MODE = "basic"
FORCE_REBUILD_PARENT = False


def _safe_json_dict(raw: str, warn: str | None = None) -> dict:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    if warn:
        print(warn)
    return {}


def set_retriever_mode(mode: str, force_rebuild_parent: bool = False) -> None:
    global RETRIEVER_MODE, FORCE_REBUILD_PARENT, PARENT_RETRIEVERS
    if mode not in {"basic", "parent"}:
        raise ValueError("retriever mode must be one of: basic, parent")
    RETRIEVER_MODE = mode
    FORCE_REBUILD_PARENT = force_rebuild_parent
    PARENT_RETRIEVERS = None


def _get_parent_retrievers():
    global PARENT_RETRIEVERS
    if PARENT_RETRIEVERS is None:
        # print("initialize parent document retrievers...")
        PARENT_RETRIEVERS = build_parent_retrievers(
            force_rebuild=FORCE_REBUILD_PARENT
        )
    return PARENT_RETRIEVERS


def _dedupe_docs(docs: Iterable[Document]) -> list[Document]:
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (
            doc.page_content,
            json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False, default=str),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def vector_db_metadata_tool(sample_size: int = 10) -> dict:
    meta_info = {}
    for key, db in VECTOR_DBS.items():
        sample_keys = set()
        count = None
        try:
            data = db.get(include=["metadatas"], limit=sample_size)
            metadatas = data.get("metadatas") or []
            for md in metadatas:
                sample_keys.update(md.keys())
        except Exception:
            metadatas = []
        try:
            count = db._collection.count()
        except Exception:
            count = None
        meta_info[key] = {
            "sample_keys": sorted(sample_keys),
            "sample_key_count": len(sample_keys),
            "count": count,
        }
    return meta_info


def user_input(state: GraphState) -> GraphState:
    question = input("You:").strip()
    # Reset per-question transient states to avoid cross-turn carryover.
    return GraphState(
        question=question,
        query={},
        docs_raw=[],
        docs="",
        answer="",
        do_retrieve=False,
        retrieve_types=[],
        db_meta={},
        isrel="",
        issup="",
        irrelevant_count=0,
        relevant_loop=0,
        supporting_loop=0,
        scoring_loop=0,
        feedback="",
        ispass="",
        no_docs=False,
    )


def routing(state: GraphState) -> GraphState:
    llm = load_openai()
    question = state["question"]
    prompt = PROMPTS["routing"]["sys_prompt"].format(
        question=question,
        history=history_to_context(state.get("history", [])),
    )
    raw = llm.invoke(prompt).content.strip()
    data = _safe_json_dict(raw, warn="can not extract json at routing node")

    do_retrieve = bool(data.get("do_retrieve"))
    retrieve_types = data.get("retrieve_types") or []
    if not isinstance(retrieve_types, list):
        retrieve_types = []

    # Always reset retrieval artifacts at routing step to avoid stale docs reuse.
    next_state = GraphState(
        do_retrieve=do_retrieve,
        retrieve_types=retrieve_types,
        db_meta={},
        query={},
        docs_raw=[],
        docs="",
    )
    return next_state


def query_gen(state: GraphState) -> GraphState:
    llm = load_openai()
    template = PROMPTS["query_gen"]["template"]
    prompt = PromptTemplate.from_template(template=template)
    targets = ",".join(state.get("retrieve_types", [])) or "none"
    formatted_input = (
        prompt.format(
            user_question=state["question"],
            history=history_to_context(state.get("history", [])),
        )
        + f"\n\n검색 대상 DB: {targets}"
    )
    raw = llm.invoke(formatted_input).content.strip()
    data = _safe_json_dict(raw)
    return GraphState(query=data)


def _retrieve_from_db(
    query: str, db_key: str, k: int = 3, score_threshold: float = 0.5
) -> list[tuple]:
    db = VECTOR_DBS.get(db_key)
    if db is None:
        return []
    results = db.similarity_search_with_score(query=query, k=k)
    return [pair for pair in results if pair[1] >= score_threshold]


def _retrieve_parent_docs(query: str, db_key: str, k: int = 3) -> list[Document]:
    retriever = _get_parent_retrievers().get(db_key)
    if retriever is None:
        return []
    docs = retriever.invoke(query)
    return list(docs)[:k]


def retrieve(state: GraphState) -> GraphState:
    queries = state.get("query", {}) or {}
    if RETRIEVER_MODE == "parent":
        docs: list[Document] = []
        for db_key, q in queries.items():
            if not q:
                continue
            docs.extend(_retrieve_parent_docs(q, db_key))
        docs = _dedupe_docs(docs)
    else:
        docs_with_scores: list[tuple] = []
        for db_key, q in queries.items():
            if not q:
                continue
            docs_with_scores.extend(_retrieve_from_db(q, db_key))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        docs = [d for d, _s in docs_with_scores]
    return GraphState(docs_raw=docs, docs=docs_to_context(docs))


def relevant(state: GraphState) -> GraphState:
    current_count = state.get("irrelevant_count", 0)
    if current_count >= MAX_IRRELEVANT:
        # print("relevant overflow")
        return GraphState(
            isrel="relevant",
            docs=NO_DOCS_MESSAGE,
            irrelevant_count=0,
            relevant_loop=0,
            no_docs=True,
        )

    llm = load_openai()
    sys_message = PROMPTS["relevant"]["sys_message"]
    parser = PydanticOutputParser(pydantic_object=RelevantSummary)
    sys_message = PromptTemplate.from_template(sys_message)
    prompt = sys_message.partial(format=parser.get_format_instructions())
    formatted_input = prompt.format(
        documents=state.get("docs", ""),
        user_input=state["question"],
    )
    output = llm.invoke(formatted_input)
    parsed_output = parser.parse(output.content)
    next_count = (
        current_count + 1 if parsed_output.isrel == "irrelevant" else current_count
    )
    loop_count = state.get("relevant_loop", 0)
    next_loop = loop_count + 1 if parsed_output.isrel == "irrelevant" else 0
    return GraphState(
        isrel=parsed_output.isrel,
        feedback=parsed_output.feedback,
        irrelevant_count=next_count,
        relevant_loop=next_loop,
        no_docs=False,
    )


def supporting(state: GraphState) -> GraphState:
    if state.get("no_docs", False):
        return GraphState(
            issup="fully supported",
            supporting_loop=0,
            no_docs=True,
        )

    llm = load_openai()
    sys_message = PROMPTS["supporting"]["sys_message"]
    parser = PydanticOutputParser(pydantic_object=SupportSummary)
    sys_message = PromptTemplate.from_template(sys_message)
    prompt = sys_message.partial(format=parser.get_format_instructions())
    formatted_input = prompt.format(
        documents=state.get("docs", ""),
        answer=state.get("answer", ""),
        user_input=state.get("question", ""),
    )
    output = llm.invoke(formatted_input)
    parsed_output = parser.parse(output.content)
    loop_count = state.get("supporting_loop", 0)
    next_loop = loop_count + 1 if parsed_output.issup == "no support" else 0
    return GraphState(
        issup=parsed_output.issup,
        feedback=parsed_output.feedback,
        supporting_loop=next_loop,
        no_docs=False,
    )


def rerank(state: GraphState) -> GraphState:
    docs = state.get("docs_raw", [])
    # if not docs:
    #     return GraphState(docs="")

    # try:
    #     from langchain_community.document_compressors import CohereRerank
    # except Exception:
    #     return GraphState(docs=docs_to_context(docs))

    # api_key = load_api_key("COHERE_API_KEY")
    # if not api_key:
    #     return GraphState(docs=docs_to_context(docs))

    # reranker = CohereRerank(cohere_api_key=api_key, top_n=min(5, len(docs)))
    # reranked = reranker.compress_documents(docs, query=state["question"])
    # return GraphState(docs_raw=reranked, docs=docs_to_context(reranked))
    return GraphState(docs_raw=docs, docs=docs_to_context(docs))

def answering(state: GraphState) -> GraphState:
    llm = load_openai()
    sys_message = SystemMessagePromptTemplate.from_template(
        PROMPTS["answering"]["sys_message"]
    )
    human_message = HumanMessagePromptTemplate.from_template("{user_input}")
    docs_message = ChatMessagePromptTemplate.from_template(
        role="documents", template="{documents}"
    )
    history_message = AIMessagePromptTemplate.from_template("{history}")
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_message,
            history_message,
            docs_message,
            human_message,
        ]
    )
    var = {
        "user_input": state["question"],
        "history": history_to_context(state["history"]),
        "documents": state.get("docs", ""),
    }
    formatted_message = chat_prompt.format(**var)
    answer = llm.invoke(formatted_message)
    return GraphState(answer=answer.content)


def scoring(state: GraphState) -> GraphState:
    llm = load_openai()

    sys_message = PROMPTS["scoring"]["sys_message"]
    parser = PydanticOutputParser(pydantic_object=ScoreSummary)
    sys_message = PromptTemplate.from_template(sys_message)
    prompt = sys_message.partial(format=parser.get_format_instructions())
    formatted_input = prompt.format(
        documents=state.get("docs", ""),
        user_input=state["question"],
        answer=state["answer"],
        query=state.get("query", ""),
    )
    output = llm.invoke(formatted_input)
    parsed_output = parser.parse(output.content)
    loop_count = state.get("scoring_loop", 0)
    next_loop = loop_count + 1 if parsed_output.score != "Pass" else 0
    return GraphState(
        feedback=parsed_output.feedback,
        ispass=parsed_output.score,
        scoring_loop=next_loop,
    )


def requery(state: GraphState) -> GraphState:
    llm = load_openai()
    sys_prompt = PROMPTS["requery"]["sys_prompt"]
    formatted_message = sys_prompt.format(
        prev_query=state.get("query", ""),
        user_input=state["question"],
        feedback=state["feedback"],
    )
    raw = llm.invoke(formatted_message).content.strip()
    data = _safe_json_dict(raw)
    return GraphState(query=data)


def summarize(state: GraphState) -> GraphState:
    prev_question = state["question"]
    prev_answer = state["answer"]
    return GraphState(
        history=[
            {
                "question": prev_question,
                "answer": prev_answer,
            }
        ]
    )

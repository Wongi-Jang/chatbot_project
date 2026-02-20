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
    get_retrievers,
    load_openai,
    load_prompts,
    safe_json_dict,
)
from preprocess import (
    build_requery_inputs,
    dedupe_docs,
    docs_to_context,
    docs_to_query_context,
    history_to_context,
)
from state import (
    GraphState,
    QuerySummary,
    ScoreSummary,
    RelevantSummary,
    SupportSummary,
)

from flashrank import Ranker, RerankRequest
from rich.console import Console


PROMPTS = load_prompts(str(Path(__file__).with_name("prompts.yaml")))
LLM_CONFIG = load_prompts(str(Path(__file__).with_name("llm_config.yaml")))

MAX_IRRELEVANT = 3


# def vector_db_metadata_tool(sample_size: int = 10) -> dict:
#     meta_info = {}
#     for key, db in VECTOR_DBS.items():
#         sample_keys = set()
#         count = None
#         try:
#             data = db.get(include=["metadatas"], limit=sample_size)
#             metadatas = data.get("metadatas") or []
#             for md in metadatas:
#                 sample_keys.update(md.keys())
#         except Exception:
#             metadatas = []
#         try:
#             count = db._collection.count()
#         except Exception:
#             count = None
#         meta_info[key] = {
#             "sample_keys": sorted(sample_keys),
#             "sample_key_count": len(sample_keys),
#             "count": count,
#         }
#     return meta_info


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


def routing(state: GraphState) -> GraphState:
    llm = load_openai(
        model=LLM_CONFIG["routing"]["model"],
        temperature=float(LLM_CONFIG["routing"]["temperature"]),
    )
    question = state["question"]
    prompt = PROMPTS["routing"]["sys_prompt"].format(
        question=question,
        history=history_to_context(state.get("history", [])),
    )
    raw = llm.invoke(prompt).content.strip()
    data = safe_json_dict(raw, warn="can not extract json at routing node")

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
        isrel={},
        query_state={},
        relevant_feedback={},
        supporting_feedback="",
        target_brands=[],
    )
    return next_state


def query_gen(state: GraphState) -> GraphState:
    llm = load_openai(
        model=LLM_CONFIG["query_gen"]["model"],
        temperature=float(LLM_CONFIG["query_gen"]["temperature"]),
    )
    template = PROMPTS["query_gen"]["template"]
    parser = PydanticOutputParser(pydantic_object=QuerySummary)
    prompt = PromptTemplate.from_template(template=template).partial(
        format=parser.get_format_instructions()
    )
    targets = ",".join(state.get("retrieve_types", [])) or "none"
    formatted_input = prompt.format(
        user_question=state["question"],
        history=history_to_context(state.get("history", [])),
    )
    formatted_input = f"{formatted_input}\n\n검색 대상 DB: {targets}"
    output = llm.invoke(formatted_input)
    parsed = parser.parse(output.content)
    return GraphState(
        query={
            "law": parsed.law_query,
            "franchise": [fq.model_dump() for fq in parsed.franchise_query],
        },
        target_brands=parsed.target_brands,
    )


def _retrieve_docs(
    query: str,
    db_key: str,
    k: int = 3,
    score_threshold: float = 0.5,
    filter: dict | None = None,
) -> list[tuple]:
    retriever = get_retrievers().get(db_key)
    if retriever is None:
        return []
    if filter:
        return retriever.invoke(
            query, k=k, score_threshold=score_threshold, filter=filter
        )
    return retriever.invoke(query, k=k, score_threshold=score_threshold)


def retrieve(state: GraphState) -> GraphState:
    queries = state.get("query", {}) or {}
    target_brands = state.get("target_brands", []) or []

    franchise_filter = None
    if target_brands:
        if len(target_brands) == 1:
            franchise_filter = {"brand": target_brands[0]}
        else:
            franchise_filter = {"brand": {"$in": target_brands}}

    docs_with_scores: dict[str, list[tuple]] = {}
    for db_key, query_list in queries.items():
        if not query_list:
            continue
        if not isinstance(query_list, list):
            continue

        for q_item in query_list:
            q_str = ""
            current_filter = None

            if isinstance(q_item, dict):
                q_str = q_item.get("query", "")
                brand = q_item.get("brand", "")
                if brand and db_key == "franchise":
                    current_filter = {"brand": brand}
                elif db_key == "franchise" and target_brands:
                    if len(target_brands) == 1:
                        current_filter = {"brand": target_brands[0]}
                    else:
                        current_filter = {"brand": {"$in": target_brands}}
            elif isinstance(q_item, str):
                q_str = q_item
                current_filter = franchise_filter if db_key == "franchise" else None

            if not isinstance(q_str, str) or not q_str.strip():
                continue

            docs_with_scores.setdefault(q_str, []).extend(
                _retrieve_docs(q_str, db_key, filter=current_filter)
            )

    docs: dict[str, list[Document]] = {}
    for q, pairs in docs_with_scores.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        per_query_docs = [d for d, _s in pairs]
        docs[q] = dedupe_docs(per_query_docs)
    return GraphState(docs_raw=docs, docs=docs_to_query_context(docs))


def relevant(state: GraphState) -> GraphState:
    current_count = state.get("irrelevant_count", 0)
    query_docs = state.get("docs_raw", {}) or {}
    if current_count >= MAX_IRRELEVANT:
        overflow_queries = {q: True for q in query_docs.keys()}
        if not overflow_queries:
            overflow_queries = {"_no_query": True}
        return GraphState(
            isrel={q: [] for q in overflow_queries.keys()},
            query_state=overflow_queries,
            relevant_feedback={},
            docs={},
            irrelevant_count=0,
            relevant_loop=0,
            no_docs=True,
        )
    llm = load_openai(
        model=LLM_CONFIG["relevant"]["model"],
        temperature=float(LLM_CONFIG["relevant"]["temperature"]),
    )
    parser = PydanticOutputParser(pydantic_object=RelevantSummary)
    prompt = PromptTemplate.from_template(PROMPTS["relevant"]["sys_message"]).partial(
        format=parser.get_format_instructions()
    )
    docs_by_query = state.get("docs", {}) or {}
    isrel: dict[str, list[str]] = {}
    feedback: dict[str, str] = {}
    query_state: dict[str, bool] = {}

    for query_text, docs in query_docs.items():
        formatted_input = prompt.format(
            query=query_text,
            documents=docs_by_query.get(query_text, ""),
            user_input=state["question"],
        )
        output = llm.invoke(formatted_input)
        parsed_output = parser.parse(output.content)
        labels = parsed_output.isrel
        isrel[query_text] = labels
        has_relevant = False
        for i in range(len(docs)):
            if i < len(labels) and labels[i] == "relevant":
                has_relevant = True
                break
        query_state[query_text] = has_relevant
        if has_relevant:
            continue
        else:
            fallback = "해당 쿼리와 관련성이 높은 문서를 찾지 못했습니다."
            text = (
                parsed_output.relevant_feedback.strip()
                if isinstance(parsed_output.relevant_feedback, str)
                else ""
            )
            feedback[query_text] = text or fallback

    all_queries_passed = bool(query_state) and all(query_state.values())
    next_count = current_count + 1 if not all_queries_passed else 0
    loop_count = state.get("relevant_loop", 0)
    next_loop = loop_count + 1 if not all_queries_passed else 0

    return GraphState(
        isrel=isrel,
        query_state=query_state,
        relevant_feedback=feedback,
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

    llm = load_openai(
        model=LLM_CONFIG["supporting"]["model"],
        temperature=float(LLM_CONFIG["supporting"]["temperature"]),
    )
    sys_message = PROMPTS["supporting"]["sys_message"]
    parser = PydanticOutputParser(pydantic_object=SupportSummary)
    sys_message = PromptTemplate.from_template(sys_message)
    prompt = sys_message.partial(format=parser.get_format_instructions())
    formatted_input = prompt.format(
        documents=docs_to_context(state.get("docs_raw", {}), include_query=True),
        answer=state.get("answer", ""),
        user_input=state.get("question", ""),
    )
    output = llm.invoke(formatted_input)
    parsed_output = parser.parse(output.content)
    loop_count = state.get("supporting_loop", 0)
    next_loop = loop_count + 1 if parsed_output.issup == "no support" else 0
    return GraphState(
        issup=parsed_output.issup,
        supporting_feedback=parsed_output.feedback,
        supporting_loop=next_loop,
        no_docs=False,
    )


def _rerank_list(docs: list, query: str) -> list:
    """Helper to rerank a list of documents or dicts."""
    if not docs:
        return []

    # If Flashrank is missing/failed
    if Ranker is None:
        return docs

    try:
        ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache"
        )
    except Exception as e:
        if Console:
            Console().print(f"[red]Flashrank init failed: {e}[/red]")
        return docs

    passages = []
    for i, doc in enumerate(docs):
        if isinstance(doc, dict):
            text = doc.get("page_content", "")
            meta = doc.get("metadata", {})
        else:
            text = doc.page_content
            meta = doc.metadata
        passages.append({"id": str(i), "text": text, "meta": meta})

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    reranked_docs = []
    for res in results:
        meta = res.get("meta", {})
        score = res.get("score")
        if score is not None:
            score = float(score)
        meta["rerank_score"] = score
        # Reconstruct as dicts since that's what we likely want downstream?
        # Actually retrieve returns dicts with 'page_content' and 'metadata' usually?
        # Wait, retrieve returns Documents or dicts?
        # _retrieve_docs returns list[tuple[Document, float]] -> retrieve makes list[Document]
        # So we should probably return Documents if possible, but Flashrank returns text/meta.
        # Let's return dicts to be safe as per original code,
        # OR recreate Document objects if downstream expects them.
        # Original rerank returned dicts: {"page_content": ..., "metadata": ...}
        reranked_docs.append(Document(page_content=res["text"], metadata=meta))

    return reranked_docs[:5]


def rerank(state: GraphState) -> GraphState:
    print("DEBUG: Entering rerank node")
    docs_raw = state.get("docs_raw", [])
    question = state["question"]

    if not docs_raw:
        return GraphState(docs="")

    # If dict (per query)
    if isinstance(docs_raw, dict):
        new_docs_raw = {}
        for query_key, doc_list in docs_raw.items():
            print(f"DEBUG: Reranking for query: {query_key}")
            new_docs_raw[query_key] = _rerank_list(doc_list, question)

        return GraphState(
            docs_raw=new_docs_raw, docs=docs_to_query_context(new_docs_raw)
        )

    # If list (global / legacy)
    else:
        print("DEBUG: Reranking flat list")
        reranked = _rerank_list(docs_raw, question)
        return GraphState(
            docs_raw=reranked,
            docs=docs_to_context(reranked),  # Note: docs_to_context expects diff sig?
            # preprocess.py: docs_to_context(docs: dict[str, list[Document]], ...)
            # Wait, docs_to_context signature in preprocess.py is:
            # def docs_to_context(docs: dict[str, list[Document]], include_query: bool = True) -> str:
            # So passing a list to docs_to_context will FAIL.
            # We must wrap it? Or use a different function?
            # Looking at preprocess.py again...
            # The commented out version took list. The active one takes dict.
            # So we MUST return a dict structure even if input was list?
            # Or wraps list in dummy query?
        )
        # Actually, since retrieve always returns dict now, we might not need list path.
        # But for safety, if it is a list, we wrap it.

        # However, let's look at preprocess.py again to be sure.
        # def docs_to_context(docs: dict[str, list[Document]], ...)

        # So if we have a list, we can't call docs_to_context directly with it.
        # But previous rerank implementation called docs_to_context(reranked_docs).
        # This implies previous implementation was broken too if docs_to_context changed?
        # Or maybe I misread preprocess.py.
        # Let's assume input is always dict from retrieve.

        return GraphState(docs="")  # Should not happen if retrieved correctly


def answering(state: GraphState) -> GraphState:
    llm = load_openai(
        model=LLM_CONFIG["answering"]["model"],
        temperature=float(LLM_CONFIG["answering"]["temperature"]),
    )
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
        "documents": docs_to_context(state.get("docs_raw", {}), include_query=False),
    }
    formatted_message = chat_prompt.format(**var)
    answer = llm.invoke(formatted_message)
    return GraphState(answer=answer.content)


def scoring(state: GraphState) -> GraphState:
    llm = load_openai(
        model=LLM_CONFIG["scoring"]["model"],
        temperature=float(LLM_CONFIG["scoring"]["temperature"]),
    )

    sys_message = PROMPTS["scoring"]["sys_message"]
    parser = PydanticOutputParser(pydantic_object=ScoreSummary)
    sys_message = PromptTemplate.from_template(sys_message)
    prompt = sys_message.partial(format=parser.get_format_instructions())
    formatted_input = prompt.format(
        documents=docs_to_context(state.get("docs_raw", {}), include_query=True),
        user_input=state["question"],
        answer=state["answer"],
        query=state.get("query", ""),
    )
    output = llm.invoke(formatted_input)
    parsed_output = parser.parse(output.content)
    loop_count = state.get("scoring_loop", 0)
    next_loop = loop_count + 1 if parsed_output.score != "Pass" else 0
    return GraphState(
        supporting_feedback=parsed_output.feedback,
        ispass=parsed_output.score,
        scoring_loop=next_loop,
    )


def requery(state: GraphState) -> GraphState:
    llm = load_openai(
        model=LLM_CONFIG["requery"]["model"],
        temperature=float(LLM_CONFIG["requery"]["temperature"]),
    )
    parser = PydanticOutputParser(pydantic_object=QuerySummary)
    prompt = PromptTemplate.from_template(PROMPTS["requery"]["sys_prompt"]).partial(
        format=parser.get_format_instructions()
    )
    requery_inputs = build_requery_inputs(
        queries=state.get("query", {}),
        query_state=state.get("query_state", {}),
        relevant_feedback=state.get("relevant_feedback", {}),
        supporting_feedback=state.get("supporting_feedback", ""),
    )
    formatted_message = prompt.format(
        law_rel_queries=requery_inputs["law"]["rel_queries"],
        law_irrel_queries=requery_inputs["law"]["irrel_queries"],
        law_feedback=requery_inputs["law"]["feedback"],
        franchise_rel_queries=requery_inputs["franchise"]["rel_queries"],
        franchise_irrel_queries=requery_inputs["franchise"]["irrel_queries"],
        franchise_feedback=requery_inputs["franchise"]["feedback"],
        user_input=state["question"],
    )
    output = llm.invoke(formatted_message)
    parsed = parser.parse(output.content)
    prev_queries = state.get("query", {}) or {}
    query_state = state.get("query_state", {}) or {}

    def _merge_with_relevant_existing(db_key: str, new_queries: list) -> list:
        base = prev_queries.get(db_key, []) if isinstance(prev_queries, dict) else []
        if not isinstance(base, list):
            base = []

        def get_q_str(item):
            return item.get("query", "") if isinstance(item, dict) else item

        rel_existing = [
            item
            for item in base
            if isinstance(get_q_str(item), str)
            and get_q_str(item).strip()
            and bool(query_state.get(get_q_str(item), False))
        ]

        merged = rel_existing + new_queries

        seen = set()
        deduped = []
        for item in merged:
            q_str = get_q_str(item)
            if not isinstance(q_str, str) or not q_str.strip():
                continue
            if q_str not in seen:
                seen.add(q_str)
                if hasattr(item, "model_dump"):
                    deduped.append(item.model_dump())
                else:
                    deduped.append(item)
        return deduped

    prev_target_brands = state.get("target_brands", []) or []
    if isinstance(prev_target_brands, list) and isinstance(parsed.target_brands, list):
        merged_brands = list(dict.fromkeys(prev_target_brands + parsed.target_brands))
    else:
        merged_brands = prev_target_brands

    return GraphState(
        query={
            "law": _merge_with_relevant_existing("law", parsed.law_query),
            "franchise": _merge_with_relevant_existing(
                "franchise", parsed.franchise_query
            ),
        },
        target_brands=merged_brands,
    )


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

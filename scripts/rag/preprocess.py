from langchain_core.documents import Document
from pathlib import Path
import json


# def docs_to_context(docs: list[Document | dict]) -> str:
#     parts = []
#     for d in docs:
#         if isinstance(d, dict):
#             metadata = d.get("metadata", {})
#             page_content = d.get("page_content", "")
#         else:
#             metadata = d.metadata
#             page_content = d.page_content

#         brand = metadata.get("brand", "")
#         page = metadata.get("page", "")
#         source = metadata.get("source", "")
#         if source:
#             source = Path(source).name
#         meta = f"brand={brand}, source={source}, page={page}"
#         parts.append(
#             "<document>\n"
#             f"<meta>{meta}</meta>\n"
#             f"<content>{page_content}</content>\n"
#             "</document>"
#         )
#     return "\n".join(parts)


def docs_to_context(
    docs: dict[str, list[Document]],
    include_query: bool = True,
) -> str:
    parts = []

    for q_idx, (query, query_docs) in enumerate(docs.items(), start=1):
        if include_query:
            parts.append(f"<query>{query}</query>")
        for d_idx, d in enumerate(query_docs, start=1):
            brand = d.metadata.get("brand", "")
            page = d.metadata.get("page", "")
            source = d.metadata.get("source", "")
            score = d.metadata.get("score", "")
            rerank_score = d.metadata.get("rerank_score", "")
            if source:
                source = Path(source).name
            meta = f"brand={brand}, page={page}, source={source}"
            if score != "":
                meta += f", score={score}"
            if rerank_score != "":
                meta += f", rerank_score={rerank_score}"
            doc_open = (
                f'<document id="d{q_idx}-{d_idx}">\n'
                if include_query
                else "<document>\n"
            )
            parts.append(
                doc_open + f"<meta>{meta}</meta>\n"
                f"<content>{d.page_content}</content>\n"
                "</document>"
            )
    return "\n".join(parts)


def docs_to_query_context(docs: dict[str, list[Document]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for query, query_docs in docs.items():
        result[query] = docs_to_context({query: query_docs}, include_query=True)
    return result


def history_to_context(history: list[dict[str]]) -> str:
    part = ""
    for i in range(len(history)):
        h = history[i]
        a = h["answer"]
        q = h["question"]
        part = part + f"<history>user:{q}\nAi:{a}</history>\n"

    return part


def feedback_to_text(feedback: dict[str, str]) -> str:
    lines = []
    for query, reason in feedback.items():
        lines.append(f"[query] {query}\n[feedback] {reason}")
    return "\n\n".join(lines)


def build_requery_inputs(
    queries: dict[str, list[str]] | dict,
    query_state: dict[str, bool] | dict,
    relevant_feedback: dict[str, str] | dict,
    supporting_feedback: str,
) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {
        "law": {"rel_queries": "[]", "irrel_queries": "[]", "feedback": ""},
        "franchise": {"rel_queries": "[]", "irrel_queries": "[]", "feedback": ""},
    }

    for key in ("law", "franchise"):
        q_list = queries.get(key, []) if isinstance(queries, dict) else []
        if not isinstance(q_list, list):
            q_list = []
        normalized = []
        for q in q_list:
            q_str = q.get("query", "") if isinstance(q, dict) else q
            if isinstance(q_str, str) and q_str.strip():
                normalized.append(q)

        if not normalized:
            continue

        rel_queries = []
        irrel_queries = []
        feedback_lines: list[str] = []
        for q in normalized:
            q_str = q.get("query", "") if isinstance(q, dict) else q
            is_rel = (
                bool(query_state.get(q_str, False))
                if isinstance(query_state, dict)
                else False
            )
            if is_rel:
                rel_queries.append(q)
                continue
            irrel_queries.append(q)
            if isinstance(relevant_feedback, dict):
                raw_fb = relevant_feedback.get(q_str, "")
                if isinstance(raw_fb, str) and raw_fb.strip():
                    feedback_lines.append(
                        f"[query] {q_str}\n[feedback] {raw_fb.strip()}"
                    )

        payload[key]["rel_queries"] = json.dumps(rel_queries, ensure_ascii=False)
        payload[key]["irrel_queries"] = json.dumps(irrel_queries, ensure_ascii=False)
        if feedback_lines:
            payload[key]["feedback"] = "\n\n".join(feedback_lines)
        else:
            payload[key]["feedback"] = (supporting_feedback or "").strip()

    return payload


def dedupe_docs(docs: list[Document]) -> list[Document]:
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

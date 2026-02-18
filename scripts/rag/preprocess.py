from langchain_core.documents import Document
from pathlib import Path


def docs_to_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        brand = d.metadata.get("brand", "")
        page = d.metadata.get("page", "")
        source = d.metadata.get("source", "")
        if source:
            source = Path(source).name
        meta = f"brand={brand}, source={source}, page={page}"
        parts.append(
            "<document>\n"
            f"<meta>{meta}</meta>\n"
            f"<content>{d.page_content}</content>\n"
            "</document>"
        )
    return "\n".join(parts)


def history_to_context(history: list[dict[str]]) -> str:
    part = ""
    for i in range(len(history)):
        h = history[i]
        a = h["answer"]
        q = h["question"]
        part = part + f"<history>user:{q}\nAi:{a}</history>\n"

    return part

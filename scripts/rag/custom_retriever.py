from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import BaseStore


class CustomRetriever:
    """Unified retriever for basic/parent modes with the same invoke interface."""

    def __init__(
        self,
        *,
        vectorstore: Chroma,
        retriever_mode: str = "basic",
        docstore: BaseStore[str, Document] | None = None,
        id_key: str = "doc_id",
    ) -> None:
        if retriever_mode not in {"basic", "parent"}:
            raise ValueError("retriever_mode must be one of: basic, parent")
        if retriever_mode == "parent" and docstore is None:
            raise ValueError("docstore is required when retriever_mode='parent'")

        self.vectorstore = vectorstore
        self.retriever_mode = retriever_mode
        self.docstore = docstore
        self.id_key = id_key

    def invoke(
        self,
        query: str,
        *,
        k: int = 3,
        score_threshold: float | None = None,
        **kwargs,
    ) -> list[tuple[Document, float]]:
        child_results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            **kwargs,
        )

        filtered_children = []
        for child_doc, score in child_results:
            if score_threshold is not None and score < score_threshold:
                continue
            filtered_children.append((child_doc, score))

        if self.retriever_mode == "basic":
            return filtered_children

        parent_ids: list[str] = []
        parent_scores: dict[str, float] = {}
        for child_doc, score in filtered_children:
            parent_id = child_doc.metadata.get(self.id_key)
            if not parent_id or parent_id in parent_scores:
                continue
            parent_ids.append(parent_id)
            parent_scores[parent_id] = score

        if not parent_ids:
            return []

        parent_docs = self.docstore.mget(parent_ids) 
        results: list[tuple[Document, float]] = []
        for parent_id, parent_doc in zip(parent_ids, parent_docs):
            if parent_doc is None:
                continue
            results.append((parent_doc, parent_scores[parent_id]))
        return results

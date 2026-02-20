from dotenv import load_dotenv
import os

import json
import shutil
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import TYPE_CHECKING
from custom_retriever import CustomRetriever

if TYPE_CHECKING:
    from langchain.retrievers import ParentDocumentRetriever

EMBEDDING_MODEL = "text-embedding-3-large"
STORAGE_PATH = "./Chroma_db"
PDF_PATH = "./pdf_folder"
LAW_PDF_PATH = f"{PDF_PATH}/law"
FRANCHISE_PDF_PATH = f"{PDF_PATH}/franchise"
RETRIEVER_MODE = "basic"
FORCE_REBUILD = False
RETRIEVERS = None


def load_api_key(name: str):
    load_dotenv()
    return os.getenv(name)


def _load_docs(pdf_root: str, doc_type: str) -> list[Document]:
    all_docs = []
    for file in Path(pdf_root).rglob("*.pdf"):
        brand = file.stem
        loader = PyPDFLoader(file, mode="page")
        doc = loader.load()
        doc = [
            Document(
                page_content=d.page_content,
                metadata={**d.metadata, "brand": brand, "type": doc_type},
            )
            for d in doc
        ]
        all_docs.extend(doc)
    return all_docs


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _collection_configs() -> dict[str, dict[str, str | Path]]:
    return {
        "law": {
            "pdf_path": LAW_PDF_PATH,
            "collection": "law_db",
            "persist_dir": f"{STORAGE_PATH}/law",
            "docstore_dir": Path(STORAGE_PATH) / "docstore" / "law",
            "doc_type": "law",
        },
        "franchise": {
            "pdf_path": FRANCHISE_PDF_PATH,
            "collection": "franchise_db",
            "persist_dir": f"{STORAGE_PATH}/franchise",
            "docstore_dir": Path(STORAGE_PATH) / "docstore" / "franchise",
            "doc_type": "franchise",
        },
    }


def _make_embedding():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=load_api_key("OPENAI_API_KEY"),
    )


def _make_parent_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )


def _make_child_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=128,
        length_function=len,
        is_separator_regex=False,
    )


def build_vector_store(store: bool) -> dict[str, Chroma]:
    """Build or load unified child vector stores shared by basic/parent modes."""
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import LocalFileStore

    try:
        from langchain.storage import create_kv_docstore
    except ImportError:
        from langchain.storage._lc_store import create_kv_docstore
    embedding = _make_embedding()
    collections = _collection_configs()
    stores: dict[str, Chroma] = {}

    if store:
        print("store new data to Chroma DB")
        parent_bootstrap_batch_size = 50
        for key, cfg in collections.items():
            persist_dir = Path(cfg["persist_dir"])
            docstore_dir = Path(cfg["docstore_dir"])
            if persist_dir.exists():
                shutil.rmtree(persist_dir)
            if docstore_dir.exists():
                shutil.rmtree(docstore_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            docstore_dir.mkdir(parents=True, exist_ok=True)

            vectorstore = Chroma(
                collection_name=cfg["collection"],
                embedding_function=embedding,
                persist_directory=str(persist_dir),
            )
            byte_store = LocalFileStore(str(docstore_dir))
            store = create_kv_docstore(byte_store)
            parent_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=_make_child_splitter(),
                parent_splitter=_make_parent_splitter(),
            )
            all_docs = _load_docs(Path(cfg["pdf_path"]), str(cfg["doc_type"]))
            if all_docs:
                for doc_batch in _batched(all_docs, parent_bootstrap_batch_size):
                    parent_retriever.add_documents(doc_batch)
            else:
                print(f"[WARN] no documents found for '{key}' at {cfg['pdf_path']}")
            stores[key] = vectorstore
        return stores

    for key, cfg in collections.items():
        stores[key] = Chroma(
            collection_name=cfg["collection"],
            embedding_function=embedding,
            persist_directory=str(cfg["persist_dir"]),
        )
    return stores


def build_retrievers(force_rebuild: bool = False) -> dict[str, CustomRetriever]:
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import LocalFileStore

    try:
        from langchain.storage import create_kv_docstore
    except ImportError:
        from langchain.storage._lc_store import create_kv_docstore

    collections = _collection_configs()
    embedding = _make_embedding()
    retrievers: dict[str, CustomRetriever] = {}
    parent_bootstrap_batch_size = 50

    for key, cfg in collections.items():
        persist_dir = Path(cfg["persist_dir"])
        docstore_dir = Path(cfg["docstore_dir"])
        if force_rebuild and persist_dir.exists():
            shutil.rmtree(persist_dir)
        if force_rebuild and docstore_dir.exists():
            shutil.rmtree(docstore_dir)

        persist_dir.mkdir(parents=True, exist_ok=True)
        docstore_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma(
            collection_name=cfg["collection"],
            embedding_function=embedding,
            persist_directory=str(persist_dir),
        )

        byte_store = LocalFileStore(str(docstore_dir))
        store = create_kv_docstore(byte_store)

        if RETRIEVER_MODE == "parent":
            parent_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=_make_child_splitter(),
                parent_splitter=_make_parent_splitter(),
            )
            try:
                existing_children = vectorstore._collection.count()
            except Exception:
                existing_children = 0
            has_docstore_files = any(p.is_file() for p in docstore_dir.rglob("*"))
            should_bootstrap = existing_children == 0 or not has_docstore_files
            if should_bootstrap:
                all_docs = _load_docs(Path(cfg["pdf_path"]), str(cfg["doc_type"]))
                if all_docs:
                    for doc_batch in _batched(all_docs, parent_bootstrap_batch_size):
                        parent_retriever.add_documents(doc_batch)
                else:
                    print(
                        f"[WARN] no parent documents found for '{key}' at {cfg['pdf_path']}"
                    )

        retrievers[key] = CustomRetriever(
            vectorstore=vectorstore,
            retriever_mode=RETRIEVER_MODE,
            docstore=store,
            id_key="doc_id",
        )

    return retrievers


def safe_json_dict(raw: str, warn: str | None = None) -> dict:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    if warn:
        print(warn)
    return {}


def set_retriever_mode(mode: str, force_rebuild: bool = False) -> None:
    global RETRIEVER_MODE, FORCE_REBUILD, RETRIEVERS
    if mode not in {"basic", "parent"}:
        raise ValueError("retriever mode must be one of: basic, parent")
    RETRIEVER_MODE = mode
    FORCE_REBUILD = force_rebuild
    RETRIEVERS = None


def get_retriever_mode() -> str:
    return RETRIEVER_MODE


def get_retrievers():
    global RETRIEVERS
    if RETRIEVERS is None:
        RETRIEVERS = build_retrievers(force_rebuild=FORCE_REBUILD)
    return RETRIEVERS


def load_openai(model="gpt-4o-mini", temperature=0.5):
    OPENAI_API_KEY = load_api_key("OPENAI_API_KEY")
    return ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=temperature)


def load_prompts(path: str) -> dict:
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "PyYAML is required to load YAML prompts. Install it with: pip install pyyaml"
            ) from e

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    def normalize(value):
        if isinstance(value, list):
            return "\n".join(value)
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        return value

    return normalize(raw)


def render_text_with_graphs(text: str) -> list[dict[str, str]]:
    """
    Parses text for ```json:graph blocks, renders them with plotext,
    and returns a list of segments:
    [{"type": "text", "content": "..."}, {"type": "graph", "content": "..."}]
    """
    # try:
    #     import plotext as plt
    # except ImportError:
    #     return [{"type": "text", "content": text}]

    import re

    # Regex to find json:graph blocks with capturing group for content
    # Regex to find graph blocks (supports json:graph, json, or just ``` with type: "line"|"bar" inside)
    # This is a bit complex to do with a single split, so we'll look for code blocks and inspect them.
    pattern = r"(```(?:json|json:graph)?\n.*?\n```)"
    parts = re.split(pattern, text, flags=re.DOTALL)

    segments = []

    for part in parts:
        if not part:
            continue

        if part.startswith("```"):
            # Check if it's a graph block
            is_graph = False
            json_str = ""

            try:
                # Remove fences
                content = re.sub(r"^```(?:json|json:graph)?\n", "", part)
                content = re.sub(r"\n```$", "", content)
                json_str = content.strip()

                # Check if it looks like our graph schema
                if '"type":' in json_str and '"data":' in json_str:
                    is_graph = True
            except Exception:
                pass

            if is_graph:
                # Use the raw JSON string for the frontend to parse and render
                segments.append({"type": "graph_json", "content": json_str})
            else:
                segments.append({"type": "text", "content": part})
        else:
            # Regular text
            if part.strip():
                segments.append({"type": "text", "content": part})

    return segments

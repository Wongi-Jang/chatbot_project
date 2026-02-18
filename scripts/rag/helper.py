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

if TYPE_CHECKING:
    from langchain.retrievers import ParentDocumentRetriever

EMBEDDING_MODEL = "text-embedding-3-large"
STORAGE_PATH = "./Chroma_db"
PDF_PATH = "./pdf_folder"
LAW_PDF_PATH = f"{PDF_PATH}/law"
FRANCHISE_PDF_PATH = f"{PDF_PATH}/franchise"


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


def _collection_configs() -> dict[str, dict[str, str]]:
    return {
        "law": {
            "pdf_path": LAW_PDF_PATH,
            "collection": "law_db",
            "persist_dir": f"{STORAGE_PATH}/law",
            "doc_type": "law",
        },
        "franchise": {
            "pdf_path": FRANCHISE_PDF_PATH,
            "collection": "franchise_db",
            "persist_dir": f"{STORAGE_PATH}/franchise",
            "doc_type": "franchise",
        },
    }


def build_vector_store(store: bool) -> dict[str, Chroma]:
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=load_api_key("OPENAI_API_KEY"),
        chunk_size=100,
    )
    collections = _collection_configs()

    stores: dict[str, Chroma] = {}

    if store:
        print("store new data to Chroma DB")
        for key, cfg in collections.items():
            all_docs = _load_docs(cfg["pdf_path"], cfg["doc_type"])
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=128,
                length_function=len,
                is_separator_regex=False,
            )
            tokens = text_splitter.split_documents(all_docs)
            stores[key] = Chroma.from_documents(
                documents=tokens,
                collection_name=cfg["collection"],
                embedding=embedding,
                persist_directory=cfg["persist_dir"],
            )
        return stores

    # print("load exsiting Chroma DB")
    for key, cfg in collections.items():
        stores[key] = Chroma(
            collection_name=cfg["collection"],
            embedding_function=embedding,
            persist_directory=cfg["persist_dir"],
        )
    return stores


def build_parent_retrievers(
    force_rebuild: bool = False,
) -> dict[str, "ParentDocumentRetriever"]:
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import LocalFileStore

    try:
        from langchain.storage import create_kv_docstore
    except ImportError:
        from langchain.storage._lc_store import create_kv_docstore

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=load_api_key("OPENAI_API_KEY"),
        chunk_size=100,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=128,
        length_function=len,
        is_separator_regex=False,
    )
    parent_root = Path(STORAGE_PATH) / "parent"
    parent_root.mkdir(parents=True, exist_ok=True)
    retrievers: dict[str, ParentDocumentRetriever] = {}
    for key, cfg in _collection_configs().items():
        chroma_dir = parent_root / key / "chroma"
        docstore_dir = parent_root / key / "docstore"
        if force_rebuild:
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
            if docstore_dir.exists():
                shutil.rmtree(docstore_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        docstore_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma(
            collection_name=f"{cfg['collection']}_parent",
            embedding_function=embedding,
            persist_directory=str(chroma_dir),
        )
        byte_store = LocalFileStore(str(docstore_dir))
        store = create_kv_docstore(byte_store)
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        try:
            existing_children = vectorstore._collection.count()
        except Exception:
            existing_children = 0
        has_docstore_files = any(p.is_file() for p in docstore_dir.rglob("*"))
        should_bootstrap = existing_children == 0 or not has_docstore_files

        if should_bootstrap:
            # print(f"build parent retriever index: {key}")
            all_docs = _load_docs(cfg["pdf_path"], cfg["doc_type"])
            retriever.add_documents(all_docs)
        else:
            pass
            # print(f"load parent retriever index: {key}")

        retrievers[key] = retriever
    return retrievers


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


def render_graph_from_text(text: str) -> str:
    """
    Parses text for ```json:graph blocks, renders them with plotext,
    and replaces the block with the rendered graph.
    """
    try:
        import plotext as plt
    except ImportError:
        return text

    import re

    # Regex to find json:graph blocks
    pattern = r"```json:graph\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    for json_str in matches:
        try:
            data = json.loads(json_str)
            plt.clf()
            plt.theme("dark")
            plt.frame(True)
            plt.grid(True)

            # Common setup
            plt.title(data.get("title", "Graph"))

            graph_type = data.get("type", "line")
            series_list = data.get("data", [])

            for series in series_list:
                label = series.get("label", "")
                x = series.get("x", [])
                y = series.get("y", [])

                if graph_type == "bar":
                    plt.bar(x, y, label=label)
                else:
                    plt.plot(x, y, label=label)

            graph_output = plt.build()

            # Replace the code block with the rendered graph
            block = f"```json:graph\n{json_str}\n```"
            text = text.replace(block, f"\n{graph_output}\n")

        except Exception as e:
            # pass
            print(f"Failed to render graph: {e}")

    return text


def render_text_with_graphs(text: str) -> list[dict[str, str]]:
    """
    Parses text for ```json:graph blocks, renders them with plotext,
    and returns a list of segments:
    [{"type": "text", "content": "..."}, {"type": "graph", "content": "..."}]
    """
    try:
        import plotext as plt
    except ImportError:
        return [{"type": "text", "content": text}]

    import re

    # Regex to find json:graph blocks with capturing group for content
    pattern = r"(```json:graph\n.*?\n```)"
    # Split by the pattern. capture group implies we get [text, block, text, block, ...]
    parts = re.split(pattern, text, flags=re.DOTALL)

    segments = []

    for part in parts:
        if not part:
            continue

        if part.startswith("```json:graph"):
            # Process graph block
            try:
                # Extract JSON content
                json_str = part.replace("```json:graph\n", "").replace("\n```", "")
                data = json.loads(json_str)

                plt.clf()
                plt.theme("dark")
                plt.frame(True)
                plt.grid(True)
                plt.title(data.get("title", "Graph"))

                graph_type = data.get("type", "line")
                series_list = data.get("data", [])

                for series in series_list:
                    label = series.get("label", "")
                    x = series.get("x", [])
                    y = series.get("y", [])

                    if graph_type == "bar":
                        plt.bar(x, y, label=label)
                    else:
                        plt.plot(x, y, label=label)

                graph_output = plt.build()
                segments.append({"type": "graph", "content": graph_output})

            except Exception as e:
                print(f"[Graph Render Error]: {e}")
                # Fallback: Treat as text code block if parsing fails
                segments.append({"type": "text", "content": part})
        else:
            # Regular text
            if part.strip():
                segments.append({"type": "text", "content": part})

    return segments

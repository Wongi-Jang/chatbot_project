from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
import argparse
import os
import time
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from helper import render_text_with_graphs, docs_to_context

## argparse
parser = argparse.ArgumentParser(description="rag test")
parser.add_argument("--store", action="store_true", help="initialize Chroma DB")
parser.add_argument("--chat", action="store_true", help="chat with existing Chroma DB")
parser.add_argument("--system", type=str, default="", help="system prompt/persona")
parser.add_argument(
    "--system-file", type=str, default="", help="path to system prompt file"
)
args = parser.parse_args()

##
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "test_db"
STORAGE_PATH = "./Chroma_db"
PDF_PATH = "./pdf_folder"
SCORE_THRESHOLD = 0.5
##


embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=OPENAI_API_KEY, chunk_size=100
)


def build_vector_store(store: bool) -> Chroma:
    if store:
        print("store new data to Chroma DB")
        all_docs = []
        for file in Path(PDF_PATH).rglob("*.pdf"):
            brand = file.stem
            loader = PyPDFLoader(file, mode="page")
            doc = loader.load()
            doc = [
                Document(
                    page_content=d.page_content,
                    metadata={**d.metadata, "brand": brand, "type": "brand"},
                )
                for d in doc
            ]
            all_docs.extend(doc)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=128,
            length_function=len,
            is_separator_regex=False,
        )

        tokens = text_splitter.split_documents(all_docs)
        if not tokens:
            print("No documents found. Exiting.")
            return None

        return Chroma.from_documents(
            documents=tokens,
            collection_name=COLLECTION_NAME,
            embedding=embedding,
            persist_directory=STORAGE_PATH,
        )

    print("load exsiting Chroma DB")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=STORAGE_PATH,
    )


def chatting(vector_store: Chroma, system_prompt: str) -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.5)
    k = 3
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt or "You are a helpful assistant."),
            (
                "human",
                "대화 기록:\n{chat_history}\n\n다음 컨텍스트를 참고해서 질문에 답변하세요.\n\n{context}\n\n질문: {question}",
            ),
        ]
    )

    chat_history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break
        history_text = (
            "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history]).strip()
            or "(없음)"
        )
        t0 = time.perf_counter()
        results = vector_store.similarity_search_with_score(
            query=user_input,
            k=k,
        )
        docs = [doc for doc, score in results if score >= SCORE_THRESHOLD]
        t1 = time.perf_counter()
        context_text = docs_to_context(docs)
        messages = qa_prompt.format_messages(
            chat_history=history_text,
            context=context_text,
            question=user_input,
        )
        t2 = time.perf_counter()
        response = llm.invoke(messages)
        t3 = time.perf_counter()
        answer = getattr(response, "content", "") or ""

        segments = render_text_with_graphs(answer)
        if Console and Markdown:
            Console().print(Markdown("**AI:**"))
            for seg in segments:
                if seg["type"] == "text":
                    Console().print(Markdown(seg["content"]))
                elif seg["type"] == "graph_json":
                    # Render JSON graph for CLI using plotext
                    try:
                        import json
                        import plotext as plt

                        data = json.loads(seg["content"])
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

                        plt.show()
                    except Exception as e:
                        print(f"Failed to render graph: {e}")
                        print(seg["content"])
                else:
                    print(seg["content"])
        else:
            print(f"\nAI: {answer}")

        print(f"\n[timing] retrieval: {(t1 - t0):.3f}s, generation: {(t3 - t2):.3f}s")
        chat_history.append((user_input, answer))


if __name__ == "__main__":
    sys_prompt = """
    [System Role] You are the WinWIn AI Consultant, a professional advisor dedicated to supporting franchise owners. Your primary mission is to provide objective information based on Franchise Information Disclosure Documents, the Fair Transactions in Franchise Business Act, and Standard Franchise Agreement laws. Beyond just providing data, you must offer strategic advice to help prospective and current owners make informed decisions. 
    [Response Strategy & Logic] 1. Data Integration: Use specific figures from disclosure documents (e.g., annual sales, startup costs, store counts) to support your claims. 2. Comparative Insight: Do not just list numbers; compare them with industry averages or competing brands to highlight strengths and weaknesses. 3. Critical Advice: If a brand shows high revenue but also has disproportionately high initial investment costs or a decreasing store count, provide a balanced view.  Warn users that high revenue does not always guarantee high profitability. 4. Transparency: Always cite the source of your data (e.g., "According to the 2024 Information Disclosure Document...") to maintain trust. 5. Format: When presenting comparative data (e.g., sales vs costs, brand A vs brand B), ALWAYS use a Markdown table for clarity. 6. Citations: ALWAYS cite the source document and page number when using information. Format: (Filename, Page X). Example: (lotteria.pdf, Page 13).
    [Example Interaction]
    User: "What is the annual revenue of Kyochon Chicken, and what is the outlook?" * AI: "Kyochon Chicken's average annual sales were approximately ₩750M in 2022 and increased to ₩727M in 2024, showing a steady performance (kyochon_2024.pdf, Page 5). However, while revenue remains stable, the initial startup cost is approximately ₩196M (for a 132㎡ store), which is significantly higher than other chicken franchises. Therefore, despite the strong brand power, the high entry cost and maintenance expenses mean the outlook for net profitability requires cautious evaluation."
    """
    vector_store = build_vector_store(args.store)

    if args.chat:
        system_prompt = args.system
        if args.system_file:
            with open(args.system_file, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        chatting(vector_store, sys_prompt)

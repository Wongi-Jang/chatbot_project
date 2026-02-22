import os
import sys

from helper import load_api_key, get_retrievers

q = "BBQ 브랜드 가맹점 운영 시 사용 가능한 지식재산권 종류 및 등록 번호"
print(f"Testing query: {q}")

retrievers = get_retrievers()
franchise_retriever = retrievers.get("franchise")

print("--- Raw Similarity Search With Score ---")
raw_results = franchise_retriever.vectorstore.similarity_search_with_score(q, k=5)
for doc, score in raw_results:
    brand = doc.metadata.get("brand")
    print(
        f"Score/Distance: {score:.4f} | Brand: {brand} | Content: {doc.page_content[:50]}..."
    )

print("\n--- CustomRetriever Invoke with score_threshold=0.5 ---")
results = franchise_retriever.invoke(q, k=5, score_threshold=0.5)
print(f"Returned {len(results)} docs")
for doc, score in results:
    brand = doc.metadata.get("brand")
    print(f"Score: {score:.4f} | Brand: {brand} | Content: {doc.page_content[:50]}...")

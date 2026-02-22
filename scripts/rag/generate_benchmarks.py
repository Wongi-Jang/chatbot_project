import os
import glob
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List


# Define the structure for the output
class QuestionList(BaseModel):
    questions: List[str] = Field(
        description="A list of generated benchmark questions in Korean."
    )


def generate_questions(pdf_path, num_questions, topic):
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Extract text from middle pages to avoid just title pages
    full_text = ""
    start_page = min(2, len(docs) - 1) if len(docs) > 3 else 0
    end_page = min(start_page + 10, len(docs))  # sample up to 10 pages for context

    for i in range(start_page, end_page):
        full_text += docs[i].page_content + "\n"

    # Limit context length to avoid token limits (approx 15000 chars)
    full_text = full_text[:15000]

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    parser = JsonOutputParser(pydantic_object=QuestionList)

    prompt = PromptTemplate(
        template="""당신은 프랜차이즈 창업을 고민하는 자영업자를 돕는 AI 챗봇의 성능을 평가하기 위한 고품질 벤치마크 질문을 만드는 전문가입니다.
다음은 '{topic}'에 관련된 문서의 일부 내용입니다.
이 문서를 바탕으로 실제 자영업자가 창업이나 매장 운영 시 가장 궁금해하고 실질적으로 필요로 할 만한 구체적인 질문 {num_questions}개를 한국어로 작성해주세요.
질문은 문서에 있는 정보로 답변할 수 있어야 합니다.
특히, 프랜차이즈 문서의 경우 질문 안에 **반드시 해당 문서의 브랜드 이름({brand_name})**을 명시적으로 포함해야 합니다.

{format_instructions}

문서 내용:
{context}
""",
        input_variables=["topic", "num_questions", "context", "brand_name"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        response = chain.invoke(
            {
                "topic": topic,
                "num_questions": num_questions,
                "context": full_text,
                "brand_name": os.path.splitext(os.path.basename(pdf_path))[0],
            }
        )
        return response["questions"]
    except Exception as e:
        print(f"Error generating questions for {pdf_path}: {e}")
        return []


def generate_comparison_questions(brand_names, num_questions):
    print("Generating comparative questions...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    parser = JsonOutputParser(pydantic_object=QuestionList)
    prompt = PromptTemplate(
        template="""당신은 프랜차이즈 창업을 고민하는 자영업자를 돕는 AI 챗봇의 성능을 평가하기 위한 고품질 벤치마크 질문을 만드는 전문가입니다.
다음은 현재 시스템에 정보가공개서 문서가 등록된 프랜차이즈 브랜드들의 목록입니다:
{brand_list}

이 브랜드들을 바탕으로 실제 자영업자가 창업 시 여러 브랜드를 비교 분석하기 위해 할 법한 구체적이고 실질적인 질문 {num_questions}개를 한국어로 작성해주세요.
조건:
1. 각 질문에는 위 목록에 있는 브랜드 중 서로 다른 2~3개의 브랜드 이름이 명시적으로 포함되어야 합니다. (예: "버거킹, 맥도날드, 롯데리아의 최초가맹비를 비교해줘")
2. 질문은 창업 비용(최초가맹비, 인테리어 비용 등), 매출액, 가맹점 수, 평당 매출, 운영수익 등 구체적인 비교 항목을 하나 이상 포함해야 합니다.

{format_instructions}
""",
        input_variables=["brand_list", "num_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    try:
        response = chain.invoke(
            {"brand_list": ", ".join(brand_names), "num_questions": num_questions}
        )
        return response["questions"]
    except Exception as e:
        print(f"Error generating comparative questions: {e}")
        return []


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    law_dir = os.path.join(base_dir, "pdf_folder", "law")
    franchise_dir = os.path.join(base_dir, "pdf_folder", "franchise")

    law_pdfs = glob.glob(os.path.join(law_dir, "*.pdf"))
    franchise_pdfs = glob.glob(os.path.join(franchise_dir, "*.pdf"))

    all_questions = []

    print(f"Found {len(law_pdfs)} law PDFs and {len(franchise_pdfs)} franchise PDFs.")

    # 5 questions for law (2 PDFs)
    # We will generate 3 from the first, 2 from the second.
    q = generate_questions(law_pdfs[0], 3, "프랜차이즈 관련 법률 및 표준가맹계약서")
    for qs in q:
        all_questions.append(
            {"category": "law", "source": os.path.basename(law_pdfs[0]), "question": qs}
        )

    if len(law_pdfs) > 1:
        q = generate_questions(law_pdfs[1], 2, "프랜차이즈 관련 법률 및 표준가맹계약서")
        for qs in q:
            all_questions.append(
                {
                    "category": "law",
                    "source": os.path.basename(law_pdfs[1]),
                    "question": qs,
                }
            )

    # Generate exactly 35 questions for franchise
    franchise_questions_list = []
    pdf_idx = 0
    while len(franchise_questions_list) < 35:
        pdf_path = franchise_pdfs[pdf_idx % len(franchise_pdfs)]
        # Request a chunk of questions
        q = generate_questions(
            pdf_path, 2, "특정 프랜차이즈 브랜드 정보가공개서 및 관련 정보"
        )
        for qs in q:
            if len(franchise_questions_list) < 35:
                franchise_questions_list.append(
                    {
                        "category": "franchise",
                        "source": os.path.basename(pdf_path),
                        "question": qs,
                    }
                )
        pdf_idx += 1

    all_questions.extend(franchise_questions_list)

    # 10 questions for comparison
    franchise_brand_names = [
        os.path.splitext(os.path.basename(p))[0] for p in franchise_pdfs
    ]
    comp_qs = generate_comparison_questions(franchise_brand_names, 10)
    for qs in comp_qs:
        all_questions.append(
            {"category": "comparison", "source": "여러 브랜드 비교", "question": qs}
        )

    # Keep exactly 50 questions
    law_qs = [q for q in all_questions if q["category"] == "law"][:5]
    franchise_qs = [q for q in all_questions if q["category"] == "franchise"][:35]
    comparison_qs = [q for q in all_questions if q["category"] == "comparison"][:10]
    final_questions = law_qs + franchise_qs + comparison_qs

    output_path = os.path.join(base_dir, "benchmark_questions.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 챗봇 성능 평가 벤치마크 질문 50선\n\n")

        f.write("## 1. 프랜차이즈 관련 법률 및 표준가맹계약서 (5문제)\n")
        f.write(
            "이 질문들은 `pdf_folder/law` 디렉토리의 문서들을 바탕으로 생성되었습니다.\n\n"
        )
        for i, q in enumerate(law_qs, 1):
            f.write(f"{i}. **{q['question']}** (출처: *{q['source']}*)\n")

        f.write("\n## 2. 특정 브랜드 정보가공개서 단일 질의 (35문제)\n")
        f.write(
            "이 질문들은 `pdf_folder/franchise` 디렉토리의 단일 브랜드 문서들을 바탕으로 생성되었습니다.\n\n"
        )
        for i, q in enumerate(franchise_qs, 1):
            f.write(f"{i}. **{q['question']}** (출처: *{q['source']}*)\n")

        f.write("\n## 3. 여러 브랜드 비교 질의 (10문제)\n")
        f.write(
            "이 질문들은 여러 프랜차이즈 브랜드를 동시에 비교 분석하기 위해 생성되었습니다.\n\n"
        )
        for i, q in enumerate(comparison_qs, 1):
            f.write(f"{i}. **{q['question']}** (출처: *{q['source']}*)\n")

    print(
        f"\nSuccessfully generated {len(final_questions)} questions and saved to {output_path}"
    )


if __name__ == "__main__":
    main()

import sys
import os
import json
import re
import uuid
from langgraph.checkpoint.memory import MemorySaver

# Add current directory to path so we can import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from graph_base import create_graph


def parse_markdown_questions(md_path):
    questions = []
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # regex matches e.g. "1. **질문내용** (출처: *source*)"
    pattern = re.compile(
        r"^\d+\.\s+\*\*(.+?)\*\*\s+\(출처:\s+\*(.+?)\*\)", re.MULTILINE
    )
    for match in pattern.finditer(content):
        question = match.group(1).strip()
        source = match.group(2).strip()
        questions.append({"question": question, "source": source})
    return questions


def evaluate_questions(questions, output_file):
    memory = MemorySaver()
    # Initialize the graph in "web mode" so it returns immediately after processing
    app = create_graph(
        enable_scoring=True,
        retriever_mode="basic",
        is_web=True,
        checkpointer=memory,
    )

    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, q_dict in enumerate(questions, 1):
            question_text = q_dict["question"]
            source = q_dict["source"]
            print(f"[{idx}/{len(questions)}] Evaluating: {question_text}")

            # Use a unique thread ID for each question to avoid memory cross-contamination
            config = {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": 50,
            }
            initial_input = {"question": question_text, "history": []}

            try:
                # Invoke the LangGraph pipeline
                final_state = app.invoke(initial_input, config=config)

                # Extract the final answer and any relevant docs list if we want
                answer = final_state.get("answer", "No answer generated")

            except Exception as e:
                print(f"Error evaluating question {idx}: {e}")
                answer = f"Error: {e}"

            result = {
                "id": idx,
                "question": question_text,
                "expected_source": source,
                "chatbot_answer": answer,
            }
            # Write to JSONL
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"Done. Answer length: {len(answer)}\n")


if __name__ == "__main__":
    md_path = os.path.join(current_dir, "benchmark_questions.md")
    output_path = os.path.join(current_dir, "evaluation_results.jsonl")

    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found. Please run generate_benchmarks.py first.")
        sys.exit(1)

    qs = parse_markdown_questions(md_path)
    print(f"Parsed {len(qs)} questions from {md_path}.")

    print(f"Starting evaluation. This will take several minutes.")
    evaluate_questions(qs, output_path)
    print(f"Evaluation complete! Results saved to {output_path}")

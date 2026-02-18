from scripts.rag.helper import load_prompts
from pathlib import Path
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

try:
    path = Path("scripts/rag/prompts.yaml")
    PROMPTS = load_prompts(str(path))

    sys_msg_text = PROMPTS["answering"]["sys_message"]
    print(f"Loaded sys_message (snippet): {sys_msg_text[-100:]}")

    sys_message = SystemMessagePromptTemplate.from_template(sys_msg_text)
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
        "user_input": "Test Input",
        "history": "Test History",
        "documents": "Test Docs",
    }

    formatted = chat_prompt.format(**var)
    print("\nSUCCESS: Prompt formatted successfully!")

except Exception as e:
    print(f"\nFAILURE: {e}")
    import traceback

    traceback.print_exc()

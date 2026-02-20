import streamlit as st
import sys
import json
import pandas as pd
import time
from pathlib import Path

# Add the current directory and the rag scripts directory to sys.path
sys.path.append(str(Path(__file__).parent))

from graph_base import create_graph
from helper import (
    render_text_with_graphs,
    build_vector_store,
)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("LangGraph RAG Chatbot")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    retriever_mode = st.selectbox(
        "Retriever Mode",
        options=["basic", "parent"],
        index=0,
        help="Choose between basic retrieval or parent document retrieval.",
    )
    enable_scoring = st.toggle(
        "Enable Scoring / Self-Correction",
        value=False,
        help="If enabled, the system will evaluate its own answers and potentially requery.",
    )

    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.app_state = {"history": []}
        st.rerun()

    st.divider()

    if st.button("Rebuild Vector Store"):
        with st.spinner("Rebuilding Vector Store from PDFs... This may take a while."):
            try:
                build_vector_store(store=True)
                st.success("Vector Store Rebuilt Successfully!")
            except Exception as e:
                st.error(f"Failed to rebuild vector store: {e}")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "app_state" not in st.session_state:
    st.session_state.app_state = {"history": []}


# Always create/update the app based on current config
app = create_graph(
    enable_scoring=enable_scoring,
    retriever_mode=retriever_mode,
    is_web=True,
)

# Simple thread ID for memory
config = {"configurable": {"thread_id": "streamlit_user"}}

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # If content has graph segments, render them
        if msg.get("segments"):
            for seg in msg["segments"]:
                if seg["type"] == "text":
                    st.markdown(seg["content"])
                elif seg["type"] == "graph_json":
                    try:
                        data = json.loads(seg["content"])
                        graph_type = data.get("type", "line")
                        series_list = data.get("data", [])
                        title = data.get("title", "Graph")

                        st.caption(title)

                        df_data = {}
                        if series_list:
                            x_labels = series_list[0].get("x", [])
                            df_data["x"] = x_labels
                            for series in series_list:
                                label = series.get("label", "Data")
                                y_values = series.get("y", [])
                                if len(y_values) == len(x_labels):
                                    df_data[label] = y_values

                            df = pd.DataFrame(df_data).set_index("x")
                            if graph_type == "bar":
                                st.bar_chart(df)
                            else:
                                st.line_chart(df)
                    except Exception:
                        st.code(seg["content"], language="json")
                else:
                    st.code(seg["content"], language="text")
        else:
            st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with LangGraph
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        segments = []

        # RESET State for new question (Critical for stateless-like behavior in loops)
        # We manually initialize the state similar to 'user_input' node
        inputs = {
            "question": prompt,
            "query": {},
            "docs_raw": [],
            "docs": "",
            "answer": "",
            "do_retrieve": False,
            "retrieve_types": [],
            "db_meta": {},
            "isrel": "",
            "query_state": {},
            "issup": "",
            "irrelevant_count": 0,
            "relevant_loop": 0,
            "supporting_loop": 0,
            "scoring_loop": 0,
            "relevant_feedback": {},
            "supporting_feedback": "",
            "ispass": "",
            "no_docs": False,
            # We pass history if needed, LangGraph memory handles persistence usually,
            # but we can pass explicit history if the graph logic relies on state["history"]
            "history": st.session_state.app_state["history"],
        }

        try:
            start_time = time.time()
            last_node_time = start_time

            for event in app.stream(inputs, config=config):
                current_time = time.time()
                duration = current_time - last_node_time
                last_node_time = current_time

                # event is a dict of node_name -> state_update

                # Check for answer update
                if "answer" in event:
                    answer_state = event["answer"]
                    answer = answer_state.get("answer", "")
                    if answer:
                        full_response = answer
                        segments = render_text_with_graphs(answer)

                        # Display
                        with message_placeholder.container():
                            for seg in segments:
                                if seg["type"] == "text":
                                    st.markdown(seg["content"])
                                elif seg["type"] == "graph_json":
                                    try:
                                        data = json.loads(seg["content"])
                                        graph_type = data.get("type", "line")
                                        series_list = data.get("data", [])
                                        title = data.get("title", "Graph")

                                        st.caption(title)

                                        df_data = {}
                                        x_labels = []

                                        if series_list:
                                            x_labels = series_list[0].get("x", [])
                                            df_data["x"] = x_labels

                                            for series in series_list:
                                                label = series.get("label", "Data")
                                                y_values = series.get("y", [])
                                                if len(y_values) == len(x_labels):
                                                    df_data[label] = y_values

                                            df = pd.DataFrame(df_data)
                                            df = df.set_index("x")

                                            if graph_type == "bar":
                                                st.bar_chart(df)
                                            else:
                                                st.line_chart(df)
                                    except Exception as e:
                                        st.error(f"Failed to render graph: {e}")
                                        st.code(seg["content"], language="json")
                                else:
                                    st.code(seg["content"], language="text")

                # Simply show status of what node is running
                # "retrieve", "query_gen", "relevant", etc.
                for node_name in event:
                    if node_name != "answer" and node_name != "summarize":
                        with st.status(
                            f"Processing: {node_name} ({duration:.2f}s)", expanded=False
                        ):
                            st.write(event[node_name])

                # Update history in app state if summarize happened
                if "summarize" in event:
                    new_history = event["summarize"].get("history", [])
                    if new_history:
                        # Append to our local session state history if needed
                        # But LangGraph memory might be enough.
                        # We'll update just in case next turn needs it from input
                        st.session_state.app_state["history"].extend(new_history)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Add assistant response to history
        if full_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "segments": segments}
            )

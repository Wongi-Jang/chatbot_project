import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path

# Add the current directory and the rag scripts directory to sys.path
sys.path.append(str(Path(__file__).parent))
# sys.path.append(str(Path(__file__).parent / "scripts" / "rag"))

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

if "app" not in st.session_state:
    # We re-create the graph only when config changes would be ideal,
    # but for simplicity we can recreate it or store it.
    # Storing it is better.
    pass

# Always create/update the app based on current config
# (In a real app, you might want to detect changes to avoid rebuilding unnecessary)
app = create_graph(
    enable_scoring=enable_scoring, retriever_mode=retriever_mode, is_interactive=False
)

if "app_state" not in st.session_state:
    st.session_state.app_state = {"history": []}

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

        # Prepare input state
        # We need to pass history if we want context.
        # But MemorySaver checkpointing might handle persistence if thread_id is used.
        # graph_base.py uses MemorySaver.

        # We pass the current question.
        inputs = {"question": prompt}

        # Stream the graph
        # We want to capture the "answer" node output.
        # And also intermediate steps if we want to debug, but mainly the answer.

        try:
            for event in app.stream(inputs, config=config):
                # event is a dict of node_name -> state_update

                # If we get an answer event
                if "answer" in event:
                    answer_state = event["answer"]
                    answer = answer_state.get("answer", "")
                    if answer:
                        full_response = answer
                        # Render graphs if any
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

                                        # Prepare data for Streamlit charts (Index=X, Columns=Series)
                                        df_data = {}
                                        x_labels = []

                                        if series_list:
                                            # Assume all series share the same X labels for simplicity
                                            # Only take the FIRST set of X labels to align the DataFrame
                                            x_labels = series_list[0].get("x", [])
                                            df_data["x"] = x_labels

                                            for series in series_list:
                                                label = series.get("label", "Data")
                                                y_values = series.get("y", [])
                                                # Ensure lengths match
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

                # We could also show status updates like "Searching..." if we check other events
                if "retrieve" in event:
                    with st.status("Retrieving information...", expanded=False):
                        st.write("Documents retrieved.")
                        # st.write(event["retrieve"].get("docs", ""))

        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Add assistant response to history
        # We store the *segments* to allow re-rendering
        if full_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "segments": segments}
            )

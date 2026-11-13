import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from LLM import MODELS, azure_openai
from RetAuGen.advisor import WORKFLOW_STEPS, advisor_context, build_advisor, last_reply
from RetAuGen.memory_agent import memory_agent
from Utility.agent_memory_context import AgentMemoryContext
from Utility.chat_ui import STREAM_MODES, render_stream, show_message

st.set_page_config(page_title="GABAI", layout="wide")

st.title("Assistant Conversation")

# Pick the LLM; every model is a LangChain chat model, so the advisor graph runs with any of them
model_name = st.sidebar.selectbox("Model", list(MODELS))

# Switch for the memory agent: when off, no memory is created or changed (stored memory is still given to the main agent)
create_memory = st.sidebar.toggle("Create memory", value=True)

# The advisor agent is a LangGraph graph (RetAuGen/advisor.py), shared with the test-runner page. Its checkpointer keeps
# this conversation (messages, workflow state, approval requests) under one thread id for the session.
if "advisor" not in st.session_state:
    st.session_state.advisor = build_advisor(checkpointer=InMemorySaver())
    st.session_state.advisor_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
advisor = st.session_state.advisor
config = st.session_state.advisor_config


# Agent memory, (last_update, memory) or None. Loaded from the db once per session, and refreshed whenever the
# memory agent saves; the advisor puts it at the tail of the main agent's prompt.
def load_memory():
    memory_db = AgentMemoryContext()
    try:
        return memory_db.get()
    finally:
        memory_db.close()


if "agent_memory" not in st.session_state:
    try:
        st.session_state.agent_memory = load_memory()
    except Exception as e:  # the app must still work without the memory db
        st.session_state.agent_memory = None
        st.sidebar.warning(f"Agent memory unavailable: {e}")


def observe(user_message, response):
    """Independent observer: the memory agent decides if this exchange is worth saving to the agent memory."""
    memory_db = AgentMemoryContext()
    try:
        saved = memory_agent(user_message, response, memory_db, azure_openai(temperature=0))
        if saved:
            st.session_state.agent_memory = memory_db.get()  # the main agent's prompt picks this up on its next call
        return saved
    finally:
        memory_db.close()


state = advisor.get_state(config).values

# The conversation so far (tool results are sent to the model but not shown)
for message in state.get("messages", []):
    show_message(message)

# Workflow state of the current question, shown in the sidebar under the model picker
st.sidebar.subheader("Workflow")
workflow_placeholder = st.sidebar.empty()


def show_workflow(workflow):
    # workflow_placeholder lives in the sidebar; updating it redraws in place
    workflow_placeholder.markdown("\n\n".join(
        f":green[**☑ {step}**]" if workflow.get(step) else f"☐ {step}" for step in WORKFLOW_STEPS
    ))


show_workflow(state.get("workflow", {}))

# this is where the user will type their message. st.chat_input() is a built-in Streamlit component that creates a text input and automatically scrolls to the bottom of the page when new messages are added.
prompt = st.chat_input("Type your message")

# A toast sent right before st.rerun() is lost, so it is queued in session state and shown after the redraw
if "toast" in st.session_state:
    message, icon = st.session_state.pop("toast")
    st.toast(message, icon=icon)

# run this logic every time the user submits a new message
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        context = advisor_context(MODELS[model_name](), memory=st.session_state.agent_memory)
        events = advisor.stream({"messages": [HumanMessage(prompt)]}, config, context=context, stream_mode=STREAM_MODES)
        final = render_stream(events, st.container(), nodes=("agent",), on_values=lambda s: show_workflow(s.get("workflow", {})))
    except Exception as e:  # skip the redraw so the error message stays visible
        st.error(f"{model_name}: {e}")
    else:
        final_response = last_reply(final)
        if final_response and create_memory:
            try:
                if observe(prompt, final_response):
                    st.session_state.toast = ("Memory updated", "🧠")
            except Exception as e:  # the observer must never break the chat
                st.session_state.toast = (f"Memory agent failed: {e}", "⚠️")
        # Redraw the page from the saved history. The live-streamed bubbles (and the progress captions) don't line up
        # with the history layout, so without this, leftovers from this run show up as "ghost" text on the next prompt.
        st.rerun()

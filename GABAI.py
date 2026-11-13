from dataclasses import asdict

import streamlit as st

from LLM import MODELS, AzureOpenAI, ContentDelta, ThinkingDelta, ToolCall
from RetAuGen.advisor import MAX_TOOL_ROUNDS, WORKFLOW_STEPS, Advisor
from RetAuGen.memory_agent import memory_agent
from Utility.agent_memory_context import AgentMemoryContext

st.set_page_config(page_title="GABAI", layout="wide")

st.title("Assistant Conversation")

# Pick the LLM; every provider streams the same Thinking / Content / ToolCall events
model_name = st.sidebar.selectbox("Model", list(MODELS))

# Switch for the memory agent: when off, no memory is created or changed (stored memory is still given to the main agent)
create_memory = st.sidebar.toggle("Create memory", value=True)

# The advisor agent (prompt, tools and the workflow state of the current question) lives in RetAuGen/advisor.py, shared with
# the test-runner page. It is kept in session state, so the approval requests of this conversation are remembered.
if "advisor" not in st.session_state:
    st.session_state.advisor = Advisor()
advisor = st.session_state.advisor
advisor.notify = st.caption  # the tools report their progress as captions in the chat; set on every run, never stored


def show_workflow():
    # workflow_placeholder lives in the sidebar; updating it redraws in place
    workflow_placeholder.markdown("\n\n".join(
        f":green[**☑ {step}**]" if advisor.workflow[step] else f"☐ {step}" for step in WORKFLOW_STEPS
    ))


advisor.on_change = show_workflow


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
        # tool_choice=required makes the model call a tool right away (no free text, no reasoning)
        saved = memory_agent(user_message, response, memory_db, AzureOpenAI(options={"temperature": 0, "tool_choice": "required"}))
        if saved:
            st.session_state.agent_memory = memory_db.get()  # the main agent's prompt picks this up on its next call
        return saved
    finally:
        memory_db.close()


# We will store the conversation history in Streamlit's session state, which allows us to keep track of messages across user interactions.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": advisor.system_prompt(st.session_state.agent_memory)}]
else:
    # Refresh the system prompt on every rerun so the date stays current if the app stays open past midnight
    st.session_state.messages[0]["content"] = advisor.system_prompt(st.session_state.agent_memory)

# Iterate through the conversation history
for msg in st.session_state.messages:
    if msg["role"] in ("system", "tool"):
        continue  # sent to the model but not shown in the UI
    # display each message in the appropriate format based on its role (user, assistant)
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("Thinking"): st.markdown(msg["thinking"])
        for tc in msg.get("tool_calls", []):
            st.caption(f"Tool call: `{tc['name']}({tc['arguments']})`")
        if msg["content"]:
            st.markdown(msg["content"])

# Workflow state of the current question, shown in the sidebar under the model picker
st.sidebar.subheader("Workflow")
workflow_placeholder = st.sidebar.empty()
show_workflow()

# this is where the user will type their message. st.chat_input() is a built-in Streamlit component that creates a text input and automatically scrolls to the bottom of the page when new messages are added.
prompt = st.chat_input("Type your message")


def generate(use_tools=True, tool_rounds=0):
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        placeholder = st.empty()

    complete_response = ""
    complete_thinking = ""
    thinking_body = None
    tool_calls: list[ToolCall] = []

    st.session_state.messages[0]["content"] = advisor.system_prompt(st.session_state.agent_memory)  # carries the latest workflow state
    try:
        provider = MODELS[model_name]()
        for event in provider.stream(st.session_state.messages, advisor.available_tools() if use_tools else None):
            if isinstance(event, ThinkingDelta):
                if thinking_body is None:
                    # first thinking chunk: open the "Thinking" expander and stream into it
                    thinking_body = thinking_placeholder.expander("Thinking", expanded=True).empty()
                complete_thinking += event.text
                thinking_body.markdown(complete_thinking + "▌")
            elif isinstance(event, ContentDelta):
                complete_response += event.text
                placeholder.markdown(complete_response + "▌")
            elif isinstance(event, ToolCall):
                tool_calls.append(event)
    except Exception as e:
        placeholder.error(f"{model_name}: {e}")
        return False

    if complete_thinking:
        # thinking is done: collapse the expander
        thinking_placeholder.expander("Thinking", expanded=False).markdown(complete_thinking)

    if tool_calls:
        # Record the model's tool calls, then each tool's result, so any provider can continue the conversation
        st.session_state.messages.append({
            "role": "assistant",
            "content": complete_response,
            "thinking": complete_thinking,
            "tool_calls": [asdict(tc) for tc in tool_calls],
        })
        for tc in tool_calls:
            st.session_state.messages.append({
                "role": "tool",
                "name": tc.name,
                "tool_call_id": tc.id,
                "content": advisor.process_tool(tc.arguments, tc.name),
            })
        # Before classifying the model gets the search / classify tools; once classified it gets only manage_approvals
        # (create / list / update / delete) and may reply any time instead of calling it. Capped so a model that
        # keeps calling tools cannot loop forever.
        return generate(use_tools=tool_rounds + 1 < MAX_TOOL_ROUNDS, tool_rounds=tool_rounds + 1)
    else:
        placeholder.markdown(complete_response)
        advisor.mark_done("Synthesize")
        st.session_state.messages.append({"role": "assistant", "content": complete_response, "thinking": complete_thinking})
        return True

# A toast sent right before st.rerun() is lost, so it is queued in session state and shown after the redraw
if "toast" in st.session_state:
    message, icon = st.session_state.pop("toast")
    st.toast(message, icon=icon)

# run this logic every time the user submits a new message
if prompt:
    advisor.start_question()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    if generate():  # on error, skip the redraw so the error message stays visible
        final_response = st.session_state.messages[-1]["content"]
        if final_response and create_memory:
            try:
                if observe(prompt, final_response):
                    st.session_state.toast = ("Memory updated", "🧠")
            except Exception as e:  # the observer must never break the chat
                st.session_state.toast = (f"Memory agent failed: {e}", "⚠️")
        # Redraw the page from the saved history. The live-streamed bubbles (and the classification
        # caption) don't line up with the history layout, so without this, leftovers from this run
        # show up as "ghost" text on the next prompt until the new response finishes.
        st.rerun()

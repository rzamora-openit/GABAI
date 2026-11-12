import datetime
from dataclasses import asdict

import streamlit as st

from LLM import MODELS, AzureOpenAI, ContentDelta, OllamaEmbedder, ThinkingDelta, ToolCall
from RetAuGen.retrieval_pipeline import rerank_agent, retrieve
from Utility.vector_db_context import VectorDbContext

TOP_N = 5  # chunks fetched from the vector db before re-ranking
CHUNK_FOLDER = "KnowledgeBase/chunks"

st.set_page_config(page_title="GABAI", layout="wide")

st.title("Assistant Conversation")

# Pick the LLM; every provider streams the same Thinking / Content / ToolCall events
model_name = st.sidebar.selectbox("Model", list(MODELS))

# Workflow state for the current question: the steps the agent must complete, in order.
WORKFLOW_STEPS = ("Retrieve", "Classify", "Synthesize")

def new_workflow():
    return {step: False for step in WORKFLOW_STEPS}

def workflow_text():
    """Serialize the workflow state as text, e.g. 'Retrieve: Done'."""
    state = st.session_state.workflow
    return "\n".join(f"{step}: {'Done' if state[step] else 'Not Done'}" for step in WORKFLOW_STEPS)

if "workflow" not in st.session_state:
    st.session_state.workflow = new_workflow()

def show_workflow():
    # workflow_placeholder lives in the sidebar; updating it redraws in place
    state = st.session_state.workflow
    workflow_placeholder.markdown("\n\n".join(
        f":green[**☑ {step}**]" if state[step] else f"☐ {step}" for step in WORKFLOW_STEPS
    ))

def mark_done(step):
    st.session_state.workflow[step] = True
    show_workflow()

def process_tool(params, tool_name):
    if tool_name == "search_documents":
        return search_documents(params)
    if tool_name == "classify_question":
        return classify_question(params)
    return f"Tool {tool_name} not found."

def search_documents(params):
    intent = str(params.get("intent", "")).strip()
    st.caption(f"Searching documents: **{intent}**")
    db = VectorDbContext()
    try:
        texts = retrieve(intent, TOP_N, CHUNK_FOLDER, OllamaEmbedder(), db)
    finally:
        db.close()
    relevant = rerank_agent(intent, texts, AzureOpenAI(options={"temperature": 0}))
    mark_done("Retrieve")
    st.caption(f"Relevant chunks found: **{len(relevant)}**")
    if not relevant:
        return "No relevant document or information was found."
    return "\n\n".join(f"[Document {i}]\n{text}" for i, text in enumerate(relevant, start=1))

def classify_question(params):
    classification = str(params.get("classification", "")).strip().lower()
    st.caption(f"Classification: **{classification}**")
    if not st.session_state.workflow["Retrieve"]:
        return "Classify is not allowed yet: Retrieve is Not Done. Call search_documents first."
    if classification in ("answer", "escalate", "dismiss"):
        mark_done("Classify")
    if classification == "answer":
        return "answer: reply to the student's question using only the retrieved documents."
    if classification == "escalate":
        return ("escalate: do NOT answer the question. Tell the student it is outside the documents "
                "and has been escalated to a human advisor.")
    if classification == "dismiss":
        return ("dismiss: do NOT answer the question and do NOT escalate it. Politely tell the student that you "
                "can only help with academic matters at the university.")
    return (f"Invalid classification '{classification}'. "
            "Call classify_question again with 'answer', 'escalate' or 'dismiss'.")

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the university's documents. Pass the intent behind the student's question: what they "
                "actually want to know, as a short standalone statement, "
                "paraphase the sentence to be optimized for vector embedding semantic similarity search. "
                "Returns the relevant document sections, or says that no relevant information was found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "'requirements and procedure to apply for enrollment'",
                    }
                },
                "required": ["intent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_question",
            "description": (
                "Classify the student's question after searching. "
                "'answer': the retrieved documents contain the information needed. "
                "'escalate': the question is an academic matter at the university (enrollment, courses, grades, "
                "programs, applications, student records and the like), but the documents do not cover it. "
                "'dismiss': there is no evidence that the question is relevant to any academic matter, event and announcement related to the university, or any subject that a human advisor would not handle either."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {"type": "string", "enum": ["answer", "escalate", "dismiss"]}
                },
                "required": ["classification"]
            }
        }
    }
]


def system_prompt():
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    return f"""
         You are an Academic Advisor. The users are students seeking academic advice.
         Today's date is {today}.
         Workflow state for the current question (complete the steps in order; a step is Done only
         after its tool has been called successfully):
{workflow_text()}
         Answer only from the university's documents (found with search_documents), never from your own
         knowledge. When the documents cannot accommodate an academic question, you stop and escalate it.
         When a question is not about academic matters at all, you dismiss it.

         For every student question, follow these steps in order:
         1. Retrieve: call search_documents with the intent behind the question, i.e. what the student
            actually wants to know, as a short standalone statement. Resolve references to earlier
            messages so the intent makes sense on its own.
         2. Classify: after reading the search result, call classify_question with one of:
            - answer: the retrieved documents contain the information needed.
            - escalate: an academic matter at the university that the documents do not cover;
              a human advisor should handle it.
            - dismiss: outside academic advising altogether (e.g. shopping, weather, coding help,
              general trivia); not something to escalate.
         3. Synthesize: follow the classification result. When answering, use only the retrieved documents,
            be concise, and name the procedure or section you used. Never invent
            requirements, fees, offices or deadlines.
         """


# We will store the conversation history in Streamlit's session state, which allows us to keep track of messages across user interactions.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt()}]
else:
    # Refresh the system prompt on every rerun so the date stays current if the app stays open past midnight
    st.session_state.messages[0]["content"] = system_prompt()

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

    st.session_state.messages[0]["content"] = system_prompt()  # carries the latest workflow state
    try:
        provider = MODELS[model_name]()
        for event in provider.stream(st.session_state.messages, tools if use_tools else None):
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
                "content": process_tool(tc.arguments, tc.name),
            })
        # Tools stay on until the question is classified (search -> classify), then turn off so the model
        # must write the reply. Capped at 3 tool rounds so a model that never classifies cannot loop forever.
        classified = st.session_state.workflow["Classify"]
        return generate(use_tools=not classified and tool_rounds < 2, tool_rounds=tool_rounds + 1)
    else:
        placeholder.markdown(complete_response)
        mark_done("Synthesize")
        st.session_state.messages.append({"role": "assistant", "content": complete_response, "thinking": complete_thinking})
        return True

# run this logic every time the user submits a new message
if prompt:
    st.session_state.workflow = new_workflow()
    show_workflow()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    if generate():  # on error, skip the redraw so the error message stays visible
        # Redraw the page from the saved history. The live-streamed bubbles (and the classification
        # caption) don't line up with the history layout, so without this, leftovers from this run
        # show up as "ghost" text on the next prompt until the new response finishes.
        st.rerun()
    
"""GABAI chat as a Dash app: the same advisor agent (RetAuGen/advisor.py) and memory agent (RetAuGen/memory_agent.py)
as the Streamlit page GABAI.py, with the same behaviour:

- model picker, "Create memory" switch and the workflow steps (Retrieve / Classify / Synthesize) in the sidebar
- streamed thinking (collapsible) and reply, tool progress lines as captions, tool calls shown in the history
- after a reply, the memory agent decides whether to update the agent memory ("Memory updated" toast)

Each browser tab is one conversation (like a Streamlit session); reloading the page starts a new one.
A question runs in a background thread on the server, and the page polls it to draw the streamed text.

Run from the project root:   python GABAI_dash.py      then open http://127.0.0.1:8050
"""
import threading
import time
import uuid
from dataclasses import dataclass, field

from dash import Dash, Input, Output, State, dcc, html, no_update
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from LLM import MODELS, azure_openai
from RetAuGen.advisor import WORKFLOW_STEPS, advisor_context, build_advisor, last_reply, new_workflow
from RetAuGen.memory_agent import memory_agent
from Utility.agent_memory_context import AgentMemoryContext
from Utility.chat_ui import STREAM_MODES, thinking_of

POLL_MS = 300  # how often the page redraws while the advisor is working
SESSION_IDLE_SECONDS = 6 * 3600  # conversations unused this long are dropped from server memory


# ---- conversation state (what st.session_state holds in GABAI.py) ----

@dataclass
class Bubble:
    """One streamed model call of the turn in progress."""
    thinking: str = ""
    content: str = ""
    done: bool = False
    error: str = ""
    calls: dict = field(default_factory=dict)  # tool names the model called, by index


def load_memory():
    memory_db = AgentMemoryContext()
    try:
        return memory_db.get()
    finally:
        memory_db.close()


class Session:
    def __init__(self):
        self.last_used = time.time()
        self.running = False
        self.toast = None  # (message, icon), shown once
        # The items of the turn in progress, in order: Bubble (a model call) or str (a tool's progress line).
        # Cleared when the turn succeeds, so the page draws the saved history instead (GABAI.py's st.rerun()).
        self.live = []
        # The advisor agent is a LangGraph graph (RetAuGen/advisor.py). Its checkpointer keeps this conversation
        # (messages, workflow state, approval requests) under one thread id.
        self.advisor = build_advisor(checkpointer=InMemorySaver())
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        # Agent memory, (last_update, memory) or None. Loaded from the db once per conversation, and refreshed whenever
        # the memory agent saves; the advisor puts it at the tail of the main agent's prompt.
        self.memory_warning = ""
        try:
            self.agent_memory = load_memory()
        except Exception as e:  # the app must still work without the memory db
            self.agent_memory = None
            self.memory_warning = f"Agent memory unavailable: {e}"
        self.history = []  # the saved conversation (LangChain messages), drawn when no turn is in progress
        self.workflow = new_workflow()  # steps done for the current question, from the graph state


sessions: dict[str, Session] = {}
sessions_lock = threading.Lock()


def get_session(session_id) -> Session:
    with sessions_lock:
        now = time.time()
        for sid in [sid for sid, s in sessions.items() if not s.running and now - s.last_used > SESSION_IDLE_SECONDS]:
            del sessions[sid]
        if session_id not in sessions:
            sessions[session_id] = Session()
        session = sessions[session_id]
        session.last_used = now
        return session


# ---- the agent loop (GABAI.py's run, without the drawing) ----

def stream_into(s: Session, events):
    """Draw a LangGraph stream made with stream_mode=STREAM_MODES into s.live, like chat_ui.render_stream does for
    Streamlit, and return the last graph state. The model tokens of the "agent" node become bubbles; progress lines
    written by the tools become captions; the full state after each step updates the workflow steps."""
    bubble, bubble_id, state = None, None, {}
    try:
        for mode, data in events:
            if mode == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessageChunk):
                    continue
                if bubble is None or chunk.id != bubble_id:
                    if bubble:
                        bubble.done = True
                    bubble, bubble_id = Bubble(), chunk.id
                    s.live.append(bubble)
                bubble.thinking += thinking_of(chunk)
                bubble.content += chunk.text
                for call in chunk.tool_call_chunks:
                    if call.get("name"):
                        bubble.calls[call.get("index", len(bubble.calls))] = call["name"]
            elif mode == "custom":
                if isinstance(data, dict) and "progress" in data:
                    s.live.append(data["progress"])
            elif mode == "values":
                state = data
                s.workflow = data.get("workflow", {})
    finally:
        if bubble:
            bubble.done = True
    return state


def observe(s: Session, user_message, response):
    """Independent observer: the memory agent decides if this exchange is worth saving to the agent memory."""
    memory_db = AgentMemoryContext()
    try:
        saved = memory_agent(user_message, response, memory_db, azure_openai(temperature=0))
        if saved:
            s.agent_memory = memory_db.get()  # the main agent's prompt picks this up on its next call
        return saved
    finally:
        memory_db.close()


def run_turn(s: Session, prompt, model_name, create_memory):
    """One student question, run in a background thread."""
    try:
        try:
            context = advisor_context(MODELS[model_name](), memory=s.agent_memory)
            events = s.advisor.stream({"messages": [HumanMessage(prompt)]}, s.config, context=context, stream_mode=STREAM_MODES)
            final = stream_into(s, events)
        except Exception as e:  # keep the live bubbles so the error message stays visible
            s.live.append(Bubble(error=f"{model_name}: {e}", done=True))
        else:
            final_response = last_reply(final)
            if final_response and create_memory:
                try:
                    if observe(s, prompt, final_response):
                        s.toast = ("Memory updated", "\U0001F9E0")
                except Exception as e:  # the observer must never break the chat
                    s.toast = (f"Memory agent failed: {e}", "\u26A0\uFE0F")
            s.history = list(final.get("messages", []))
            s.live.clear()  # draw the saved history from now on
    finally:
        s.running = False


# ---- drawing ----

def caption(text):
    return dcc.Markdown(text, className="caption")


def chat_message(role, children):
    return html.Div([html.Div("🧑" if role == "user" else "🤖", className="avatar"), html.Div(children, className="body")],
                    className=f"message {role}")


def thinking_box(text, is_open):
    return html.Details([html.Summary("Thinking"), dcc.Markdown(text)], open=is_open, className="thinking")


def history_items(messages):
    """The saved conversation (tool results and the system prompt are sent to the model but not shown)."""
    items = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            items.append(chat_message("user", [dcc.Markdown(msg.text)]))
        elif isinstance(msg, AIMessage):
            body = []
            if thinking_of(msg):
                body.append(thinking_box(thinking_of(msg), False))
            for call in msg.tool_calls:
                body.append(caption(f"Tool call: `{call['name']}({call['args']})`"))
            if msg.text:
                body.append(dcc.Markdown(msg.text))
            items.append(chat_message("assistant", body))
    return items


def live_items(live):
    """The turn in progress: streamed bubbles (with a cursor while streaming) and the tools' progress lines."""
    items = []
    for item in list(live):
        if isinstance(item, str):
            items.append(caption(item))
            continue
        body = []
        if item.thinking:
            # open while thinking streams in, collapsed once the call is done
            body.append(thinking_box(item.thinking + ("" if item.done else "▌"), not item.done))
        if item.content or not item.done:
            body.append(dcc.Markdown(item.content + ("" if item.done else "▌")))
        for name in item.calls.values():
            body.append(caption(f"Tool call: `{name}`"))
        if item.error:
            body.append(html.Div(item.error, className="error"))
        items.append(chat_message("assistant", body))
    return items


def workflow_items(workflow):
    return [html.Div(f"☑ {step}", className="step done") if workflow.get(step) else html.Div(f"☐ {step}", className="step")
            for step in WORKFLOW_STEPS]


# ---- page ----

CSS = """
body { margin: 0; font-family: "Source Sans Pro", "Segoe UI", sans-serif; color: #31333f; background: #fff; }
.app { display: flex; height: 100vh; }
.sidebar { width: 300px; flex-shrink: 0; background: #f0f2f6; padding: 24px 20px; box-sizing: border-box; overflow-y: auto; }
.sidebar label.title { display: block; font-size: 14px; margin: 0 0 6px; }
.sidebar h3 { margin: 28px 0 12px; font-size: 20px; }
.sidebar .switch { margin-top: 18px; font-size: 14px; }
.step { margin-bottom: 14px; }
.step.done { color: #21c354; font-weight: 700; }
.warning { background: #fffce7; color: #926c05; padding: 10px 12px; border-radius: 8px; margin-top: 16px; font-size: 14px; }
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.main h1 { margin: 0; padding: 32px 48px 8px; font-size: 40px; }
.chat { flex: 1; overflow-y: auto; padding: 8px 48px 24px; }
.message { display: flex; gap: 12px; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; }
.message.user { background: #f0f2f6; }
.message .avatar { width: 32px; height: 32px; flex-shrink: 0; border-radius: 6px; display: flex; align-items: center; justify-content: center; background: #ffbd45; }
.message.user .avatar { background: #ff4b4b; }
.message .body { flex: 1; min-width: 0; }
.message .body p:first-child { margin-top: 4px; }
.caption, .caption p { color: #808495; font-size: 14px; margin: 4px 0; }
.thinking { border: 1px solid #e6e9ef; border-radius: 8px; padding: 8px 12px; margin: 4px 0 8px; }
.thinking summary { cursor: pointer; }
.error { background: #ffecec; color: #7d353b; padding: 10px 12px; border-radius: 8px; margin-top: 4px; }
.input-row { display: flex; gap: 8px; padding: 16px 48px 24px; }
.input-row input { flex: 1; padding: 12px 14px; border: 1px solid #d5d8de; border-radius: 8px; font-size: 16px; background: #f0f2f6; }
.input-row button { padding: 0 20px; border: none; border-radius: 8px; background: #ff4b4b; color: #fff; font-size: 16px; cursor: pointer; }
.input-row button:disabled, .input-row input:disabled { opacity: .5; cursor: not-allowed; }
.toast { position: fixed; right: 24px; bottom: 24px; background: #fff; box-shadow: 0 4px 16px rgba(0,0,0,.15);
         padding: 12px 18px; border-radius: 8px; animation: toast 4s forwards; }
@keyframes toast { 0% { opacity: 0; } 8% { opacity: 1; } 85% { opacity: 1; } 100% { opacity: 0; visibility: hidden; } }
"""

app = Dash(__name__, title="GABAI")
server = app.server  # for a WSGI server, e.g. waitress-serve --port=8050 GABAI_dash:server
app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + CSS + "</style>")


def layout():
    return html.Div([
        dcc.Store(id="session-id", data=str(uuid.uuid4())),  # a new conversation per page load
        dcc.Interval(id="poll", interval=POLL_MS, disabled=True),
        html.Div([
            html.Label("Model", className="title"),
            # Pick the LLM; every model is a LangChain chat model, so the advisor graph runs with any of them
            dcc.Dropdown(id="model", options=list(MODELS), value=list(MODELS)[0], clearable=False),
            # Switch for the memory agent: when off, no memory is created or changed (stored memory is still given to the main agent)
            dcc.Checklist(id="create-memory", options=[{"label": " Create memory", "value": "on"}], value=["on"], className="switch"),
            html.Div(id="memory-warning"),
            # Workflow state of the current question
            html.H3("Workflow"),
            html.Div(id="workflow"),
        ], className="sidebar"),
        html.Div([
            html.H1("Assistant Conversation"),
            html.Div(id="chat", className="chat"),
            html.Div([
                dcc.Input(id="prompt", type="text", placeholder="Type your message", debounce=False, autoComplete="off"),
                html.Button("Send", id="send"),
            ], className="input-row"),
        ], className="main"),
        html.Div(id="toast"),
    ], className="app")


app.layout = layout


@app.callback(
    Output("prompt", "value"),
    Output("poll", "disabled", allow_duplicate=True),
    Input("send", "n_clicks"),
    Input("prompt", "n_submit"),
    State("prompt", "value"),
    State("model", "value"),
    State("create-memory", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def submit(_clicks, _submits, prompt, model_name, create_memory, session_id):
    """Run this logic every time the user submits a new message."""
    prompt = (prompt or "").strip()
    s = get_session(session_id)
    if not prompt or s.running:
        return no_update, no_update
    s.running = True
    s.live.clear()  # drop a previous error
    s.workflow = new_workflow()
    s.history.append(HumanMessage(prompt))  # shown at once; the graph saves it with the rest when the turn is done
    threading.Thread(target=run_turn, args=(s, prompt, model_name, bool(create_memory)), daemon=True).start()
    return "", False


@app.callback(
    Output("chat", "children"),
    Output("workflow", "children"),
    Output("memory-warning", "children"),
    Output("toast", "children"),
    Output("poll", "disabled"),
    Output("send", "disabled"),
    Output("prompt", "disabled"),
    Input("poll", "n_intervals"),
    Input("session-id", "data"),
)
def render(_ticks, session_id):
    """Draw the conversation; while the advisor is working, redraw every POLL_MS with what it has streamed so far."""
    s = get_session(session_id)
    running = s.running  # read once, before drawing, so the last redraw happens after the turn has finished
    chat = history_items(list(s.history)) + live_items(s.live)
    warning = html.Div(s.memory_warning, className="warning") if s.memory_warning else None
    toast = no_update
    if s.toast and not running:
        message, icon = s.toast
        s.toast = None
        toast = html.Div(f"{icon} {message}", className="toast", key=str(uuid.uuid4()))  # a new key replays the fade
    return chat, workflow_items(s.workflow), warning, toast, not running, running, running


# keep the newest message in view, like st.chat_input
app.clientside_callback(
    "function(children) { const el = document.getElementById('chat'); if (el) { setTimeout(() => { el.scrollTop = el.scrollHeight; }, 0); } return window.dash_clientside.no_update; }",
    Output("chat", "title"),
    Input("chat", "children"),
)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)

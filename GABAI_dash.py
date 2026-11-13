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
from dataclasses import asdict, dataclass

from dash import Dash, Input, Output, State, dcc, html, no_update

from LLM import MODELS, AzureOpenAI, ContentDelta, ThinkingDelta, ToolCall
from RetAuGen.advisor import MAX_TOOL_ROUNDS, WORKFLOW_STEPS, Advisor
from RetAuGen.memory_agent import memory_agent
from Utility.agent_memory_context import AgentMemoryContext

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
        # The advisor agent (prompt, tools and the workflow state of the current question). Kept per conversation,
        # so the approval requests of this conversation are remembered. Its tools report progress as captions.
        self.advisor = Advisor(notify=self.live_caption)
        # Agent memory, (last_update, memory) or None. Loaded from the db once per conversation, and refreshed whenever
        # the memory agent saves; the advisor puts it at the tail of the main agent's prompt.
        self.memory_warning = ""
        try:
            self.agent_memory = load_memory()
        except Exception as e:  # the app must still work without the memory db
            self.agent_memory = None
            self.memory_warning = f"Agent memory unavailable: {e}"
        self.messages = [{"role": "system", "content": self.advisor.system_prompt(self.agent_memory)}]

    def live_caption(self, text):
        self.live.append(text)


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


# ---- the agent loop (GABAI.py's generate and observe, without the drawing) ----

def generate(s: Session, model_name, use_tools=True, tool_rounds=0):
    """Stream one model call into a new live bubble; run its tool calls and call the model again until it replies in
    text. Returns False if the model call failed (the error is shown in the bubble)."""
    bubble = Bubble()
    s.live.append(bubble)
    tool_calls: list[ToolCall] = []

    s.messages[0]["content"] = s.advisor.system_prompt(s.agent_memory)  # carries the latest workflow state
    try:
        provider = MODELS[model_name]()
        for event in provider.stream(s.messages, s.advisor.available_tools() if use_tools else None):
            if isinstance(event, ThinkingDelta):
                bubble.thinking += event.text
            elif isinstance(event, ContentDelta):
                bubble.content += event.text
            elif isinstance(event, ToolCall):
                tool_calls.append(event)
    except Exception as e:
        bubble.error = f"{model_name}: {e}"
        bubble.done = True
        return False
    bubble.done = True

    if tool_calls:
        # Record the model's tool calls, then each tool's result, so any provider can continue the conversation
        s.messages.append({
            "role": "assistant",
            "content": bubble.content,
            "thinking": bubble.thinking,
            "tool_calls": [asdict(tc) for tc in tool_calls],
        })
        for tc in tool_calls:
            s.messages.append({
                "role": "tool",
                "name": tc.name,
                "tool_call_id": tc.id,
                "content": s.advisor.process_tool(tc.arguments, tc.name),
            })
        # Before classifying the model gets the search / classify tools; once classified it gets only manage_approvals
        # (create / list / update / delete) and may reply any time instead of calling it. Capped so a model that
        # keeps calling tools cannot loop forever.
        return generate(s, model_name, use_tools=tool_rounds + 1 < MAX_TOOL_ROUNDS, tool_rounds=tool_rounds + 1)
    s.advisor.mark_done("Synthesize")
    s.messages.append({"role": "assistant", "content": bubble.content, "thinking": bubble.thinking})
    return True


def observe(s: Session, user_message, response):
    """Independent observer: the memory agent decides if this exchange is worth saving to the agent memory."""
    memory_db = AgentMemoryContext()
    try:
        # tool_choice=required makes the model call a tool right away (no free text, no reasoning)
        saved = memory_agent(user_message, response, memory_db, AzureOpenAI(options={"temperature": 0, "tool_choice": "required"}))
        if saved:
            s.agent_memory = memory_db.get()  # the main agent's prompt picks this up on its next call
        return saved
    finally:
        memory_db.close()


def run_turn(s: Session, prompt, model_name, create_memory):
    """One student question, run in a background thread."""
    try:
        if generate(s, model_name):  # on error, keep the live bubbles so the error message stays visible
            final_response = s.messages[-1]["content"]
            if final_response and create_memory:
                try:
                    if observe(s, prompt, final_response):
                        s.toast = ("Memory updated", "🧠")
                except Exception as e:  # the observer must never break the chat
                    s.toast = (f"Memory agent failed: {e}", "⚠️")
            s.live.clear()  # draw the saved history from now on
    except Exception as e:  # e.g. a tool failed (the vector db is down): show it instead of losing the thread silently
        s.live.append(Bubble(error=f"{type(e).__name__}: {e}", done=True))
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
    """The saved conversation (system prompt and tool results are sent to the model but not shown)."""
    items = []
    for msg in messages:
        if msg["role"] in ("system", "tool"):
            continue
        body = []
        if msg.get("thinking"):
            body.append(thinking_box(msg["thinking"], False))
        for tc in msg.get("tool_calls", []):
            body.append(caption(f"Tool call: `{tc['name']}({tc['arguments']})`"))
        if msg["content"]:
            body.append(dcc.Markdown(msg["content"]))
        items.append(chat_message(msg["role"], body))
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
        if item.error:
            body.append(html.Div(item.error, className="error"))
        items.append(chat_message("assistant", body))
    return items


def workflow_items(workflow):
    return [html.Div(f"☑ {step}", className="step done") if workflow[step] else html.Div(f"☐ {step}", className="step")
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
            # Pick the LLM; every provider streams the same Thinking / Content / ToolCall events
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
    s.advisor.start_question()
    s.messages.append({"role": "user", "content": prompt})
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
    chat = history_items(list(s.messages)) + live_items(s.live)
    warning = html.Div(s.memory_warning, className="warning") if s.memory_warning else None
    toast = no_update
    if s.toast and not running:
        message, icon = s.toast
        s.toast = None
        toast = html.Div(f"{icon} {message}", className="toast", key=str(uuid.uuid4()))  # a new key replays the fade
    return chat, workflow_items(s.advisor.workflow), warning, toast, not running, running, running


# keep the newest message in view, like st.chat_input
app.clientside_callback(
    "function(children) { const el = document.getElementById('chat'); if (el) { setTimeout(() => { el.scrollTop = el.scrollHeight; }, 0); } return window.dash_clientside.no_update; }",
    Output("chat", "title"),
    Input("chat", "children"),
)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)

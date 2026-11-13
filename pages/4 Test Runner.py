import csv
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from LLM import MODELS, ContentDelta, ThinkingDelta, ToolCall
from RetAuGen.advisor import MAX_TOOL_ROUNDS, WORKFLOW_STEPS, Advisor, InMemoryApprovals
from Utility.approval_context import ApprovalContext

ROOT = Path(__file__).resolve().parent.parent
QUESTION_FOLDER = ROOT / "Question"  # CSVs with a question column (and optionally an expected behavior column)
RESULT_FOLDER = QUESTION_FOLDER / "results"  # one results CSV per question CSV, same columns as AccuracyTest_Results_v3.csv
HEADER = ["Question", "Expected Behavior", "Agent Classify", "Agent Response", "Is Classify Expected", "Is Response Expected", "Note", "Retrieved Chunks"]

st.set_page_config(page_title="Test Runner", layout="wide")

st.title("Test Runner: GABAI workflow")

# Pick the LLM that plays the advisor; Azure OpenAI is the default
models = list(MODELS)
model_name = st.sidebar.selectbox("Model", models, index=models.index("Azure OpenAI"))

csv_files = sorted(QUESTION_FOLDER.glob("*.csv"))
if not csv_files:
    st.error(f"No CSV file found in {QUESTION_FOLDER}. Put a CSV with a question column there (an 'expected behavior' column is optional).")
    st.stop()
csv_file = st.sidebar.selectbox("Question CSV", csv_files, format_func=lambda f: f.name)
result_file = RESULT_FOLDER / f"{csv_file.stem}_results.csv"

save_approvals = st.sidebar.toggle(
    "Save approval requests to the database", value=False,
    help="Off: escalations create their approval requests in memory only, so a test run leaves the database untouched.",
)


def load_questions(file):
    """The CSV rows as dicts with question and expected (answer / escalate / approval / dismiss, or '' when not given)."""
    with open(file, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = {name.strip().lower(): name for name in reader.fieldnames or []}
        if "question" not in columns:
            raise ValueError(f"{file.name} needs a question column (found: {', '.join(reader.fieldnames or []) or 'none'})")
        expected_column = columns.get("expected behavior") or columns.get("expected")
        return [
            {"question": (row[columns["question"]] or "").strip(), "expected": ((row.get(expected_column) or "") if expected_column else "").strip().lower()}
            for row in reader
            if (row[columns["question"]] or "").strip()
        ]


try:
    questions = load_questions(csv_file)
except (OSError, ValueError, csv.Error) as e:
    st.error(f"Cannot read {csv_file.name}: {e}")
    st.stop()
if not questions:
    st.error(f"{csv_file.name} has no questions.")
    st.stop()

# Results live in session state (shared across pages, so the keys are prefixed). Switching CSV starts over.
if st.session_state.get("runner_file") != csv_file.name:
    st.session_state.runner_file = csv_file.name
    st.session_state.runner_rows = []
    st.session_state.runner_prompt = Advisor().system_prompt()
rows = st.session_state.runner_rows

run_all = st.sidebar.button("Run all questions")
if st.sidebar.button("Clear results"):
    st.session_state.runner_rows = []
    st.rerun()
progress_placeholder = st.sidebar.empty()
st.sidebar.subheader("Workflow")
state_placeholder = st.sidebar.empty()

with st.expander(f"System prompt ({len(questions)} questions from {csv_file.name}; it shows the question in progress)", expanded=False):
    with st.container(height=300):
        prompt_placeholder = st.empty()
prompt_placeholder.code(st.session_state.runner_prompt, language=None, wrap_lines=True)

live_slot = st.empty()  # the advisor's streaming reply and progress lines for the current question
st.subheader("Results")
results_placeholder = st.empty()


def show_results():
    if not rows:
        results_placeholder.markdown("No results yet. Click **Run all questions**.")
        return
    with results_placeholder.container():
        st.dataframe([
            {
                "#": n,
                "Question": r["Question"],
                "Expected": r["Expected Behavior"],
                "Classify": r["Agent Classify"],
                "Classify OK": r["Is Classify Expected"],
                "Chunks": r["Retrieved Chunks"].replace("UPOU Academic Procedures.", "").replace(".md", ""),
                "Response": r["Agent Response"],
            }
            for n, r in enumerate(rows, start=1)
        ], hide_index=True)
        st.caption(f"Saved to {result_file}")


def write_results():
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def show_workflow(advisor):
    """The advisor's workflow steps for the question in progress, as checkboxes (green when done)."""
    state_placeholder.markdown("\n\n".join(
        f":green[**☑ {step}**]" if advisor.workflow[step] else f"☐ {step}" for step in WORKFLOW_STEPS
    ))


def ask(advisor, box, messages, use_tools=True, tool_rounds=0):
    """The advisor's reply loop, the same as GABAI's: stream the model, run its tool calls, and ask again until it replies in text."""
    with box:
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            placeholder = st.empty()

    prompt = advisor.system_prompt()  # carries the latest workflow state
    messages[0]["content"] = prompt
    st.session_state.runner_prompt = prompt
    prompt_placeholder.code(prompt, language=None, wrap_lines=True)

    reply, thinking, thinking_body = "", "", None
    tool_calls: list[ToolCall] = []
    provider = MODELS[model_name]()
    for event in provider.stream(messages, advisor.available_tools() if use_tools else None):
        if isinstance(event, ThinkingDelta):
            if thinking_body is None:
                thinking_body = thinking_placeholder.expander("Thinking", expanded=True).empty()
            thinking += event.text
            thinking_body.markdown(thinking + "▌")
        elif isinstance(event, ContentDelta):
            reply += event.text
            placeholder.markdown(reply + "▌")
        elif isinstance(event, ToolCall):
            tool_calls.append(event)

    if thinking:
        thinking_placeholder.expander("Thinking", expanded=False).markdown(thinking)

    if tool_calls:
        messages.append({"role": "assistant", "content": reply, "thinking": thinking, "tool_calls": [asdict(tc) for tc in tool_calls]})
        for tc in tool_calls:
            messages.append({"role": "tool", "name": tc.name, "tool_call_id": tc.id, "content": advisor.process_tool(tc.arguments, tc.name)})
        return ask(advisor, box, messages, use_tools=tool_rounds + 1 < MAX_TOOL_ROUNDS, tool_rounds=tool_rounds + 1)

    placeholder.markdown(reply)
    advisor.mark_done("Synthesize")
    return reply


def run_question(i, approval_store):
    """Run question i through the whole GABAI workflow in a fresh conversation, record what the advisor did, then recurse to the next question."""
    if i >= len(questions):
        progress_placeholder.progress(1.0, text=f"Done: {len(questions)} questions")
        state_placeholder.empty()
        live_slot.empty()
        return
    item = questions[i]
    progress_placeholder.progress(i / len(questions), text=f"Running question {i + 1} of {len(questions)}")

    box = live_slot.container()
    advisor = Advisor(approvals=ApprovalContext if save_approvals else (lambda: approval_store))
    advisor.notify = box.caption  # the tools' progress lines (searching, classification, approvals)
    advisor.on_change = lambda: show_workflow(advisor)
    advisor.start_question()
    with box:
        with st.chat_message("user"):
            st.markdown(item["question"])

    messages = [{"role": "system", "content": advisor.system_prompt()}, {"role": "user", "content": item["question"]}]
    reply, error = None, None
    try:
        reply = ask(advisor, box, messages)
    except Exception as e:  # one failed question (a filtered prompt, a rate limit, a db error) must not stop the run
        error = f"{type(e).__name__}: {e}"
        box.error(f"{model_name}: {error}")

    note = f"Tools called: {', '.join(advisor.tools_called) or 'none'}."
    if advisor.approval_requests:
        note += " Approval request created: " + " | ".join(advisor.approval_requests) + "."
    if error:
        note += f" Error: {error}"
    rows.append({
        "Question": item["question"],
        "Expected Behavior": item["expected"],
        "Agent Classify": advisor.classification or "(none)",
        "Agent Response": (reply or "").strip() or "(no reply - the request failed)",
        "Is Classify Expected": str(advisor.classification == item["expected"]).upper() if item["expected"] else "",
        "Is Response Expected": "",  # judged separately, e.g. on the LaaJ page
        "Note": note,
        "Retrieved Chunks": ",".join(advisor.retrieved),
    })
    write_results()
    show_results()
    run_question(i + 1, approval_store)


show_results()
if run_all:
    rows.clear()
    show_results()
    run_question(0, InMemoryApprovals())

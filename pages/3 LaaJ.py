import csv
import re
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from LLM import MODELS, ContentDelta, ThinkingDelta, ToolCall

ROOT = Path(__file__).resolve().parent.parent
EVAL_FOLDER = ROOT / "Eval"  # the question,answer[,chunks] CSVs to judge
CHUNK_FOLDER = ROOT / "KnowledgeBase" / "chunks"  # where the retrieved files named in the CSV's chunks column live
MAX_ROUNDS = 6  # model calls per question (file fetches, the score, and corrections) before it is recorded as it is

st.set_page_config(page_title="LaaJ", layout="wide")

st.title("LaaJ: LLM as a Judge")

# Pick the LLM; every provider streams the same Thinking / Content / ToolCall events. Azure OpenAI is the default for judging.
models = list(MODELS)
model_name = st.sidebar.selectbox("Model", models, index=models.index("Azure OpenAI"))

csv_files = sorted(EVAL_FOLDER.glob("*.csv"))
if not csv_files:
    st.error(f"No CSV file found in {EVAL_FOLDER}. Put a CSV with the columns question,answer,chunks there.")
    st.stop()
csv_file = st.sidebar.selectbox("Eval CSV", csv_files, format_func=lambda f: f.name)


def load_pairs(file):
    """The CSV rows as dicts with question, answer and files (the chunk file names in the chunks column; None if the CSV has no such column)."""
    with open(file, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = {name.strip().lower(): name for name in reader.fieldnames or []}
        answer_column = columns.get("answer") or columns.get("agent response")  # "agent response" is the Test Runner's column name
        if "question" not in columns or not answer_column:
            raise ValueError(f"{file.name} needs the columns question,answer (found: {', '.join(reader.fieldnames or []) or 'none'})")
        chunks_column = columns.get("chunks") or columns.get("retrieved chunks")
        pairs = []
        for row in reader:
            files = None
            if chunks_column:
                files = [name.strip() for name in (row.get(chunks_column) or "").split(",") if name.strip()]
            pairs.append({"question": (row[columns["question"]] or "").strip(), "answer": (row[answer_column] or "").strip(), "files": files})
        return pairs


try:
    pairs = load_pairs(csv_file)
except (OSError, ValueError, csv.Error) as e:
    st.error(f"Cannot read {csv_file.name}: {e}")
    st.stop()
if not pairs:
    st.error(f"{csv_file.name} has no question-answer rows.")
    st.stop()
if pairs[0]["files"] is None:
    st.warning(f"{csv_file.name} has no chunks column, so grounding cannot be evaluated (it is scored N/A). Pick a CSV with a chunks column to evaluate grounding.")

fetch_file_tool = {
    "type": "function",
    "function": {
        "name": "fetch_file",
        "description": "Fetch the content of one retrieved file listed in the current evaluation, to check the answer against it.",
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string", "description": "A file name exactly as listed under 'Retrieved files'."}},
            "required": ["filename"],
        },
    },
}


def system_prompt(i):
    """The judge's instructions plus the evaluation currently in progress (pair i, 0-based)."""
    pair = pairs[i]
    if pair["files"] is None:
        retrieved = "Unknown: this CSV has no chunks column. Do not fetch anything and score grounding as N/A."
    elif pair["files"]:
        retrieved = "\n".join(f"- {name}" for name in pair["files"])
    else:
        retrieved = "(none: no document was retrieved for this question)"
    return f"""
You are a judge that evaluates one answer at a time. You are now evaluating question {i + 1} of {len(pairs)}.
Score the answer from 0 to 10, with whole numbers, on three criteria:
- Relevance: how directly the answer addresses what the question asks (0 = unrelated, 10 = fully on point with nothing off-topic).
- Completeness: how fully the answer covers every part of the question (0 = nothing covered, 10 = every part covered in full).
- Grounding: how well the claims in the answer are supported by the content of the retrieved files (0 = nothing is supported or it contradicts the files, 10 = every claim is supported).
  Use only the fetched file content as evidence. If no document was retrieved, score 10 when the answer makes no factual claims about the university (for example a refusal or an escalation), otherwise 0.

Current evaluation:

Question {i + 1}: {pair["question"]}

Answer: {pair["answer"]}

Retrieved files:
{retrieved}

Follow these steps in order:
1. Fetch: call fetch_file once for each retrieved file listed above, to read its content. Skip this step when there are none.
2. Score: give the three scores. Reply with only one Markdown table row and no other text, notes or explanation, with the question number
   and the scores in this order: | {i + 1} | relevance | completeness | grounding |
"""


# Results live in session state (shared across pages, so the keys are prefixed). Switching CSV starts over.
if st.session_state.get("laaj_file") != csv_file.name:
    st.session_state.laaj_file = csv_file.name
    st.session_state.laaj_results = []
    st.session_state.laaj_prompt = system_prompt(0)
results = st.session_state.laaj_results

evaluate_all = st.sidebar.button("Evaluate all pairs")
if st.sidebar.button("Clear results"):
    st.session_state.laaj_results = []
    st.session_state.laaj_prompt = system_prompt(0)
    st.rerun()
progress_placeholder = st.sidebar.empty()
st.sidebar.subheader("Current evaluation")
state_placeholder = st.sidebar.empty()

with st.expander(f"System prompt ({len(pairs)} question-answer pairs from {csv_file.name}; it shows the evaluation in progress)", expanded=True):
    with st.container(height=300):
        prompt_placeholder = st.empty()
prompt_placeholder.code(st.session_state.laaj_prompt, language=None, wrap_lines=True)

live_slot = st.empty()  # the agent's streaming reply and tool calls for the current question
st.subheader("Results")
results_placeholder = st.empty()
details_placeholder = st.empty()


def show_results():
    lines = ["| Question | Relevance | Completeness | Grounding |", "|---|---:|---:|---:|"]
    for r in results:
        scores = r["scores"] or ("?", "?", "?")
        lines.append(f"| {r['n']} | {scores[0]} | {scores[1]} | {scores[2]} |")
    results_placeholder.markdown("\n".join(lines) if results else "No results yet. Click **Evaluate all pairs**.")
    with details_placeholder.container():
        if results:
            with st.expander("Details (files fetched and raw replies)"):
                for r in results:
                    fetched = ", ".join(r["fetched"]) or "none"
                    st.markdown(f"**Question {r['n']}**  \nFiles fetched: {fetched}  \nReply: `{r['reply'] or '(no reply)'}`")


def show_state(i, ev):
    """The fetch / score steps of the evaluation in progress, as checkboxes (green when done)."""
    files = pairs[i]["files"] or []
    fetch = f"Fetch files ({len(ev['fetched'])}/{len(files)})" if files else "Fetch files (none to fetch)"
    steps = [(fetch, len(ev["fetched"]) >= len(files)), ("Score", ev["scored"])]
    state_placeholder.markdown("\n\n".join(f":green[**☑ {label}**]" if done else f"☐ {label}" for label, done in steps))


ROW = re.compile(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+|N/A)\s*\|")


def parse_row(reply, n):
    """(relevance, completeness, grounding) from a '| n | r | c | g |' row for question n, or None if it is missing or invalid."""
    match = ROW.search(reply or "")
    if not match or int(match[1]) != n:
        return None
    relevance, completeness = int(match[2]), int(match[3])
    grounding = match[4] if match[4] == "N/A" else int(match[4])
    scores = [relevance, completeness] + ([] if grounding == "N/A" else [grounding])
    return (relevance, completeness, grounding) if all(0 <= s <= 10 for s in scores) else None


def fetch_file(i, params, ev):
    filename = str(params.get("filename", "")).strip()
    allowed = pairs[i]["files"] or []
    if filename not in allowed:
        return f"File not available. You can only fetch: {', '.join(allowed) or 'no files'}."
    try:
        content = (CHUNK_FOLDER / filename).read_text(encoding="utf-8")
    except OSError as e:
        return f"Cannot read {filename}: {e}"
    ev["fetched"].add(filename)
    show_state(i, ev)
    return content


def correction(i, ev, reply):
    """What to tell the model when its reply is not acceptable yet; None when it is."""
    missing = [name for name in pairs[i]["files"] or [] if name not in ev["fetched"]]
    if missing:
        return f"Fetch the remaining files with fetch_file before scoring: {', '.join(missing)}."
    if parse_row(reply, i + 1) is None:
        return f"Reply with only one table row in this form, with whole numbers from 0 to 10: | {i + 1} | relevance | completeness | grounding |"
    return None


def judge(i, ev, box, messages, rounds=0):
    """One model call for question i. Fetch tool calls are run and the model is called again; a reply that skipped the
    fetches or is not a valid row is sent back with a correction. Returns the final reply, or None if the model call failed.
    The recursion is bounded by MAX_ROUNDS."""
    with box:
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            placeholder = st.empty()

    reply, thinking, thinking_body = "", "", None
    tool_calls: list[ToolCall] = []
    try:
        provider = MODELS[model_name]()
        for event in provider.stream(messages, [fetch_file_tool] if pairs[i]["files"] else None):
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
    except Exception as e:
        placeholder.error(f"{model_name}: {e}")
        return None

    if thinking:
        thinking_placeholder.expander("Thinking", expanded=False).markdown(thinking)
    placeholder.markdown(reply)

    if tool_calls:
        messages.append({"role": "assistant", "content": reply, "thinking": thinking, "tool_calls": [asdict(tc) for tc in tool_calls]})
        for tc in tool_calls:
            with box:
                st.caption(f"Tool call: `{tc.name}({tc.arguments})`")
            messages.append({"role": "tool", "name": tc.name, "tool_call_id": tc.id, "content": fetch_file(i, tc.arguments, ev) if tc.name == "fetch_file" else f"Tool {tc.name} not found."})
        return judge(i, ev, box, messages, rounds + 1) if rounds + 1 < MAX_ROUNDS else reply

    messages.append({"role": "assistant", "content": reply, "thinking": thinking})
    problem = correction(i, ev, reply)
    if problem and rounds + 1 < MAX_ROUNDS:
        messages.append({"role": "user", "content": problem})
        return judge(i, ev, box, messages, rounds + 1)
    return reply


def evaluate(i):
    """Evaluate question i (the current evaluation), record its scores, then recurse to the next question."""
    if i >= len(pairs):
        progress_placeholder.progress(1.0, text=f"Done: {len(pairs)} questions evaluated")
        state_placeholder.empty()
        live_slot.empty()
        return
    progress_placeholder.progress(i / len(pairs), text=f"Evaluating question {i + 1} of {len(pairs)}")
    ev = {"fetched": set(), "scored": False}
    show_state(i, ev)

    prompt = system_prompt(i)
    st.session_state.laaj_prompt = prompt
    prompt_placeholder.code(prompt, language=None, wrap_lines=True)
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": f"Evaluate question {i + 1}."}]

    box = live_slot.container()
    reply = judge(i, ev, box, messages)
    if reply is None:  # the model call failed: stop here and keep the results so far
        return
    scores = parse_row(reply, i + 1)
    ev["scored"] = scores is not None
    show_state(i, ev)
    results.append({"n": i + 1, "scores": scores, "fetched": sorted(ev["fetched"]), "reply": reply.strip()})
    show_results()
    evaluate(i + 1)


show_results()
if evaluate_all:
    results.clear()
    show_results()
    evaluate(0)

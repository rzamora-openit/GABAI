import csv
from pathlib import Path

import streamlit as st

from LLM import MODELS
from RetAuGen.judge import JudgeContext, build_judge, judge_input, last_reply, parse_row, system_prompt
from Utility.chat_ui import STREAM_MODES, render_stream

ROOT = Path(__file__).resolve().parent.parent
EVAL_FOLDER = ROOT / "Eval"  # the question,answer[,chunks] CSVs to judge
CHUNK_FOLDER = ROOT / "KnowledgeBase" / "chunks"  # where the retrieved files named in the CSV's chunks column live

st.set_page_config(page_title="LaaJ", layout="wide")

st.title("LaaJ: LLM as a Judge")

# Pick the LLM; every model is a LangChain chat model. Azure OpenAI is the default for judging.
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


def judge_context(i, llm=None):
    """The judge graph's context for pair i (0-based)."""
    pair = pairs[i]
    return JudgeContext(llm=llm, number=i + 1, total=len(pairs), question=pair["question"], answer=pair["answer"],
                        files=pair["files"], chunk_folder=CHUNK_FOLDER)


# Results live in session state (shared across pages, so the keys are prefixed). Switching CSV starts over.
if st.session_state.get("laaj_file") != csv_file.name:
    st.session_state.laaj_file = csv_file.name
    st.session_state.laaj_results = []
    st.session_state.laaj_prompt = system_prompt(judge_context(0))
results = st.session_state.laaj_results

evaluate_all = st.sidebar.button("Evaluate all pairs")
if st.sidebar.button("Clear results"):
    st.session_state.laaj_results = []
    st.session_state.laaj_prompt = system_prompt(judge_context(0))
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


def show_state(i, fetched, scored):
    """The fetch / score steps of the evaluation in progress, as checkboxes (green when done)."""
    files = pairs[i]["files"] or []
    fetch = f"Fetch files ({len(fetched)}/{len(files)})" if files else "Fetch files (none to fetch)"
    steps = [(fetch, len(fetched) >= len(files)), ("Score", scored)]
    state_placeholder.markdown("\n\n".join(f":green[**☑ {label}**]" if done else f"☐ {label}" for label, done in steps))


def evaluate(i, graph):
    """Run the judge graph on question i, record its scores, then recurse to the next question."""
    if i >= len(pairs):
        progress_placeholder.progress(1.0, text=f"Done: {len(pairs)} questions evaluated")
        state_placeholder.empty()
        live_slot.empty()
        return
    progress_placeholder.progress(i / len(pairs), text=f"Evaluating question {i + 1} of {len(pairs)}")
    show_state(i, [], False)

    ctx = judge_context(i, MODELS[model_name]())
    prompt = system_prompt(ctx)
    st.session_state.laaj_prompt = prompt
    prompt_placeholder.code(prompt, language=None, wrap_lines=True)

    box = live_slot.container()
    try:
        events = graph.stream(judge_input(ctx), context=ctx, stream_mode=STREAM_MODES)
        final = render_stream(events, box, nodes=("judge",), on_values=lambda s: show_state(i, s.get("fetched", []), False))
    except Exception as e:  # the model call failed: stop here and keep the results so far
        box.error(f"{model_name}: {e}")
        return
    reply = last_reply(final)
    scores = parse_row(reply, i + 1)
    show_state(i, final.get("fetched", []), scores is not None)
    results.append({"n": i + 1, "scores": scores, "fetched": sorted(final.get("fetched", [])), "reply": reply.strip()})
    show_results()
    evaluate(i + 1, graph)


show_results()
if evaluate_all:
    results.clear()
    show_results()
    evaluate(0, build_judge())

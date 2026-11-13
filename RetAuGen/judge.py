"""LaaJ (LLM as a Judge), as a LangGraph graph: scores one question-answer pair for relevance, completeness and
grounding, fetching the retrieved chunk files to check the answer against them.

    START -> judge --(tool calls)--> fetch --> judge
             judge --(reply is not a valid score row yet)--> correct --> judge
             judge --(valid row, or MAX_ROUNDS model calls)--> END

    graph = build_judge()
    for mode, data in graph.stream(judge_input(ctx), context=ctx, stream_mode=["messages", "custom", "values"]):
        ...
    scores = parse_row(last_reply(state), ctx.number)

This module has no Streamlit code.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command

from RetAuGen.advisor import last_reply

MAX_ROUNDS = 6  # model calls per question (file fetches, the score, and corrections) before it is recorded as it is


@dataclass
class JudgeContext:
    """One evaluation: the pair being judged and the model judging it."""

    llm: BaseChatModel
    number: int  # 1-based question number
    total: int  # questions in the CSV
    question: str
    answer: str
    files: list[str] | None  # retrieved chunk file names; None when the CSV has no chunks column
    chunk_folder: Path


def _unique(old: list, new: list) -> list:
    return list(dict.fromkeys([*old, *new]))


class JudgeState(MessagesState):
    fetched: Annotated[list[str], _unique]  # files fetched so far
    rounds: int  # model calls so far


def system_prompt(ctx: JudgeContext) -> str:
    """The judge's instructions plus the evaluation currently in progress."""
    if ctx.files is None:
        retrieved = "Unknown: this CSV has no chunks column. Do not fetch anything and score grounding as N/A."
    elif ctx.files:
        retrieved = "\n".join(f"- {name}" for name in ctx.files)
    else:
        retrieved = "(none: no document was retrieved for this question)"
    n = ctx.number
    return f"""
You are a judge that evaluates one answer at a time. You are now evaluating question {n} of {ctx.total}.
Score the answer from 0 to 10, with whole numbers, on three criteria:
- Relevance: how directly the answer addresses what the question asks (0 = unrelated, 10 = fully on point with nothing off-topic).
- Completeness: how fully the answer covers every part of the question (0 = nothing covered, 10 = every part covered in full).
- Grounding: how well the claims in the answer are supported by the content of the retrieved files (0 = nothing is supported or it contradicts the files, 10 = every claim is supported).
  Use only the fetched file content as evidence. If no document was retrieved, score 10 when the answer makes no factual claims about the university (for example a refusal or an escalation), otherwise 0.

Current evaluation:

Question {n}: {ctx.question}

Answer: {ctx.answer}

Retrieved files:
{retrieved}

Follow these steps in order:
1. Fetch: call fetch_file once for each retrieved file listed above, to read its content. Skip this step when there are none.
2. Score: give the three scores. Reply with only one Markdown table row and no other text, notes or explanation, with the question number
   and the scores in this order: | {n} | relevance | completeness | grounding |
"""


ROW = re.compile(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+|N/A)\s*\|")


def parse_row(reply: str | None, n: int):
    """(relevance, completeness, grounding) from a '| n | r | c | g |' row for question n, or None if it is missing or invalid."""
    match = ROW.search(reply or "")
    if not match or int(match[1]) != n:
        return None
    relevance, completeness = int(match[2]), int(match[3])
    grounding = match[4] if match[4] == "N/A" else int(match[4])
    scores = [relevance, completeness] + ([] if grounding == "N/A" else [grounding])
    return (relevance, completeness, grounding) if all(0 <= s <= 10 for s in scores) else None


def correction(ctx: JudgeContext, fetched: list[str], reply: str) -> str | None:
    """What to tell the model when its reply is not acceptable yet; None when it is."""
    missing = [name for name in ctx.files or [] if name not in fetched]
    if missing:
        return f"Fetch the remaining files with fetch_file before scoring: {', '.join(missing)}."
    if parse_row(reply, ctx.number) is None:
        return f"Reply with only one table row in this form, with whole numbers from 0 to 10: | {ctx.number} | relevance | completeness | grounding |"
    return None


@tool(parse_docstring=True)
def fetch_file(filename: str, runtime: ToolRuntime[JudgeContext, JudgeState]) -> Command:
    """Fetch the content of one retrieved file listed in the current evaluation, to check the answer against it.

    Args:
        filename: A file name exactly as listed under 'Retrieved files'.
    """
    filename = filename.strip()
    allowed = runtime.context.files or []

    def reply(content, fetched=()):
        message = ToolMessage(content, tool_call_id=runtime.tool_call_id, name="fetch_file")
        return Command(update={"messages": [message], "fetched": list(fetched)})

    if filename not in allowed:
        return reply(f"File not available. You can only fetch: {', '.join(allowed) or 'no files'}.")
    try:
        content = (Path(runtime.context.chunk_folder) / filename).read_text(encoding="utf-8")
    except OSError as e:
        return reply(f"Cannot read {filename}: {e}")
    return reply(content, fetched=[filename])


def judge(state: JudgeState, runtime: Runtime[JudgeContext]) -> dict:
    ctx = runtime.context
    llm = ctx.llm.bind_tools([fetch_file]) if ctx.files else ctx.llm
    response = llm.invoke([SystemMessage(system_prompt(ctx)), *state["messages"]])
    return {"messages": [response], "rounds": state.get("rounds", 0) + 1}


def route(state: JudgeState, runtime: Runtime[JudgeContext]) -> str:
    if state["rounds"] >= MAX_ROUNDS:
        return END
    last = state["messages"][-1]
    if last.tool_calls:
        return "fetch"
    return "correct" if correction(runtime.context, state.get("fetched", []), last.text) else END


def correct(state: JudgeState, runtime: Runtime[JudgeContext]) -> dict:
    """Send the reply back with what is wrong with it."""
    problem = correction(runtime.context, state.get("fetched", []), state["messages"][-1].text)
    return {"messages": [HumanMessage(problem)]}


def build_judge():
    graph = StateGraph(JudgeState, context_schema=JudgeContext)
    graph.add_node("judge", judge)
    graph.add_node("fetch", ToolNode([fetch_file]))
    graph.add_node("correct", correct)
    graph.add_edge(START, "judge")
    graph.add_conditional_edges("judge", route, ["fetch", "correct", END])
    graph.add_edge("fetch", "judge")
    graph.add_edge("correct", "judge")
    return graph.compile()


def judge_input(ctx: JudgeContext) -> dict:
    return {"messages": [HumanMessage(f"Evaluate question {ctx.number}.")], "fetched": [], "rounds": 0}


__all__ = ["MAX_ROUNDS", "JudgeContext", "JudgeState", "build_judge", "correction", "judge_input", "last_reply",
           "parse_row", "system_prompt"]

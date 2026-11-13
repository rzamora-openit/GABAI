"""The GABAI advisor agent, as a LangGraph graph: system prompt, tools and the workflow state of the current question.

    START -> start_question -> agent --(tool calls)--> tools --+
                                 ^                             |
                                 +-----------------------------+
                               agent --(text reply)--> END

Shared by the chat page (GABAI.py) and the test-runner page, so a test run exercises the same agent the students talk to.
This module has no Streamlit code. Each invoke handles one student question:

    graph = build_advisor(checkpointer=InMemorySaver())   # the checkpointer keeps the conversation per thread_id
    config = {"configurable": {"thread_id": "..."}}
    for mode, data in graph.stream({"messages": [HumanMessage(question)]}, config,
                                   context=advisor_context(llm), stream_mode=["messages", "custom", "values"]):
        ...  # "messages": (token chunk, metadata); "custom": {"progress": str} lines from the tools;
             # "values": the full AdvisorState after each step (workflow, retrieved, classification, ...)

The graph state (AdvisorState) holds what the old Advisor class kept: the workflow steps, the chunks retrieved, the
classification, the tools called and the approval requests (per question), and the approval ids (per conversation).
"""
import datetime
import itertools
import operator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from langchain_core.documents import BaseDocumentCompressor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime, tools_condition
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite

from Utility.approval_context import Approval, ApprovalContext
from Utility.config import PROJECT_ROOT

TOP_N = 5  # chunks fetched from the vector db before re-ranking
MAX_TOOL_ROUNDS = 5  # tool rounds per question (search, classify, then approvals) before the model must reply
CHUNK_FOLDER = PROJECT_ROOT / "KnowledgeBase" / "chunks"

# The steps the agent must complete for every question, in order.
WORKFLOW_STEPS = ("Retrieve", "Classify", "Synthesize")


# ---- state ----

def _merge(old: dict, new: dict) -> dict:
    return {**old, **new}


def _last(old: Any, new: Any) -> Any:
    return new


def _unique(old: list, new: list) -> list:
    return list(dict.fromkeys([*old, *new]))  # keeps order, drops repeats


class AdvisorState(MessagesState):
    """The conversation and the workflow state. Fields with a reducer accept updates from parallel tool calls."""

    workflow: Annotated[dict[str, bool], _merge]  # step -> done, for the current question
    retrieved: Annotated[list[str], _unique]  # file names of the chunks the re-ranker kept, for the current question
    classification: Annotated[str, _last]  # the last accepted classification of the current question
    tools_called: Annotated[list[str], operator.add]  # for the current question
    approval_requests: Annotated[list[str], operator.add]  # texts of the requests created for the current question
    approval_ids: Annotated[list[int], operator.add]  # requests made in this conversation: the only ones the student can see or change


def new_workflow() -> dict[str, bool]:
    return {step: False for step in WORKFLOW_STEPS}


@dataclass
class AdvisorContext:
    """What a run of the graph uses, passed as `context=` to invoke / stream."""

    llm: BaseChatModel  # the advisor's model
    retriever: BaseRetriever  # returns chunk Documents with metadata["filename"]
    reranker: BaseDocumentCompressor  # keeps the relevant Documents, most relevant first
    approvals: Callable[[], Any] = ApprovalContext  # returns a store with create / getall / update / delete / close
    memory: tuple[date, str] | None = None  # (last_update, memory text) of the agent memory; goes at the tail of the prompt


def advisor_context(llm: BaseChatModel, approvals: Callable[[], Any] = ApprovalContext,
                    memory: tuple[date, str] | None = None) -> AdvisorContext:
    """The context the app uses: pgvector + Ollama embeddings for retrieval, Azure OpenAI as the re-ranker."""
    from LLM import OllamaEmbedder, azure_openai
    from RetAuGen.retrieval_pipeline import ChunkRetriever, LLMReranker

    return AdvisorContext(
        llm=llm,
        retriever=ChunkRetriever(embeddings=OllamaEmbedder(), chunk_folder=CHUNK_FOLDER, k=TOP_N),
        reranker=LLMReranker(llm=azure_openai(temperature=0)),
        approvals=approvals,
        memory=memory,
    )


# ---- approvals kept in memory (test runs) ----

def approval_status(a: Approval) -> str:
    return "approved" if a.approved else "rejected" if a.rejected else "pending"


class InMemoryApprovals:
    """Approval requests kept in memory, so a test run writes nothing to the database.

    Use one instance for the whole run: advisor_context(llm, approvals=lambda: store).
    """

    def __init__(self):
        self.items = []
        self._ids = itertools.count(1)

    def create(self, request):
        self.items.append(Approval(next(self._ids), datetime.date.today(), request, False, False))
        return self.items[-1].id

    def getall(self):
        return list(self.items)

    def update(self, id, approved=None, rejected=None, request=None):
        for a in self.items:
            if a.id == id:
                a.request = request if request is not None else a.request
        return True

    def delete(self, id):
        self.items = [a for a in self.items if a.id != id]
        return True

    def close(self):
        pass


# ---- prompt ----

def workflow_text(workflow: dict[str, bool]) -> str:
    """Serialize the workflow state as text, e.g. 'Retrieve: Done'."""
    return "\n".join(f"{step}: {'Done' if workflow.get(step) else 'Not Done'}" for step in WORKFLOW_STEPS)


def system_prompt(workflow: dict[str, bool] | None = None, memory: tuple[date, str] | None = None) -> str:
    """memory: (last_update, memory text) of the agent memory, or None; it goes at the tail of the prompt."""
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    memory_section = f"\n         Agent memory (last updated {memory[0]}):\n         {memory[1]}\n" if memory else ""
    return f"""
         You are an Academic Advisor. The users are students seeking academic advice.
         Today's date is {today}.
         Workflow state for the current question (complete the steps in order; a step is Done only
         after its tool has been called successfully):
{workflow_text(workflow or new_workflow())}
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
              a human advisor should handle it; this sends an approval request to the advisor.
            - approval: the student asks to see, change or cancel an approval request made earlier.
            - dismiss: outside academic advising altogether (e.g. shopping, weather, coding help,
              general trivia); not something to escalate.
         3. Synthesize: follow the classification result. When answering, use only the retrieved documents,
            be concise, and name the procedure or section you used. Never invent
            requirements, fees, offices or deadlines.
         Approval requests (manage_approvals, available once the question is classified): create one when you
         escalate; list, update or delete earlier ones only when the student asks. Never say a request was
         created, changed or deleted unless manage_approvals confirmed it.
{memory_section}         """


# ---- tools ----

Runtime_ = ToolRuntime[AdvisorContext, AdvisorState]


def _result(runtime: Runtime_, tool_name: str, content: str, **updates) -> Command:
    """The tool's reply to the model, plus the state updates it makes."""
    message = ToolMessage(content, tool_call_id=runtime.tool_call_id, name=tool_name)
    return Command(update={"messages": [message], "tools_called": [tool_name], **updates})


@tool(parse_docstring=True)
def search_documents(intent: str, runtime: Runtime_) -> Command:
    """Search the university's documents. Pass the intent behind the student's question: what they actually want to know, as a short standalone statement, paraphase the sentence to be optimized for vector embedding semantic similarity search. Returns the relevant document sections, or says that no relevant information was found.

    Args:
        intent: 'requirements and procedure to apply for enrollment'
    """
    intent = intent.strip()
    ctx = runtime.context
    runtime.stream_writer({"progress": f"Searching documents: **{intent}**"})
    documents = ctx.retriever.invoke(intent)
    relevant = ctx.reranker.compress_documents(documents, intent)
    runtime.stream_writer({"progress": f"Relevant chunks found: **{len(relevant)}**"})
    content = ("\n\n".join(f"[Document {i}]\n{doc.page_content}" for i, doc in enumerate(relevant, start=1))
               or "No relevant document or information was found.")
    return _result(runtime, "search_documents", content, workflow={"Retrieve": True},
                   retrieved=[doc.metadata["filename"] for doc in relevant])


CLASSIFY_REPLIES = {
    "answer": "answer: reply to the student's question using only the retrieved documents.",
    "escalate": ("escalate: do NOT answer the question. First call manage_approvals with action 'create' and a request "
                 "that summarizes the student's question and what the human advisor needs to decide. Then tell the "
                 "student it is outside the documents and has been escalated to a human advisor. Say it was "
                 "escalated only if manage_approvals confirmed the request was created."),
    "approval": ("approval: the student wants to see, change or cancel an approval request made earlier. Call "
                 "manage_approvals with action 'list' to find the request, then 'update' (with the new request text) "
                 "or 'delete'. Only pending requests can be changed. Then tell the student the outcome."),
    "dismiss": ("dismiss: do NOT answer the question and do NOT escalate it. Politely tell the student that you "
                "can only help with academic matters at the university."),
}


@tool
def classify_question(classification: Literal["answer", "escalate", "approval", "dismiss"], runtime: Runtime_) -> Command:
    """Classify the student's question after searching. 'answer': the retrieved documents contain the information needed. 'escalate': the question is an academic matter at the university (enrollment, courses, grades, programs, applications, student records and the like), but the documents do not cover it. 'approval': the student asks to see, change or cancel an approval request they made earlier. 'dismiss': there is no evidence that the question is relevant to any academic matter, event and announcement related to the university, or any subject that a human advisor would not handle either."""
    runtime.stream_writer({"progress": f"Classification: **{classification}**"})
    if not runtime.state.get("workflow", {}).get("Retrieve"):
        return _result(runtime, "classify_question",
                       "Classify is not allowed yet: Retrieve is Not Done. Call search_documents first.")
    return _result(runtime, "classify_question", CLASSIFY_REPLIES[classification],
                   classification=classification, workflow={"Classify": True})


@tool(parse_docstring=True)
def manage_approvals(action: Literal["create", "list", "update", "delete"], runtime: Runtime_,
                     id: int | None = None, request: str | None = None) -> Command:
    """Manage approval requests that a human advisor will review. 'create': send a new request (needs 'request'); do this when a question is escalated. 'list': show the requests made in this conversation with their ids and status. 'update': change the text of a pending request (needs 'id' and 'request'). 'delete': cancel a pending request (needs 'id'). Requests that are already approved or rejected cannot be changed.

    Args:
        action: What to do.
        id: The request id, for update and delete.
        request: What the human advisor is asked to review or approve, as a short standalone statement.
    """
    # A student only sees and changes requests made in this conversation (there are no student accounts yet), and only
    # while they are still pending.
    runtime.stream_writer({"progress": f"Approvals: **{action}**"})
    request = (request or "").strip()
    mine = runtime.state.get("approval_ids", [])

    def reply(content, **updates):
        return _result(runtime, "manage_approvals", content, **updates)

    try:
        db = runtime.context.approvals()
        try:
            if action == "create":
                if not request:
                    return reply("A request text is required. Call manage_approvals again with 'request'.")
                new_id = db.create(request)
                return reply(f"Approval request #{new_id} was created. A human will review it.",
                             approval_ids=[new_id], approval_requests=[request])
            records = {a.id: a for a in db.getall() if a.id in mine}
            if action == "list":
                if not records:
                    return reply("No approval requests were made in this conversation.")
                return reply("\n".join(f"#{a.id} ({a.date_requested}, {approval_status(a)}): {a.request}" for a in records.values()))
            if id is None:
                return reply("An integer 'id' is required. Call manage_approvals with action 'list' to find it.")
            a = records.get(id)
            if a is None:
                return reply(f"There is no approval request #{id} in this conversation. Call action 'list' to see them.")
            if a.approved or a.rejected:
                return reply(f"Approval request #{id} is already {approval_status(a)} and can no longer be changed.")
            if action == "delete":
                db.delete(id)
                return reply(f"Approval request #{id} was deleted.")
            if not request:
                return reply("A request text is required to update. Call manage_approvals again with 'request'.")
            db.update(id, request=request)
            return reply(f"Approval request #{id} was updated.")
        finally:
            db.close()
    except Exception as e:  # a db problem must not crash the chat
        return reply(f"Approval requests are unavailable right now: {e}")


SEARCH_TOOLS = [search_documents, classify_question]
# Offered only after the question is classified, so approvals never skip the retrieve / classify steps
APPROVAL_TOOLS = [manage_approvals]


# ---- nodes ----

def start_question(state: AdvisorState) -> dict:
    """Reset the workflow and the facts collected for the new student question (the conversation is kept)."""
    return {
        "workflow": new_workflow(),
        "retrieved": Overwrite([]),
        "classification": "",
        "tools_called": Overwrite([]),
        "approval_requests": Overwrite([]),
    }


def tool_rounds(messages: list[AnyMessage]) -> int:
    """Tool-calling replies of the model since the student's last message."""
    rounds = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            rounds += 1
    return rounds


def answer_dangling_tool_calls(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The messages with a placeholder result after every tool call that never got one (a run that failed half-way),
    since chat APIs reject a tool call without a result."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    out = []
    for message in messages:
        out.append(message)
        if isinstance(message, AIMessage):
            out.extend(ToolMessage("The tool did not run.", tool_call_id=call["id"], name=call["name"])
                       for call in message.tool_calls if call["id"] not in answered)
    return out


def agent(state: AdvisorState, runtime: Runtime[AdvisorContext]) -> dict:
    """Call the model with the prompt that carries the latest workflow state. Before classifying it gets the search /
    classify tools; once classified it gets only manage_approvals and may reply any time instead of calling it. After
    MAX_TOOL_ROUNDS it gets no tools, so a model that keeps calling tools cannot loop forever."""
    ctx = runtime.context
    workflow = state.get("workflow") or new_workflow()
    tools = APPROVAL_TOOLS if workflow.get("Classify") else SEARCH_TOOLS
    llm = ctx.llm.bind_tools(tools) if tool_rounds(state["messages"]) < MAX_TOOL_ROUNDS else ctx.llm
    response = llm.invoke([SystemMessage(system_prompt(workflow, ctx.memory)), *answer_dangling_tool_calls(state["messages"])])
    update: dict = {"messages": [response]}
    if not response.tool_calls:
        update["workflow"] = {"Synthesize": True}
    return update


def build_advisor(checkpointer=None):
    """The compiled advisor graph. Pass a checkpointer (e.g. InMemorySaver()) to keep the conversation per thread_id."""
    graph = StateGraph(AdvisorState, context_schema=AdvisorContext)
    graph.add_node("start_question", start_question)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(SEARCH_TOOLS + APPROVAL_TOOLS))
    graph.add_edge(START, "start_question")
    graph.add_edge("start_question", "agent")
    graph.add_conditional_edges("agent", tools_condition, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def last_reply(state: dict) -> str:
    """The text of the model's last reply in the state, or ''."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message.text
    return ""

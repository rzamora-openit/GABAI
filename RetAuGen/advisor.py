"""The GABAI advisor agent: system prompt, tools and the workflow state of the current question.

Shared by the chat page (GABAI.py) and the test-runner page, so a test run exercises the same agent the students talk to.
This module has no Streamlit code: the page passes `notify` (a progress line, shown as a caption) and `on_change`
(called when the workflow state changes) to draw the UI.
"""
import datetime
import itertools

from LLM import AzureOpenAI, OllamaEmbedder
from RetAuGen.retrieval_pipeline import rerank_agent, retrieve_chunks
from Utility.approval_context import Approval, ApprovalContext
from Utility.vector_db_context import VectorDbContext

TOP_N = 5  # chunks fetched from the vector db before re-ranking
MAX_TOOL_ROUNDS = 5  # tool rounds per question (search, classify, then approvals) before the model must reply
CHUNK_FOLDER = "KnowledgeBase/chunks"

# The steps the agent must complete for every question, in order.
WORKFLOW_STEPS = ("Retrieve", "Classify", "Synthesize")

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
                "'approval': the student asks to see, change or cancel an approval request they made earlier. "
                "'dismiss': there is no evidence that the question is relevant to any academic matter, event and announcement related to the university, or any subject that a human advisor would not handle either."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {"type": "string", "enum": ["answer", "escalate", "approval", "dismiss"]}
                },
                "required": ["classification"]
            }
        }
    }
]

# Offered only after the question is classified, so approvals never skip the retrieve / classify steps
approval_tools = [
    {
        "type": "function",
        "function": {
            "name": "manage_approvals",
            "description": (
                "Manage approval requests that a human advisor will review. "
                "'create': send a new request (needs 'request'); do this when a question is escalated. "
                "'list': show the requests made in this conversation with their ids and status. "
                "'update': change the text of a pending request (needs 'id' and 'request'). "
                "'delete': cancel a pending request (needs 'id'). "
                "Requests that are already approved or rejected cannot be changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "list", "update", "delete"]},
                    "id": {"type": "integer", "description": "The request id, for update and delete."},
                    "request": {"type": "string", "description": "What the human advisor is asked to review or approve, as a short standalone statement."},
                },
                "required": ["action"]
            }
        }
    }
]


def approval_status(a):
    return "approved" if a.approved else "rejected" if a.rejected else "pending"


class InMemoryApprovals:
    """Approval requests kept in memory, so a test run writes nothing to the database.

    Use one instance for the whole run: Advisor(approvals=lambda: store).
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


class Advisor:
    def __init__(self, approvals=ApprovalContext, notify=None, on_change=None):
        """approvals: called to get an approvals context (create / getall / update / delete / close); the database by default.
        notify: called with a short progress line, e.g. 'Classification: **answer**'.
        on_change: called when the workflow state changes.
        """
        self.approvals = approvals
        self.notify = notify or (lambda text: None)
        self.on_change = on_change or (lambda: None)
        self.approval_ids = []  # ids of the approval requests made in this conversation: the only ones the student can list, change or delete
        self.start_question()

    def start_question(self):
        """Reset the workflow and the facts collected for a new student question."""
        self.workflow = {step: False for step in WORKFLOW_STEPS}
        self.retrieved = []  # file names of the chunks the re-ranker kept, in order, without repeats
        self.classification = ""  # the last accepted classification
        self.tools_called = []
        self.approval_requests = []  # texts of the approval requests created for this question
        self.on_change()

    def workflow_text(self):
        """Serialize the workflow state as text, e.g. 'Retrieve: Done'."""
        return "\n".join(f"{step}: {'Done' if self.workflow[step] else 'Not Done'}" for step in WORKFLOW_STEPS)

    def mark_done(self, step):
        self.workflow[step] = True
        self.on_change()

    def available_tools(self):
        """Before classifying the model gets the search / classify tools; once classified it gets only manage_approvals."""
        return approval_tools if self.workflow["Classify"] else tools

    def system_prompt(self, memory=None):
        """memory: (last_update, memory text) of the agent memory, or None; it goes at the tail of the prompt."""
        today = datetime.date.today().strftime("%A, %B %d, %Y")
        memory_section = f"\n         Agent memory (last updated {memory[0]}):\n         {memory[1]}\n" if memory else ""
        return f"""
         You are an Academic Advisor. The users are students seeking academic advice.
         Today's date is {today}.
         Workflow state for the current question (complete the steps in order; a step is Done only
         after its tool has been called successfully):
{self.workflow_text()}
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

    def process_tool(self, params, tool_name):
        self.tools_called.append(tool_name)
        if tool_name == "search_documents":
            return self.search_documents(params)
        if tool_name == "classify_question":
            return self.classify_question(params)
        if tool_name == "manage_approvals":
            return self.manage_approvals(params)
        return f"Tool {tool_name} not found."

    def search_documents(self, params):
        intent = str(params.get("intent", "")).strip()
        self.notify(f"Searching documents: **{intent}**")
        db = VectorDbContext()
        try:
            chunks = retrieve_chunks(intent, TOP_N, CHUNK_FOLDER, OllamaEmbedder(), db)
        finally:
            db.close()
        relevant = rerank_agent(intent, [text for _, text in chunks], AzureOpenAI(options={"temperature": 0}))
        names = {text: name for name, text in chunks}
        self.retrieved = list(dict.fromkeys([*self.retrieved, *(names[text] for text in relevant if text in names)]))
        self.mark_done("Retrieve")
        self.notify(f"Relevant chunks found: **{len(relevant)}**")
        if not relevant:
            return "No relevant document or information was found."
        return "\n\n".join(f"[Document {i}]\n{text}" for i, text in enumerate(relevant, start=1))

    def classify_question(self, params):
        classification = str(params.get("classification", "")).strip().lower()
        self.notify(f"Classification: **{classification}**")
        if not self.workflow["Retrieve"]:
            return "Classify is not allowed yet: Retrieve is Not Done. Call search_documents first."
        if classification in ("answer", "escalate", "approval", "dismiss"):
            self.classification = classification
            self.mark_done("Classify")
        if classification == "answer":
            return "answer: reply to the student's question using only the retrieved documents."
        if classification == "escalate":
            return ("escalate: do NOT answer the question. First call manage_approvals with action 'create' and a request "
                    "that summarizes the student's question and what the human advisor needs to decide. Then tell the "
                    "student it is outside the documents and has been escalated to a human advisor. Say it was "
                    "escalated only if manage_approvals confirmed the request was created.")
        if classification == "approval":
            return ("approval: the student wants to see, change or cancel an approval request made earlier. Call "
                    "manage_approvals with action 'list' to find the request, then 'update' (with the new request text) "
                    "or 'delete'. Only pending requests can be changed. Then tell the student the outcome.")
        if classification == "dismiss":
            return ("dismiss: do NOT answer the question and do NOT escalate it. Politely tell the student that you "
                    "can only help with academic matters at the university.")
        return (f"Invalid classification '{classification}'. "
                "Call classify_question again with 'answer', 'escalate', 'approval' or 'dismiss'.")

    def manage_approvals(self, params):
        """Create, list, update or delete approval requests. A student only sees and changes requests made in this
        conversation (there are no student accounts yet), and only while they are still pending."""
        action = str(params.get("action", "")).strip().lower()
        request = str(params.get("request", "")).strip()
        self.notify(f"Approvals: **{action}**")
        mine = self.approval_ids
        try:
            db = self.approvals()
            try:
                if action == "create":
                    if not request:
                        return "A request text is required. Call manage_approvals again with 'request'."
                    mine.append(db.create(request))
                    self.approval_requests.append(request)
                    return f"Approval request #{mine[-1]} was created. A human will review it."
                records = {a.id: a for a in db.getall() if a.id in mine}
                if action == "list":
                    if not records:
                        return "No approval requests were made in this conversation."
                    return "\n".join(f"#{a.id} ({a.date_requested}, {approval_status(a)}): {a.request}" for a in records.values())
                if action in ("update", "delete"):
                    try:
                        id = int(params.get("id"))
                    except (TypeError, ValueError):
                        return "An integer 'id' is required. Call manage_approvals with action 'list' to find it."
                    a = records.get(id)
                    if a is None:
                        return f"There is no approval request #{id} in this conversation. Call action 'list' to see them."
                    if a.approved or a.rejected:
                        return f"Approval request #{id} is already {approval_status(a)} and can no longer be changed."
                    if action == "delete":
                        db.delete(id)
                        mine.remove(id)
                        return f"Approval request #{id} was deleted."
                    if not request:
                        return "A request text is required to update. Call manage_approvals again with 'request'."
                    db.update(id, request=request)
                    return f"Approval request #{id} was updated."
                return f"Invalid action '{action}'. Use 'create', 'list', 'update' or 'delete'."
            finally:
                db.close()
        except Exception as e:  # a db problem must not crash the chat
            return f"Approval requests are unavailable right now: {e}"

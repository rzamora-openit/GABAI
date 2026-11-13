"""Memory agent: an independent observer that runs after the main agent has answered. It sees the last user message
and the final response and decides whether anything is worth remembering."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from Utility.agent_memory_context import AgentMemoryContext

MEMORY_PROMPT = """You are the memory of an academic advising assistant. You observe one exchange between a student and the assistant.
Decide if it holds information worth remembering for future conversations with this student:
facts about the student (program, major, standing, goals), their stated preferences or constraints,
and open matters such as escalated questions or pending requirements.
Do not remember what the university documents already say, small talk, or questions that were dismissed.
The memory is one short text. If you save, write the full updated memory: keep what is still important from the
current memory, add the new information, and drop what is outdated.
Do not think or explain. If there is nothing to remember, leave memory empty (null) so the current memory stays as it is."""


class MemoryDecision(BaseModel):
    """Save the full updated memory, replacing the current one, or leave it as it is."""

    memory: str | None = Field(
        default=None,
        description="The complete updated memory text; null when there is nothing important to remember.",
    )


def memory_agent(user_message: str, response: str, memory_db: AgentMemoryContext, llm: BaseChatModel) -> bool:
    """Save to the agent memory if the exchange holds something important. Returns True if the memory was saved.

    The memory is a single text, so a save replaces it; the agent sees the current memory and writes the full new one.
    llm: the model to use, e.g. azure_openai(temperature=0). The structured output forces it to answer with a tool call.
    """
    current = memory_db.get()
    label = f"Current memory (last updated {current[0]})" if current else "Current memory"
    current_text = current[1] if current else "(empty)"
    content = (f"{label}:\n{current_text}\n\n"
               f"Last user message:\n{user_message}\n\n"
               f"Assistant's final response:\n{response}")
    decision = llm.with_structured_output(MemoryDecision, method="function_calling").invoke(
        [SystemMessage(MEMORY_PROMPT), HumanMessage(content)]
    )
    memory = (decision.memory or "").strip() if decision else ""
    if not memory:
        return False  # nothing to remember, or an empty save
    memory_db.save(memory)
    return True

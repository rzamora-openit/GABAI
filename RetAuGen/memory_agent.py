"""Memory agent: an independent observer that runs after the main agent has answered. It sees the last user message
and the final response and decides whether anything is worth remembering."""
from LLM import ChatProvider, ToolCall
from Utility.agent_memory_context import AgentMemoryContext


def memory_agent(user_message: str, response: str, memory_db: AgentMemoryContext, llm: ChatProvider) -> bool:
    """Save to the agent memory if the exchange holds something important. Returns True if the memory was saved.

    The memory is a single text, so a save replaces it; the agent sees the current memory and writes the full new one.
    llm: the model to use. It must call a tool straight away, with no thinking, e.g.
        AzureOpenAI(options={"temperature": 0, "tool_choice": "required"})
    (AzureOpenAI already turns reasoning off whenever tools are sent.)
    """

    system_prompt = """You are the memory of an academic advising assistant. You observe one exchange between a student and the assistant.
Decide if it holds information worth remembering for future conversations with this student:
facts about the student (program, major, standing, goals), their stated preferences or constraints,
and open matters such as escalated questions or pending requirements.
Do not remember what the university documents already say, small talk, or questions that were dismissed.
The memory is one short text. If you save, write the full updated memory: keep what is still important from the
current memory, add the new information, and drop what is outdated.
Do not think or explain. Immediately call exactly one tool: save_memory, or conclude if there is nothing to remember."""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "save_memory",
                "description": "Save the full updated memory, replacing the current one.",
                "parameters": {
                    "type": "object",
                    "properties": {"memory": {"type": "string", "description": "The complete memory text."}},
                    "required": ["memory"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "conclude",
                "description": "Nothing important to remember; leave the memory as it is.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    current = memory_db.get()
    label = f"Current memory (last updated {current[0]})" if current else "Current memory"
    current_text = current[1] if current else "(empty)"
    content = (f"{label}:\n{current_text}\n\n"
               f"Last user message:\n{user_message}\n\n"
               f"Assistant's final response:\n{response}")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    for event in llm.stream(messages, tools):
        if isinstance(event, ToolCall):
            if event.name == "save_memory" and str(event.arguments.get("memory", "")).strip():
                memory_db.save(event.arguments["memory"].strip())
                return True
            return False  # conclude, or an empty save
    return False

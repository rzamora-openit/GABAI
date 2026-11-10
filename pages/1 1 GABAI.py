from dataclasses import asdict

import streamlit as st

from LLM import PROVIDERS, ContentDelta, ThinkingDelta, ToolCall, get_provider

st.set_page_config(page_title="GABAI", layout="wide")

st.title("Assistant Conversation")

# Pick the LLM; every provider streams the same Thinking / Content / ToolCall events
model_name = st.sidebar.selectbox("Model", list(PROVIDERS))

def process_tool(params, tool_name):
    if tool_name == "classify_question":
        return classify_question(params)
    return f"Tool {tool_name} not found."

def classify_question(params):
    classification = str(params.get("classification", "")).strip().lower()
    st.caption(f"Classification: **{classification}**")
    if classification == "answer":
        return "answer: reply to the student's question using only the guidelines."
    if classification == "escalate":
        return ("escalate: do NOT answer the question. Tell the student it is outside the guidelines "
                "and has been escalated to a human advisor.")
    return f"Invalid classification '{classification}'. Call classify_question again with 'answer' or 'escalate'."

tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_question",
            "description": (
                "Classify the student's question. Use 'answer' if the question can be answered by the "
                "guidelines or is within the scope of the guidelines. Use 'escalate' if it cannot be "
                "answered by the guidelines and is outside their scope."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {"type": "string", "enum": ["answer", "escalate"]}
                },
                "required": ["classification"]
            }
        }
    }
]


# We will store the conversation history in Streamlit's session state, which allows us to keep track of messages across user interactions.
if "messages" not in st.session_state:
    st.session_state.messages = [
        # Also, initialize the system prompt
        {"role": "system", "content": 
         """
         You will act as a Academic Advisor, you will refer to the guidelines, 
         and when the guidelines cannot accomodate the situation, you will stop.
         The users are students seeking academic advice.
         For every student question, first call the classify_question tool, then follow its result.

        ## Guidelines

        Program Goals
        
        The program aims to develop professional nurses who are able to:
        ●	Apply their understanding of nursing theories, research principles, methodologies to current problems and issues affecting the nursing profession and the health care delivery system;
        ●	Gain insight into self and others as fully functioning individuals;
        ●	Broaden their understanding of a related discipline that should serve as a further base for specialization in nursing;
        ●	Develop innovative measures in the delivery of health care services; and
        ●	Help in developing independent nursing practice.
        Admission Requirements
        ●	Applicant must be a holder of bachelor’s degree in nursing, i.e. BS Nursing.
        ●	An application form must be duly accomplished and submitted to the Office of the University Registrar, together with the following documents:
        ●	Original copy of the official transcript of academic records stating degree obtained, date of graduation, and Special Order from the Bureau of Private Schools (S.O.B. No).
        ●	Two letters of reference from former professors or supervisors, using UPOU standard format.
        ●	Official receipt of the application fee paid at any Philippine National Bank (PNB) branch.
        Admission Policies
        Applicants are screened by the Admission committee headed by the Program chair, (i.e. either Adult Health Nursing, Maternal Child Nursing, Nursing Administration or Geriatrics and Gerontology) whichever major program the applicant indicated in his/her application form.
        Criteria for screening of applicants are as follows:
        ●	For initial screening, GWA of 2.125 is used as cut-off grade.
        ●	Applicants who have a GWA of 2.125 or better and who possess at least one year clinical and nursing-related experience are admitted to the program as degree students
        ●	Students whose GWA are on the borderlines are further evaluated through the composite scoring system (see box below).
        ●	The composite score consists of: equivalent score for GWA, length of experience, and nature of experience.
        ●	Applicants who have a composite score of 2.0 or better are admitted as degree students
        ●	Applicants who were not able to satisfy the GWA cut-off point or the composite score are admitted as probationary, for evaluation after 12 units.

         """}
    ]

# Iterate through the conversation history
for msg in st.session_state.messages:
    if msg["role"] in ("system", "tool"):
        continue  # sent to the model but not shown in the UI
    # display each message in the appropriate format based on its role (user, assistant)
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("Thinking"): st.markdown(msg["thinking"])
        for tc in msg.get("tool_calls", []):
            st.caption(f"Tool call: `{tc['name']}({tc['arguments']})`")
        if msg["content"]:
            st.markdown(msg["content"])

# this is where the user will type their message. st.chat_input() is a built-in Streamlit component that creates a text input and automatically scrolls to the bottom of the page when new messages are added.
prompt = st.chat_input("Type your message")


def generate(use_tools=True):
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        placeholder = st.empty()

    complete_response = ""
    complete_thinking = ""
    thinking_body = None
    tool_calls: list[ToolCall] = []

    try:
        provider = get_provider(model_name)
        for event in provider.stream(st.session_state.messages, tools if use_tools else None):
            if isinstance(event, ThinkingDelta):
                if thinking_body is None:
                    # first thinking chunk: open the "Thinking" expander and stream into it
                    thinking_body = thinking_placeholder.expander("Thinking", expanded=True).empty()
                complete_thinking += event.text
                thinking_body.markdown(complete_thinking + "▌")
            elif isinstance(event, ContentDelta):
                complete_response += event.text
                placeholder.markdown(complete_response + "▌")
            elif isinstance(event, ToolCall):
                tool_calls.append(event)
    except Exception as e:
        placeholder.error(f"{model_name}: {e}")
        return False

    if complete_thinking:
        # thinking is done: collapse the expander
        thinking_placeholder.expander("Thinking", expanded=False).markdown(complete_thinking)

    if tool_calls:
        # Record the model's tool calls, then each tool's result, so any provider can continue the conversation
        st.session_state.messages.append({
            "role": "assistant",
            "content": complete_response,
            "thinking": complete_thinking,
            "tool_calls": [asdict(tc) for tc in tool_calls],
        })
        for tc in tool_calls:
            st.session_state.messages.append({
                "role": "tool",
                "name": tc.name,
                "tool_call_id": tc.id,
                "content": process_tool(tc.arguments, tc.name),
            })
        # Tools are turned off for this pass so the model must write the reply instead of classifying again
        return generate(use_tools=False)
    else:
        placeholder.markdown(complete_response)
        st.session_state.messages.append({"role": "assistant", "content": complete_response, "thinking": complete_thinking})
        return True

# run this logic every time the user submits a new message
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    if generate():  # on error, skip the redraw so the error message stays visible
        # Redraw the page from the saved history. The live-streamed bubbles (and the classification
        # caption) don't line up with the history layout, so without this, leftovers from this run
        # show up as "ghost" text on the next prompt until the new response finishes.
        st.rerun()
    
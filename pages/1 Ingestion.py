from pathlib import Path

import streamlit as st

from LLM import OllamaEmbedder, azure_openai
from RetAuGen.ingestion_pipeline import ingest
from Utility.agent_memory_context import AgentMemoryContext
from Utility.vector_db_context import VectorDbContext

if st.button("Run ingestion pipeline"):
    with st.spinner("Ingesting..."):
        db = VectorDbContext()
        try:
            sources = ingest("KnowledgeBase", "KnowledgeBase/extracted", "KnowledgeBase/chunks", azure_openai(temperature=0), OllamaEmbedder(), db)
        finally:
            db.close()
    st.success(f"Done: {len(sources)} PDF(s) checked.")

st.space()


if st.button("Reset db"):
    # Deletes the files the pipeline generated (not the PDFs) and every row in the vector table.
    for file in [*Path("KnowledgeBase/extracted").glob("*.txt"), *Path("KnowledgeBase/chunks").glob("*.chunk_*.md")]:
        file.unlink()
    db = VectorDbContext()
    try:
        db.delete_all()
    finally:
        db.close()
    st.success("Reset done.")

st.space()


if st.button("Clear memory"):
    # Deletes the agent memory row.
    memory_db = AgentMemoryContext()
    try:
        memory_db.delete()
    finally:
        memory_db.close()
    st.session_state.agent_memory = None  # the chat page keeps a copy for the main agent's prompt (session state is shared across pages)
    st.success("Memory cleared.")

from pathlib import Path

import streamlit as st

from LLM import AzureOpenAI, OllamaEmbedder
from RetAuGen.ingestion_pipeline import ingest
from Utility.vector_db_context import VectorDbContext

if st.button("Run ingestion pipeline"):
    with st.spinner("Ingesting..."):
        db = VectorDbContext()
        try:
            ingest("KnowledgeBase", "KnowledgeBase/extracted", "KnowledgeBase/chunks", AzureOpenAI(options={"temperature": 0}), OllamaEmbedder(), db)
        finally:
            db.close()
    st.success("Done.")

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

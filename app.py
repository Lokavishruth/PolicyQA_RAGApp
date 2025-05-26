# generative_policy_qa_app.py
# Streamlit-based Generative Search System for Policy Documents using LlamaIndex
# Author: [Your Name]
# Description: Upload any PDF, build a vector index, and answer questions using LlamaIndex and OpenAI LLM.

import streamlit as st
st.set_page_config(page_title="PolicyDoc QA - LlamaIndex", layout="wide")

import tempfile
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import Settings
from dotenv import load_dotenv

# ---------------------------
# CONFIGURATION
# ---------------------------
# Load environment variables from .env file for API keys and configs
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# ---------------------------
# SYSTEM DESIGN OVERVIEW
# ---------------------------
"""
System Workflow:
1. User uploads a PDF.
2. PDF is parsed and split into text chunks.
3. Chunks are embedded and indexed in a vector store.
4. User asks a question.
5. System retrieves relevant chunks and generates an answer using an LLM.
"""

# ---------------------------
# UTILITY FUNCTIONS (LlamaIndex-based)
# ---------------------------
def process_pdf_and_build_index(pdf_path, embed_model_name="local:sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads a PDF, splits into chunks, and builds a LlamaIndex vector index.
    Args:
        pdf_path (str): Path to the PDF file.
        embed_model_name (str): Embedding model to use.
    Returns:
        index (VectorStoreIndex): LlamaIndex vector index for the document.
    """
    # Read PDF and extract text
    reader = SimpleDirectoryReader(input_files=[pdf_path])
    documents = reader.load_data()

    # Set up embedding model and LLM using the latest Settings context
    embed_model = resolve_embed_model(embed_model_name)
    llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser
    # Build vector index
    index = VectorStoreIndex.from_documents(documents)
    return index

def answer_question(index, query):
    """
    Uses the vector index to retrieve relevant chunks and generate a response from the LLM.
    Args:
        index (VectorStoreIndex): The vector index.
        query (str): User's question.
    Returns:
        str: LLM-generated answer.
    """
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer()
    )
    response = query_engine.query(query)
    return str(response)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("PolicyDoc QA - Generative Search for Policy Documents")

st.markdown("""
Upload any policy PDF, ask questions, and get answers powered by LlamaIndex and OpenAI!
""")

# Upload PDF section
pdf_file = st.file_uploader("Upload any Policy PDF", type=["pdf"])

if pdf_file:
    # Save uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(pdf_file.read())
        pdf_path = tmpfile.name
    st.info("Building vector index for the uploaded PDF...")
    try:
        index = process_pdf_and_build_index(pdf_path)
        st.success("PDF indexed successfully! You can now ask questions.")
        # User query section
        query = st.text_input("Ask a question about the uploaded document:")
        if query:
            answer = answer_question(index, query)
            st.write("### Answer:")
            st.success(answer)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

    st.markdown("---")
    st.markdown("#### Documentation")
    st.markdown("""
    **Project Goals:** Build a robust generative search system for policy documents using LlamaIndex.  
    **Data Source:** User-uploaded PDF.  
    **Design Choices:** LlamaIndex for ingestion, chunking, embedding, and retrieval; OpenAI LLM for answer generation.  
    **Challenges:** Handling large PDFs, efficient chunking, accurate retrieval, and prompt engineering.  
    **How to Run:** Upload any PDF, wait for indexing, then ask questions in the input box above.
    """)

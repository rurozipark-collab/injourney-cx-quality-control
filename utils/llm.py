"""
LLM Helper for Knowledge Base Chat

Supports OpenAI and Ollama.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Lazy imports to avoid heavy startup cost
_llm = None


def get_llm():
    """Get the configured LLM (cached)."""
    global _llm
    if _llm is not None:
        return _llm
    
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        
        _llm = ChatOpenAI(
            model=model,
            temperature=0.3,
            openai_api_key=api_key,
        )
    else:
        # Ollama
        from langchain_community.chat_models import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        
        _llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.3,
        )
    
    return _llm


SYSTEM_PROMPT = """You are an expert CX auditor assistant for InJourney Airports.

You have deep knowledge of the official "InJourney Airports Customer Experience Transformation Concept" and all 5 CX Playbooks.

Rules:
- Answer ONLY using information from the provided context.
- If the answer is not in the context, say "Informasi tersebut tidak ditemukan di dalam Playbook CX InJourney."
- Be precise, professional, and cite the source document + page when possible.
- Use Indonesian language when the user asks in Indonesian.
- Structure long answers with bullet points or numbered lists.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Context from Playbooks:
{context}

Question: {question}

Answer in a clear and professional manner:""")
])


def format_docs(docs: List) -> str:
    """Format retrieved documents into a readable context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
        formatted.append(f"[{src} - Halaman {page}]\n{text}")
    return "\n\n---\n\n".join(formatted)


def ask_playbook_question(question: str, retrieved_docs: List) -> str:
    """Ask a question using RAG + LLM."""
    if not retrieved_docs:
        return "Maaf, saya tidak menemukan informasi yang relevan di dalam playbook."
    
    context = format_docs(retrieved_docs)
    llm = get_llm()
    
    chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)

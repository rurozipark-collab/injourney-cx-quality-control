"""
RAG Engine for InJourney Airports CX Quality Control

LangChain + ChromaDB powered retrieval system over the 6 CX Playbooks.
Supports both local embeddings (offline) and OpenAI embeddings.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# PDF extraction
from .pdf_processor import process_all_playbooks, get_playbook_stats

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

import os
from pathlib import Path

# Support custom data directory for cloud deployments (e.g. mounted disk)
_base = os.getenv("DATA_DIR", str(Path(__file__).parent.parent))
CHROMA_PERSIST_DIR = Path(_base) / "chroma_db"
CHROMA_COLLECTION_NAME = "injourney_cx_playbooks"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180

# =============================================================================
# EMBEDDING PROVIDERS
# =============================================================================

def get_embedding_model() -> Embeddings:
    """
    Returns the embedding model based on .env configuration.
    - local: sentence-transformers (all-MiniLM-L6-v2) — works offline
    - openai: text-embedding-3-small (best quality)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        
        logger.info(f"Using OpenAI embeddings: {model}")
        return OpenAIEmbeddings(model=model, openai_api_key=api_key)
    
    else:
        # Local embeddings (default, offline capable)
        from langchain_huggingface import HuggingFaceEmbeddings
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Using local embeddings: {model_name}")
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )


# =============================================================================
# TEXT SPLITTING (Optimized for Playbooks)
# =============================================================================

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Text splitter tuned for CX playbook documents (Indonesian + English mix).
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",           # Paragraphs
            "\n",             # Lines
            ". ",             # Sentences
            "? ", 
            "! ",
            "; ",
            ", ",
            " ",             # Words
            ""                # Characters
        ],
        keep_separator=True,
    )


# =============================================================================
# VECTOR STORE MANAGEMENT
# =============================================================================

def get_vectorstore(embedding: Optional[Embeddings] = None) -> Chroma:
    """Get or create the Chroma vector store."""
    if embedding is None:
        embedding = get_embedding_model()
    
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def is_vectorstore_ready() -> bool:
    """Check if the vector store has been populated."""
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        return count > 0
    except Exception:
        return False


def get_vectorstore_stats() -> Dict[str, Any]:
    """Return useful stats about the current vector store."""
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        return {
            "document_count": count,
            "is_ready": count > 50,
            "collection_name": CHROMA_COLLECTION_NAME,
            "persist_dir": str(CHROMA_PERSIST_DIR),
        }
    except Exception as e:
        return {"error": str(e), "is_ready": False}


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

def ingest_playbooks(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Main ingestion function.
    Extracts text from all PDFs → chunks → embeds → stores in Chroma.
    
    Returns status dict with chunk count, timing, etc.
    """
    import time
    start_time = time.time()
    
    vs = get_vectorstore()
    
    if not force_rebuild and is_vectorstore_ready():
        stats = get_vectorstore_stats()
        return {
            "status": "already_exists",
            "chunks": stats.get("document_count", 0),
            "message": "Vector store already populated. Use force_rebuild=True to re-ingest.",
        }
    
    if force_rebuild:
        logger.warning("Force rebuild requested — clearing existing collection...")
        try:
            vs._collection.delete(where={})
        except Exception:
            pass
    
    # Step 1: Extract text from all PDFs
    logger.info("Starting PDF text extraction...")
    pages = process_all_playbooks()
    
    if not pages:
        return {"status": "error", "message": "No pages extracted from PDFs."}
    
    # Step 2: Convert to LangChain Documents with rich metadata
    documents = []
    for page in pages:
        doc = Document(
            page_content=page["text"],
            metadata={
                "source": page["source"],
                "page": page["page"],
                "doc_type": page.get("doc_type", "Unknown"),
            }
        )
        documents.append(doc)
    
    # Step 3: Chunk the documents
    logger.info(f"Chunking {len(documents)} pages...")
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    
    logger.info(f"Created {len(chunks)} chunks. Adding to Chroma...")
    
    # Step 4: Add to vector store (batched for large collections)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vs.add_documents(batch)
        logger.info(f"  Ingested batch {i//batch_size + 1} ({len(batch)} chunks)")
    
    # Persist
    vs.persist()
    
    elapsed = round(time.time() - start_time, 1)
    
    return {
        "status": "success",
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "time_seconds": elapsed,
        "message": f"Successfully ingested {len(chunks)} chunks from {len(pages)} pages.",
    }


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_context(
    query: str,
    k: int = 6,
    filter_metadata: Optional[Dict] = None
) -> List[Document]:
    """
    Retrieve the most relevant chunks for a given query.
    Used by both the Knowledge Base chat and audit checklist generator.
    """
    vs = get_vectorstore()
    
    if filter_metadata:
        return vs.similarity_search(query, k=k, filter=filter_metadata)
    return vs.similarity_search(query, k=k)


def retrieve_with_scores(
    query: str,
    k: int = 6
) -> List[tuple]:
    """Retrieve chunks with similarity scores (useful for debugging)."""
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=k)


# =============================================================================
# CX-SPECIFIC HELPERS (Used for dynamic checklist generation)
# =============================================================================

PILLAR_KEYWORDS = {
    "People": [
        "people", "personnel", "staff", "petugas", "karyawan", "team",
        "safe space", "positive interaction", "professionalism", "empati",
        "komunikasi", "service attitude", "customer service"
    ],
    "Premises": [
        "premises", "fasilitas", "facility", "clean", "bersih", "environment",
        "toilet", "ruang tunggu", "check-in", "boarding", "wayfinding",
        "signage", "comfort", "kebersihan", "maintenance"
    ],
    "Process": [
        "process", "proses", "procedure", "prosedur", "information", "informasi",
        "clear information", "flow", "boarding process", "check-in process",
        "baggage", "security", "immigration", "customs", "efficiency"
    ]
}


def retrieve_by_pillar(pillar: str, k: int = 5) -> List[Document]:
    """Retrieve content specifically relevant to one of the three CX pillars."""
    keywords = PILLAR_KEYWORDS.get(pillar, [])
    query = f"{pillar} pillar CX standards " + " ".join(keywords[:4])
    return retrieve_context(query, k=k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== InJourney CX RAG Engine Test ===")
    
    stats = get_vectorstore_stats()
    print(f"Vector store status: {stats}")
    
    if not stats.get("is_ready"):
        print("\nVector store not ready. Starting ingestion...")
        result = ingest_playbooks(force_rebuild=False)
        print(result)
    else:
        print(f"\nReady with {stats['document_count']} chunks.")
        
        # Quick test retrieval
        docs = retrieve_context("Apa itu Safe Space di People Pillar?", k=3)
        print("\n--- Sample retrieval ---")
        for d in docs:
            print(f"[{d.metadata['source']} p.{d.metadata['page']}] {d.page_content[:200]}...\n")

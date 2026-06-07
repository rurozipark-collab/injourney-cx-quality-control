"""
PDF Processor for InJourney CX Quality Control

Handles robust text extraction from the 6 InJourney Playbook PDFs.
Uses pdfplumber as primary extractor (better layout) with pymupdf fallback.
"""

import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_data_folder() -> Path:
    """Return path to the data folder containing PDFs."""
    return Path(__file__).parent.parent / "data"


def extract_text_with_pdfplumber(pdf_path: Path, max_pages: int = None) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using pdfplumber (preferred for structured documents).
    Returns list of dicts with page-level content + metadata.
    """
    pages_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {pdf_path.name} with pdfplumber ({total_pages} pages)")
            
            for i, page in enumerate(pdf.pages):
                if max_pages and i >= max_pages:
                    break
                
                text = page.extract_text() or ""
                
                # Clean up common issues
                text = text.strip()
                
                if text:  # Only keep pages with actual content
                    pages_content.append({
                        "source": pdf_path.name,
                        "page": i + 1,
                        "text": text,
                        "char_count": len(text),
                    })
                    
    except Exception as e:
        logger.error(f"pdfplumber failed on {pdf_path.name}: {e}")
        return []
    
    return pages_content


def extract_text_with_pymupdf(pdf_path: Path, max_pages: int = None) -> List[Dict[str, Any]]:
    """
    Fallback extractor using pymupdf (fitz).
    More aggressive on scanned/image-heavy PDFs.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.error("pymupdf (fitz) not installed. Cannot use fallback.")
        return []
    
    pages_content = []
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"Processing {pdf_path.name} with pymupdf fallback ({total_pages} pages)")
        
        for i, page in enumerate(doc):
            if max_pages and i >= max_pages:
                break
            
            text = page.get_text("text") or ""
            text = text.strip()
            
            if text:
                pages_content.append({
                    "source": pdf_path.name,
                    "page": i + 1,
                    "text": text,
                    "char_count": len(text),
                })
        
        doc.close()
        
    except Exception as e:
        logger.error(f"pymupdf failed on {pdf_path.name}: {e}")
        return []
    
    return pages_content


def process_all_playbooks(max_pages_per_pdf: int = None) -> List[Dict[str, Any]]:
    """
    Process all 6 InJourney CX documents in the data/ folder.
    Returns a flat list of page-level documents ready for chunking.
    """
    data_folder = get_data_folder()
    
    # Define the expected files (in priority order)
    expected_files = [
        "Playbook_1.pdf",
        "Playbook_2.pdf",
        "Playbook_3.pdf",
        "Playbook_4.pdf",
        "Playbook_5.pdf",
        "Transformation_Concept.pdf",
    ]
    
    all_pages = []
    
    for filename in expected_files:
        pdf_path = data_folder / filename
        
        if not pdf_path.exists():
            logger.warning(f"Missing expected file: {filename}")
            continue
        
        # Try pdfplumber first
        pages = extract_text_with_pdfplumber(pdf_path, max_pages=max_pages_per_pdf)
        
        # Fallback to pymupdf if we got very little content
        if len(pages) < 5 or sum(p["char_count"] for p in pages) < 1000:
            logger.warning(f"Low content from pdfplumber on {filename}, trying pymupdf fallback...")
            pages = extract_text_with_pymupdf(pdf_path, max_pages=max_pages_per_pdf)
        
        if pages:
            # Add document type metadata
            doc_type = "Transformation Concept" if "Transformation" in filename else "Playbook"
            for p in pages:
                p["doc_type"] = doc_type
                p["file_path"] = str(pdf_path)
            
            all_pages.extend(pages)
            logger.info(f"✓ Extracted {len(pages)} pages from {filename}")
        else:
            logger.error(f"✗ Failed to extract any text from {filename}")
    
    logger.info(f"Total pages extracted across all documents: {len(all_pages)}")
    return all_pages


def get_playbook_stats() -> Dict[str, Any]:
    """Quick stats about the PDFs present."""
    data_folder = get_data_folder()
    pdfs = list(data_folder.glob("*.pdf"))
    
    return {
        "pdf_count": len(pdfs),
        "total_size_mb": round(sum(f.stat().st_size for f in pdfs) / (1024 * 1024), 1),
        "files": [f.name for f in pdfs],
    }


if __name__ == "__main__":
    # For testing extraction directly
    logging.basicConfig(level=logging.INFO)
    print("InJourney CX PDF Processor - Test Run")
    stats = get_playbook_stats()
    print(f"Found {stats['pdf_count']} PDFs ({stats['total_size_mb']} MB total)")
    
    pages = process_all_playbooks(max_pages_per_pdf=3)  # Limit for quick test
    print(f"\nExtracted {len(pages)} sample pages")
    if pages:
        print("\n--- Sample from first page ---")
        print(pages[0]["text"][:800])

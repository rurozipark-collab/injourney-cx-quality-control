"""
InJourney Airports CX Quality Control
Full Professional Streamlit Application

Powered by:
- LangChain + ChromaDB RAG over 6 official InJourney CX Playbooks
- Dynamic scoring aligned to People / Premises / Process pillars
- Professional PDF & Excel reporting
"""

import streamlit as st
from pathlib import Path
import os
from datetime import datetime
import json
import base64
from io import BytesIO
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# EARLY FIX FOR PROTOBUF / CHROMADB CONFLICT (must be before any heavy imports)
# This is the recommended workaround from the error message itself.
# -----------------------------------------------------------------------------
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Our modules - robust imports for Streamlit
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# OPTIONAL MODULE IMPORTS (made non-fatal + lazy for heavy RAG)
# We completely avoid importing the heavy RAG stack (rag_engine, llm, pdf_processor)
# at module load time. This prevents chromadb/sentence-transformers/protobuf
# descriptor crashes from killing the entire app at startup on Streamlit Cloud.
#
# Daily Service QC (Harian) has ZERO dependency on RAG and will always work.
# The Knowledge Base tab will attempt lazy load only when opened.
# -----------------------------------------------------------------------------
AUDIT_AVAILABLE = False
RAG_AVAILABLE = False
RAG_IMPORT_ERROR = "RAG stack is loaded lazily only inside Knowledge Base tab to keep startup stable."

# Audit manager (used by full inspection / reports). Local file ops, usually safe.
try:
    from utils.audit_manager import (
        save_audit,
        load_audit,
        list_all_audits,
        generate_audit_id,
        delete_audit,
        delete_all_audits,
        create_backup,
        delete_audit_with_backup,
        duplicate_audit,
        update_audit,
        AUDITS_DIR,
    )
    AUDIT_AVAILABLE = True
except Exception as audit_err:
    # Do not crash the whole app — Daily Service QC does not depend on this.
    pass

# NOTE: We deliberately do NOT do the "from utils.rag_engine import ..." here anymore.
# The heavy imports (which trigger the protobuf error) are moved to lazy loading
# inside render_knowledge_base() only. This keeps the app starting reliably.

# Report generators (will be created next)
# For now we implement basic versions inline

# =============================================================================
# ENVIRONMENT & CONFIGURATION HELPERS
# =============================================================================

def get_environment_status():
    """Check current environment configuration status."""
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY", "")
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    
    has_openai_key = bool(openai_key and openai_key != "sk-your-openai-api-key-here")
    
    return {
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "has_openai_key": has_openai_key,
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


def render_environment_banner():
    """Show helpful banner only if needed. In local mode we keep it minimal."""
    env = get_environment_status()
    
    # Only show a very subtle message, or nothing at all in normal use
    if not env["has_openai_key"]:
        # We hide the prominent banner by default for cleaner experience
        # User can still see status in sidebar
        pass  # No banner shown to keep UI clean


# =============================================================================
# CACHED RAG HELPERS (Important for performance)
# Completely lazy: we only attempt to import the heavy RAG modules when
# the Knowledge Base tab is actually opened. This avoids crashing the
# entire app at startup on Streamlit Cloud due to protobuf/chromadb issues.
# For normal use (Daily Service QC), these functions just return safe defaults.
# =============================================================================

def _try_import_rag():
    """Attempt to import the heavy RAG symbols. Returns True on success."""
    global RAG_AVAILABLE, RAG_IMPORT_ERROR
    if RAG_AVAILABLE:
        return True
    try:
        from utils.rag_engine import (
            ingest_playbooks,
            get_vectorstore,
            retrieve_context,
        )
        from utils.llm import ask_playbook_question
        from utils.pdf_processor import get_playbook_stats

        # Make them available in global scope for the rest of the module
        globals()["ingest_playbooks"] = ingest_playbooks
        globals()["get_vectorstore"] = get_vectorstore
        globals()["retrieve_context"] = retrieve_context
        globals()["ask_playbook_question"] = ask_playbook_question
        globals()["get_playbook_stats"] = get_playbook_stats

        RAG_AVAILABLE = True
        RAG_IMPORT_ERROR = ""
        return True
    except Exception as import_err:
        RAG_IMPORT_ERROR = str(import_err)
        RAG_AVAILABLE = False
        return False


@st.cache_resource
def get_cached_vectorstore():
    """Cache the heavy Chroma vectorstore + embedding model."""
    if not _try_import_rag():
        return None
    try:
        return get_vectorstore()
    except Exception:
        return None


@st.cache_resource
def get_cached_embedding_model():
    """Cache the embedding model."""
    if not _try_import_rag():
        return None
    try:
        from utils.rag_engine import get_embedding_model
        return get_embedding_model()
    except Exception:
        return None


def is_vectorstore_ready() -> bool:
    """Check if the vector store has been populated (uses cached store)."""
    if not _try_import_rag():
        return False
    try:
        vs = get_cached_vectorstore()
        if vs is None:
            return False
        count = vs._collection.count()
        return count > 0
    except Exception:
        return False


def get_vectorstore_stats() -> dict:
    """Get stats from the cached vectorstore."""
    if not _try_import_rag():
        return {"error": "RAG modules not available", "is_ready": False}
    try:
        vs = get_cached_vectorstore()
        if vs is None:
            return {"error": "Vectorstore not loaded", "is_ready": False}
        count = vs._collection.count()
        return {
            "document_count": count,
            "is_ready": count > 50,
            "collection_name": "injourney_cx_playbooks",
        }
    except Exception as e:
        return {"error": str(e), "is_ready": False}

# =============================================================================
# PAGE CONFIG & BRANDING
# =============================================================================

st.set_page_config(
    page_title="InJourney Airports CX Quality Control",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",  # Better for mobile
)

# PWA + iPhone specific meta tags (for installing as app)
st.markdown("""
<link rel="manifest" href="/static/manifest.json">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="CX Daily QC">
<link rel="apple-touch-icon" href="/assets/injourney_airports_logo.png">

<style>
/* Mobile-friendly tweaks */
@media (max-width: 768px) {
    .stApp {
        padding: 0.4rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        gap: 4px;
    }
    .stButton > button {
        padding: 0.7rem 1.1rem !important;
        font-size: 1rem !important;
        min-height: 48px;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        font-size: 16px !important; /* Prevent iOS zoom */
        padding: 12px !important;
    }
    .cx-card {
        padding: 0.7rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stMetric {
        font-size: 0.9rem !important;
    }
    .stExpander {
        font-size: 0.95rem !important;
    }
}

/* Make it feel more like an installed app on iPhone */
@media (display-mode: standalone) {
    .stAppHeader {
        display: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Register Service Worker for PWA / offline shell
st.markdown("""
<script>
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/service-worker.js')
            .then(function(registration) {
                console.log('ServiceWorker registered successfully');
            })
            .catch(function(error) {
                console.log('ServiceWorker registration failed: ', error);
            });
    });
}
</script>
""", unsafe_allow_html=True)

# InJourney Professional Color Palette (Light + Dark support)
INJOURNEY = {
    # Brand colors (same for both themes)
    "primary": "#00A8A8",
    "deep": "#008080",
    "blue": "#003366",
    "sky": "#00B4D8",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",

    # Light mode
    "light": {
        "bg": "#F8FAFC",
        "card": "#FFFFFF",
        "text": "#1E293B",
        "muted": "#64748B",
        "border": "#E2E8F0",
    },

    # Dark mode
    "dark": {
        "bg": "#0F172A",
        "card": "#1E293B",
        "text": "#F1F5F9",
        "muted": "#94A3B8",
        "border": "#334155",
    }
}

APP_VERSION = "v1.4 - Final"

# =============================================================================
# CX PLAYBOOK STANDARDS (for PDF reference section)
# =============================================================================

PLAYBOOK_STANDARDS = {
    "People": {
        "General Wellness & Personal Hygiene": 
            "Menurut CX Playbook, karyawan wajib menjaga kebersihan pribadi secara menyeluruh. "
            "Tidak boleh ada bau badan, bau mulut, atau keringat berlebih. Kuku harus pendek, bersih, dan rapi. "
            "Wajah harus segar dan tidak berminyak berlebihan. Gigi harus bersih dan nafas segar. "
            "Ini adalah standar dasar penampilan profesional di lingkungan bandara.",

        "Rambut & Personal Presentation":
            "Rambut harus rapi dan sesuai dengan standar yang berlaku. "
            "Pria: model Fade atau Taper yang halus. Dilarang model punk, spiky, skin-cropped berlebihan, atau undercut yang mencolok. "
            "Wanita: rambut panjang wajib disanggul rapi (Chignon atau French Twist). Rambut pendek tidak boleh menyentuh bahu. "
            "Warna rambut harus natural. Make-up dan parfum harus natural, profesional, dan tidak berlebihan.",

        "Seragam & Kelengkapan":
            "Seragam harus lengkap, rapi, dan disetrika. Tidak boleh kusut. "
            "Name tag/ID Card wajib terpasang dengan benar dan terlihat jelas dari depan. "
            "Dasi atau scarf harus rapi. Sepatu pantofel hitam formal harus bersih dan dalam kondisi baik. "
            "Panjang rok atau celana harus sesuai ketentuan perusahaan.",

        "Aksesoris & Atribut":
            "Aksesoris dibatasi maksimal 5 titik di seluruh tubuh. "
            "Cincin maksimal 1 buah dengan ukuran kecil. Kalung harus berada di dalam pakaian dan tidak terlihat. "
            "Bagi yang memakai kerudung, wajib rapi, kuat, menutupi seluruh rambut dan leher, serta menggunakan ciput.",

        "Sikap Tubuh & Ekspresi":
            "Karyawan wajib berdiri tegak dengan bahu rileks dan siap melayani. "
            "Dilarang membungkuk, menyilangkan tangan di dada, atau memasukkan tangan ke saku saat berinteraksi dengan penumpang. "
            "Harus menampilkan senyum tulus dan ramah serta melakukan kontak mata yang natural. "
            "Wajah datar, cemberut, atau sinis tidak diperbolehkan.",

        "Komunikasi & Etika Profesional":
            "Saat menerima atau menyerahkan dokumen, wajib menggunakan dua tangan. "
            "Telapak tangan harus terbuka saat menunjuk arah. "
            "Dilarang menggunakan ponsel atau earphone di area publik saat berseragam. "
            "Dilarang makan atau minum di tempat terlarang. Tidak boleh berjalan berkelompok sambil berbicara keras."
    },

    "Process": {
        "Pre-Journey & Digital Processes":
            "Website dan aplikasi harus mudah digunakan. Proses check-in online harus lancar. "
            "Informasi penerbangan real-time harus akurat. Notifikasi dan pengingat harus dikirim tepat waktu.",

        "Arrival & Ground Transportation Processes":
            "Proses dari pesawat menuju terminal harus efisien. Transportasi darat harus mudah diakses. "
            "Antrian di imigrasi dan keamanan harus dikelola dengan baik. Staf harus ramah dan membantu penumpang yang bingung.",

        "Check-in & Baggage Processes":
            "Antrian check-in harus tertata rapi dan bergerak cepat. Staf harus sopan, efisien, dan informatif. "
            "Penanganan bagasi harus hati-hati. Self check-in kiosk harus berfungsi dengan baik.",

        "Security, Immigration & Customs Processes":
            "Pemeriksaan keamanan harus dilakukan dengan sopan. Antrian harus dikelola dengan baik. "
            "Petugas harus ramah dan memberikan penjelasan jika diperlukan. Proses harus lancar tanpa penumpukan berlebihan.",

        "Boarding & Departure Processes":
            "Proses boarding harus tertib dan sesuai jadwal. Pengumuman harus jelas. "
            "Staf harus membantu penumpang prioritas dengan baik. Area gate harus nyaman dan tidak terlalu padat.",

        "Baggage Claim & Post-Arrival Processes":
            "Bagasi harus datang tepat waktu. Area klaim bagasi harus bersih. "
            "Informasi bagasi yang tertinggal harus jelas. Proses keluar terminal harus berjalan lancar."
    },

    "Premises": {
        "Public Spaces & Outdoor Areas":
            "Lantai, dinding, dan langit-langit harus bersih dari noda, debu, dan sampah. "
            "Pencahayaan harus cukup dan merata. Tidak boleh ada bau tidak sedap. "
            "Kursi dan area duduk harus dalam kondisi baik.",

        "Parking & Curbside / Arrival Facilities":
            "Area drop-off dan pick-up harus tertata rapi. Trotoar dan jalur pejalan kaki harus bersih dan mudah dilalui. "
            "Tanda petunjuk arah ke terminal harus jelas.",

        "Customer Facilities":
            "Toilet harus bersih, kering, dan tidak berbau. Perlengkapan (sabun, tissue, hand dryer) harus lengkap. "
            "Musholla harus nyaman dan memiliki fasilitas wudhu yang memadai. WiFi harus berfungsi dengan baik.",

        "Special Needs & Inclusive Facilities":
            "Harus tersedia jalur khusus untuk penyandang disabilitas (ramp, lift, toilet accessible). "
            "Tactile paving harus dalam kondisi baik. Tersedia kursi roda dan bantuan staf. "
            "Nursing room harus bersih dan nyaman.",

        "Safety & Security Facilities":
            "CCTV harus berfungsi dan area terasa aman. Tanda darurat, jalur evakuasi, dan alat pemadam api harus jelas terlihat. "
            "Tidak boleh ada area gelap atau rawan di dalam terminal.",

        "Wayfinding, Signage & Information Systems":
            "Papan petunjuk arah harus jelas, konsisten, dan mudah dibaca. "
            "Peta terminal dan informasi penerbangan (FIDS) harus akurat. "
            "Informasi harus tersedia dalam bahasa Indonesia dan Inggris. Staf informasi harus mudah ditemukan dan membantu dengan ramah."
    }
}


# =============================================================================
# CUSTOM CSS
# =============================================================================

def apply_branding():
    light = INJOURNEY["light"]
    dark = INJOURNEY["dark"]

    css = f"""
    <style>
    /* ==================== BASE (LIGHT MODE) ==================== */
    .stApp {{
        background-color: {light["bg"]} !important;
    }}

    .injourney-header {{
        background: linear-gradient(90deg, {INJOURNEY["blue"]} 0%, {INJOURNEY["primary"]} 100%);
        padding: 1rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 0.8rem;
        box-shadow: 0 6px 16px rgba(0, 168, 168, 0.18);
        color: white;
    }}

    .injourney-header h1 {{
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }}

    .cx-card {{
        background: {light["card"]};
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid {light["border"]};
        margin-bottom: 0.8rem;
        color: {light["text"]};
    }}

    .pillar-badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.35rem;
    }}
    .pillar-people {{ background:#DBEAFE; color:#1E40AF; }}
    .pillar-premises {{ background:#D1FAE5; color:#065F46; }}
    .pillar-process {{ background:#FEF3C7; color:#92400E; }}

    /* Category Cards */
    .category-card {{
        padding: 8px 14px;
        margin-bottom: 6px;
        border-radius: 8px;
        border-left: 5px solid;
        background-color: #F8FAFC;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 600;
    }}
    .category-card-people {{ border-left-color: #1E40AF; }}
    .category-card-process {{ border-left-color: #3730A3; }}
    .category-card-premises {{ border-left-color: #065F46; }}

    .category-title {{
        font-size: 0.93rem;
        font-weight: 700;
        color: #1E293B;
    }}

    /* Score colors - with subtle background for better visibility */
    .score-high {{
        color: #059669 !important;
        font-weight: 700;
        background: #D1FAE5;
        padding: 2px 8px;
        border-radius: 6px;
    }}
    .score-medium {{
        color: #D97706 !important;
        font-weight: 700;
        background: #FEF3C7;
        padding: 2px 8px;
        border-radius: 6px;
    }}
    .score-low {{
        color: #DC2626 !important;
        font-weight: 700;
        background: #FEE2E2;
        padding: 2px 8px;
        border-radius: 6px;
    }}

    /* Guide Headers (colored gradients) */
    .guide-header, .pillar-title-header {{
        color: white !important;
        padding: 12px 16px;
        border-radius: 10px;
        margin: 8px 0 6px 0;
        font-size: 1.02rem;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(0,0,0,0.18);
        display: flex;
        align-items: center;
        gap: 10px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .pillar-title-header:hover {{
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(0,0,0,0.25);
    }}

    /* Pillar Title Headers */
    .pillar-title-header.people {{ background: linear-gradient(90deg, #1E40AF 0%, #3B82F6 100%); }}
    .pillar-title-header.process {{ background: linear-gradient(90deg, #3730A3 0%, #6366F1 100%); }}
    .pillar-title-header.premises {{ background: linear-gradient(90deg, #065F46 0%, #10B981 100%); }}

    /* Expander arrow area */
    .pillar-section [data-testid="stExpander"] details summary {{
        padding: 0 6px !important;
        min-height: 16px !important;
        margin: -4px 0 4px 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }}
    .pillar-section [data-testid="stExpander"] details summary > span,
    .pillar-section [data-testid="stExpander"] details summary p {{
        display: none !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: white;
        padding: 5px;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        border-radius: 8px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {INJOURNEY["primary"]}, {INJOURNEY["sky"]});
        color: white !important;
    }}

    .stButton > button {{
        background: linear-gradient(90deg, {INJOURNEY["primary"]}, {INJOURNEY["deep"]});
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.4rem 1rem;
    }}

    .metric-card {{
        background: white;
        border-left: 5px solid {INJOURNEY["primary"]};
        padding: 0.85rem 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}

    /* ==================== DARK MODE OVERRIDES ==================== */
    [data-theme="dark"] .stApp {{
        background-color: {dark["bg"]} !important;
    }}

    [data-theme="dark"] .injourney-header {{
        background: linear-gradient(90deg, #0F172A 0%, #134E4B 100%);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    }}

    [data-theme="dark"] .cx-card {{
        background: {dark["card"]};
        border: 1px solid {dark["border"]};
        color: {dark["text"]};
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}

    [data-theme="dark"] .category-card {{
        background-color: #1E293B;
        border-color: #475569;
        color: #E2E8F0;
    }}

    [data-theme="dark"] .category-title {{
        color: #F1F5F9;
    }}

    [data-theme="dark"] .stTabs [data-baseweb="tab-list"] {{
        background: #1E293B;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }}

    [data-theme="dark"] .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {INJOURNEY["primary"]}, {INJOURNEY["sky"]});
        color: white !important;
    }}

    [data-theme="dark"] .metric-card {{
        background: {dark["card"]};
        border-left-color: {INJOURNEY["primary"]};
        color: {dark["text"]};
    }}

    [data-theme="dark"] .stMarkdown,
    [data-theme="dark"] p,
    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3,
    [data-theme="dark"] label, [data-theme="dark"] span {{
        color: {dark["text"]} !important;
    }}

    [data-theme="dark"] .stExpander,
    [data-theme="dark"] .stDataFrame,
    [data-theme="dark"] .stMetric {{
        background-color: {dark["card"]} !important;
    }}

    /* (Old logo switching CSS removed) */

    /* ========== IMPROVED DARK MODE COLORS ========== */
    [data-theme="dark"] .injourney-header {{
        background: linear-gradient(90deg, #1E293B 0%, #0F766E 100%);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }}

    /* Better pillar badges in dark */
    [data-theme="dark"] .pillar-people {{ background:#1E3A8A; color:#93C5FD; }}
    [data-theme="dark"] .pillar-premises {{ background:#064E3B; color:#6EE7B7; }}
    [data-theme="dark"] .pillar-process {{ background:#78350F; color:#FCD34D; }}

    /* (Category card dark styles already defined above) */

    /* Make guide/pillar headers slightly stronger in dark */
    [data-theme="dark"] .guide-header,
    [data-theme="dark"] .pillar-title-header {{
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35);
    }}

    /* Slightly tone down buttons in dark for better balance */
    [data-theme="dark"] .stButton > button {{
        box-shadow: 0 2px 8px rgba(0, 168, 168, 0.3);
    }}

    /* Improve tab contrast in dark */
    [data-theme="dark"] .stTabs [data-baseweb="tab"] {{
        color: #CBD5E1;
    }}

    /* =============================================
       FORCE LIGHT MODE ONLY
       (Dark & System modes disabled as per user request)
    ============================================= */
    
    /* Hide the three dots menu (theme switcher) completely - Light mode only */
    [data-testid="stHeaderActionButton"],
    header [data-testid="stHeaderActionButton"],
    .stApp header [data-testid="stHeaderActionButton"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}
    
    /* Aggressively force light theme background and colors */
    .stApp {{
        background-color: #F8FAFC !important;
    }}
    
    /* Ensure all text stays dark in light mode */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, span, div {{
        color: #1E293B !important;
    }}

    /* Protect the big colored pillar headers (People / Process / Premises) so text stays white */
    .pillar-title-header,
    .pillar-title-header * {{
        color: #FFFFFF !important;
    }}

    /* Also protect the Panduan Detail headers inside the pillars */
    .guide-header,
    .guide-header * {{
        color: #FFFFFF !important;
    }}
    
    /* Force light cards and containers */
    .stExpander, .stDataFrame, .stMetric, [data-testid="stMetric"], 
    .stTextInput, .stTextArea, .stSelectbox, .stFileUploader {{
        background-color: white !important;
    }}
    
    /* Force all text inside the main header to be pure white */
    .injourney-header,
    .injourney-header *,
    .injourney-header div,
    .injourney-header span {{
        color: #FFFFFF !important;
    }}

    /* Sidebar fallback text */
    .sidebar-fallback-title {{
        color: #003366;
        margin: 0;
        font-size: 1.35rem;
        font-weight: 800;
    }}
    .sidebar-fallback-subtitle {{
        color: #00A8A8;
        margin: 0;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
    }}

    /* App Footer */
    .app-footer {{
        text-align: center;
        color: #64748B;
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
    }}

    /* (Header date is now styled inline for simplicity) */
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_reports_dir() -> Path:
    # Support custom data directory for cloud deployments (e.g. mounted disk on Render)
    base_dir = os.getenv("DATA_DIR", str(Path(__file__).parent))
    return Path(base_dir) / "reports"


def get_audits_dir() -> Path:
    return get_reports_dir() / "audits"


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        # Logo InJourney Airports
        logo_path = Path(__file__).parent / "assets" / "injourney_airports_logo.png"
        
        if logo_path.exists():
            st.image(str(logo_path), width=200)
        else:
            st.markdown("""
            <div style="text-align:center; padding-top:0.5rem;">
                <h2 class="sidebar-fallback-title">✈️ INJOURNEY</h2>
                <p class="sidebar-fallback-subtitle">AIRPORTS</p>
            </div>
            """, unsafe_allow_html=True)

        # Visi InJourney Airports - Elegant & Compact
        st.markdown("""
        <div style="margin-top: 6px; margin-bottom: 10px; padding: 0 6px;">
            <div style="font-size: 0.72rem; font-weight: 600; color: #003366; margin-bottom: 3px; letter-spacing: 0.4px;">
                Visi InJourney Airports
            </div>
            <div style="font-size: 0.68rem; line-height: 1.42; color: #475569; font-style: italic;">
                Menjadi penghubung dunia yang lebih dari sekadar operator bandar udara dengan keunggulan layanan yang menampilkan keramahtamahan khas Indonesia.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # === Modern Context Section ===
        airport = "Soekarno-Hatta (CGK)"

        with st.container(border=True):
            st.markdown("**📍 Konteks Inspection**")
            st.caption("Soekarno-Hatta International Airport (CGK)")

            # Terminal selection - stabilized with key for mobile/PWA
            if "terminal_utama" not in st.session_state:
                st.session_state.terminal_utama = "Terminal 1"

            terminal_utama = st.selectbox(
                "Terminal",
                ["Terminal 1", "Terminal 2", "Terminal 3"],
                key="terminal_utama"
            )

            # Area / Sub Terminal - reset when terminal changes
            if terminal_utama == "Terminal 1":
                sub_options = ["A", "B", "C"]
            elif terminal_utama == "Terminal 2":
                sub_options = ["D", "E", "F"]
            else:
                sub_options = ["Domestik", "Internasional"]

            # Preserve sub choice if possible, else default to first
            if "sub_terminal" not in st.session_state or st.session_state.get("last_terminal_utama") != terminal_utama:
                st.session_state.sub_terminal = sub_options[0]
                st.session_state.last_terminal_utama = terminal_utama

            sub_terminal = st.selectbox(
                "Area / Sub Terminal",
                sub_options,
                key="sub_terminal"
            )

            terminal = f"{terminal_utama} - {sub_terminal}"
            location = f"{airport} - {terminal}"

            # Nama Inspector
            auditor = st.text_input(
                "Nama Inspector",
                key="sidebar_inspector_name",
                placeholder="Masukkan nama inspector"
            )

            # Tanggal
            audit_date = st.date_input(
                "Tanggal",
                value=datetime.now().date()
            )

            # Ringkasan Lokasi (live update)
            st.markdown("---")
            st.markdown("**Lokasi Saat Ini**")
            st.info(f"**{airport}**  •  **{terminal}**", icon="📍")
        
        st.markdown("---")
        st.caption(f"{APP_VERSION} • Powered by 6 Official CX Playbooks")

        # RAG status: we force False at sidebar level to avoid triggering
        # the heavy chromadb/sentence-transformers import during normal startup
        # (Daily Service QC + most tabs don't need it).
        # The Knowledge Base tab will attempt lazy load when the user opens it.
        rag_ready = False

        return {
            "airport": airport,
            "terminal": terminal,
            "location": location,
            "inspector": auditor,   # alias for Daily QC
            "auditor": auditor,
            "audit_date": str(audit_date),
            "rag_ready": rag_ready,
        }


# =============================================================================
# HEADER
# =============================================================================

def render_header():
    """Clean main header - full width with title on left and date on right."""
    st.markdown(f"""
    <div class="injourney-header" style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.5px; color: #FFFFFF;">
                InJourney Airports CX Quality Control
            </h1>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #FFFFFF;">
                Tools Inspection Kualitas Layanan Berdasarkan 6 Dokumen CX Playbook InJourney
            </p>
        </div>
        <div style="text-align: right; font-size: 0.78rem; color: #FFFFFF !important; line-height: 1.3;">
            {datetime.now().strftime("%d %b %Y")}<br>
            <span style="font-size: 0.65rem; color: #FFFFFF !important;">Internal Tool</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DYNAMIC CHECKLIST (Seeded + Retrieval Enhanced)
# =============================================================================

def get_base_checklist_items():
    """
    High-quality checklist items derived from InJourney CX Playbooks (Playbook 2, 3, 4).
    People Pillar items are based on detailed standards from CX Guidance for People Pillar (Playbook 4).
    """
    return [
        # =====================================================================
        # PEOPLE PILLAR - Detailed & Professional (from Playbook 4)
        # =====================================================================
        # 1. General Wellness & Personal Hygiene
        {"id": "p01", "pillar": "People", "element": "General Wellness",
         "item": "Kebersihan dan kerapian pribadi secara keseluruhan (tidak ada bau badan, bau mulut, atau keringat berlebih; tubuh, tangan, dan kaki bersih)", "weight": 1.0},
        {"id": "p02", "pillar": "People", "element": "General Wellness",
         "item": "Kondisi kuku tangan dan kaki (pendek, rapi, bersih, tidak ada kotoran, tidak menggunakan cat kuku mencolok atau nail art)", "weight": 0.9},
        {"id": "p03", "pillar": "People", "element": "General Wellness",
         "item": "Kebersihan wajah, gigi, dan nafas (wajah segar dan bersih dari minyak berlebih, gigi bersih, nafas segar)", "weight": 0.85},

        # 2. Rambut, Wajah & Personal Presentation
        {"id": "p04", "pillar": "People", "element": "Hair & Personal Presentation",
         "item": "Kerapian dan model rambut Pria (pendek, rapi, model Fade/Taper halus sesuai panduan; tidak ada punk, spiky, skin-cropped berlebihan, atau undercut)", "weight": 0.9},
        {"id": "p05", "pillar": "People", "element": "Hair & Personal Presentation",
         "item": "Kerapian dan penataan rambut Wanita (jika panjang: disanggul rapi dengan Chignon/French Twist; jika pendek: rapi tidak menyentuh bahu; warna natural hitam/coklat tua, tanpa highlight)", "weight": 0.9},
        {"id": "p06", "pillar": "People", "element": "Hair & Personal Presentation",
         "item": "Penggunaan make-up dan parfum (make-up natural & profesional, tidak berlebihan; parfum wangi ringan - citrus/aquatic/woody, tidak menyengat)", "weight": 0.8},

        # 3. Seragam & Kelengkapan
        {"id": "p07", "pillar": "People", "element": "Uniform & Attire",
         "item": "Kondisi dan cara pemakaian seragam (lengkap, rapi, disetrika, tidak kusut; jas dikancing dengan benar; kemeja dimasukkan ke dalam celana/rok)", "weight": 1.0},
        {"id": "p08", "pillar": "People", "element": "Uniform & Attire",
         "item": "Kelengkapan dan kerapian atribut seragam (Name tag/ID Card terpasang benar dan terlihat jelas; dasi/scarf rapi dan posisi tepat; sepatu pantofel hitam formal bersih)", "weight": 0.95},
        {"id": "p09", "pillar": "People", "element": "Uniform & Attire",
         "item": "Penggunaan rok/celana sesuai ketentuan (rok pendek 5 cm di bawah lutut, rok panjang mencapai mata kaki; celana tepat panjangnya; selalu menggunakan sabuk)", "weight": 0.85},

        # 4. Aksesoris & Atribut
        {"id": "p10", "pillar": "People", "element": "Accessories & Attributes",
         "item": "Penggunaan aksesoris sesuai batas (maksimal 5 titik di seluruh tubuh; cincin maksimal 1 dan ukuran kecil; kalung tidak terlihat di luar seragam)", "weight": 0.8},
        {"id": "p11", "pillar": "People", "element": "Accessories & Attributes",
         "item": "Penggunaan kerudung bagi yang memakai (rapi, kuat, menutupi seluruh rambut dan leher dengan baik; menggunakan ciput; aksesoris minimal dan serasi dengan seragam)", "weight": 0.85},

        # 5. Sikap Tubuh, Ekspresi & Perilaku
        {"id": "p12", "pillar": "People", "element": "Body Language & Posture",
         "item": "Postur tubuh dan bahasa tubuh profesional (berdiri tegak dengan bahu rileks; tidak membungkuk, tidak silang tangan di dada, tidak memasukkan tangan ke saku saat berinteraksi)", "weight": 1.0},
        {"id": "p13", "pillar": "People", "element": "Facial Expression & Eye Contact",
         "item": "Ekspresi wajah dan kontak mata (menampilkan senyum tulus dan ramah; kontak mata natural saat berbicara; tidak menunjukkan wajah datar, kesal, atau sinis)", "weight": 1.0},

        # 6. Komunikasi & Etika Profesional
        {"id": "p14", "pillar": "People", "element": "Communication & Interaction",
         "item": "Penggunaan komunikasi verbal & non-verbal sesuai panduan (menggunakan dua tangan saat menerima/menyerahkan dokumen; telapak tangan terbuka saat menunjuk arah; intonasi tenang dan jelas)", "weight": 0.95},
        {"id": "p15", "pillar": "People", "element": "Professional Ethics & Conduct",
         "item": "Etika perilaku saat bertugas (tidak menggunakan ponsel/earphone di area publik; tidak makan/minum di tempat terlarang; tidak berjalan berkelompok sambil berbicara keras)", "weight": 0.9},

        # =====================================================================
        # PREMISES PILLAR - Detailed (from Playbook 3 - CX Guidance for Premises Pillar)
        # Grouped by major facility types (consistent with People & Process)
        # =====================================================================

        # 1. Public Spaces & Outdoor Areas
        {"id": "pr01", "pillar": "Premises", "element": "Public Spaces & Outdoor",
         "item": "Jalur pedestrian bersih, memiliki pencahayaan yang memadai, permukaan anti-slip, dan dilengkapi peneduh dari cuaca ekstrem", "weight": 1.0},
        {"id": "pr02", "pillar": "Premises", "element": "Public Spaces & Outdoor",
         "item": "Area landscape/outdoor (taman, tanaman, instalasi seni) terawat dengan baik, menggunakan tanaman khas lokal, dan bebas dari sampah", "weight": 0.9},
        {"id": "pr03", "pillar": "Premises", "element": "Public Spaces & Outdoor",
         "item": "Fasilitas umum di area outdoor (tempat duduk, tempat sampah, peneduh) tersedia dalam jumlah cukup, nyaman, dan mudah dijangkau penumpang", "weight": 0.85},

        # 2. Parking & Curbside / Arrival Facilities
        {"id": "pr04", "pillar": "Premises", "element": "Parking & Curbside",
         "item": "Area drop-off/pick-up memiliki marka yang jelas, sistem antrian teratur, dan tidak menimbulkan kemacetan berlebih pada jam sibuk", "weight": 1.0},
        {"id": "pr05", "pillar": "Premises", "element": "Parking & Curbside",
         "item": "Area parkir memiliki pencahayaan yang baik, CCTV yang berfungsi, dan sistem pembayaran/tiket yang operasional dengan baik", "weight": 0.95},
        {"id": "pr06", "pillar": "Premises", "element": "Parking & Curbside",
         "item": "Petunjuk arah dari area parkir dan curbside menuju terminal jelas, konsisten, dan mudah diikuti oleh pengemudi maupun penumpang", "weight": 0.9},

        # 3. Customer Facilities
        {"id": "pr07", "pillar": "Premises", "element": "Customer Facilities",
         "item": "Toilet umum bersih, terawat, dilengkapi perlengkapan lengkap (tisu, sabun, hand dryer), dan semua fixture berfungsi dengan baik", "weight": 1.0},
        {"id": "pr08", "pillar": "Premises", "element": "Customer Facilities",
         "item": "Area tempat duduk, charging station, dan ruang tunggu dalam kondisi baik, jumlahnya memadai, serta bersih dan nyaman digunakan", "weight": 0.95},
        {"id": "pr09", "pillar": "Premises", "element": "Customer Facilities",
         "item": "Fasilitas self-service (self bag drop, self check-in) beroperasi dengan baik, memiliki petunjuk yang jelas, dan mudah digunakan penumpang", "weight": 0.9},

        # 4. Special Needs & Inclusive Facilities
        {"id": "pr10", "pillar": "Premises", "element": "Special Needs & Inclusive",
         "item": "Fasilitas untuk penumpang disabilitas (ramp, toilet accessible, jalur prioritas, lift) tersedia, berfungsi baik, dan jelas penandaannya", "weight": 1.0},
        {"id": "pr11", "pillar": "Premises", "element": "Special Needs & Inclusive",
         "item": "Ruang keluarga/nursing room dan fasilitas untuk lansia serta anak-anak tersedia, bersih, nyaman, dan mudah diakses", "weight": 0.95},
        {"id": "pr12", "pillar": "Premises", "element": "Special Needs & Inclusive",
         "item": "Trolley dan layanan bantuan khusus untuk penumpang berkebutuhan khusus tersedia di titik-titik strategis dan mudah diperoleh", "weight": 0.9},

        # 5. Safety & Security Facilities
        {"id": "pr13", "pillar": "Premises", "element": "Safety & Security",
         "item": "Sistem keselamatan (pencahayaan darurat, rambu evakuasi, alat pemadam api) terlihat jelas, tidak terhalang, dan dalam kondisi siap pakai", "weight": 1.0},
        {"id": "pr14", "pillar": "Premises", "element": "Safety & Security",
         "item": "Permukaan lantai di area publik menggunakan material anti-slip yang sesuai, terutama di zona basah atau rawan tumpahan cairan", "weight": 0.95},
        {"id": "pr15", "pillar": "Premises", "element": "Safety & Security",
         "item": "Area fasilitas umum memiliki cakupan CCTV yang memadai dengan kualitas gambar yang baik dan sistem berfungsi normal", "weight": 0.9},

        # 6. Wayfinding, Signage & Information Systems
        {"id": "pr16", "pillar": "Premises", "element": "Wayfinding & Signage",
         "item": "Sistem signage dan wayfinding di seluruh terminal konsisten, mudah dipahami (Indonesia & Inggris), serta ditempatkan di lokasi strategis", "weight": 1.0},
        {"id": "pr17", "pillar": "Premises", "element": "Wayfinding & Signage",
         "item": "Flight Information Display System (FIDS) dan peta digital berfungsi dengan baik serta menampilkan informasi yang akurat dan real-time", "weight": 0.95},
        {"id": "pr18", "pillar": "Premises", "element": "Wayfinding & Signage",
         "item": "Lokasi information desk/help point mudah terlihat, memberikan pelayanan yang responsif, dan memiliki petugas yang siap membantu", "weight": 0.9},

        # =====================================================================
        # PROCESS PILLAR - Detailed (from Playbook 2 - CX Guidance for Process Pillar)
        # Grouped by major airport process stages (Option A)
        # =====================================================================
        
        # 1. Pre-Journey & Digital Processes
        {"id": "ps01", "pillar": "Process", "element": "Pre-Journey & Digital",
         "item": "Informasi penerbangan, jadwal, dan prosedur tersedia akurat, lengkap, dan mudah diakses melalui website, aplikasi, dan channel digital lainnya", "weight": 1.0},
        {"id": "ps02", "pillar": "Process", "element": "Pre-Journey & Digital",
         "item": "Proses online check-in dan pembayaran berjalan lancar dengan instruksi yang jelas serta konfirmasi yang andal", "weight": 0.95},
        {"id": "ps03", "pillar": "Process", "element": "Pre-Journey & Digital",
         "item": "Update status penerbangan dan informasi penting tersedia secara real-time dan konsisten di berbagai platform digital", "weight": 0.9},

        # 2. Arrival, Drop-off & Ground Transportation Processes
        {"id": "ps04", "pillar": "Process", "element": "Arrival & Ground Transport",
         "item": "Proses drop-off/curbside berjalan efisien, aman, dan tidak menimbulkan kemacetan atau antrian berlebihan", "weight": 0.95},
        {"id": "ps05", "pillar": "Process", "element": "Arrival & Ground Transport",
         "item": "Petunjuk arah dan informasi dari area drop-off/transportasi umum menuju terminal jelas, mudah diikuti, dan tersedia dalam berbagai bahasa", "weight": 0.9},

        # 3. Check-in & Baggage Processes
        {"id": "ps06", "pillar": "Process", "element": "Check-in & Baggage",
         "item": "Waktu tunggu di check-in counter dan bag drop dalam batas wajar serta sesuai standar operasional", "weight": 1.0},
        {"id": "ps07", "pillar": "Process", "element": "Check-in & Baggage",
         "item": "Prosedur check-in, verifikasi dokumen, dan penanganan bagasi dilakukan secara konsisten dan profesional oleh petugas", "weight": 0.95},
        {"id": "ps08", "pillar": "Process", "element": "Check-in & Baggage",
         "item": "Penanganan penumpang dengan kebutuhan khusus (disabilitas, lansia, unaccompanied minor, dll) dilakukan secara efisien dan dengan prioritas yang tepat", "weight": 0.9},

        # 4. Security, Immigration & Customs Processes
        {"id": "ps09", "pillar": "Process", "element": "Security & Immigration",
         "item": "Proses screening keamanan berjalan efisien dengan pengelolaan antrian yang baik dan waktu tunggu yang minim", "weight": 1.0},
        {"id": "ps10", "pillar": "Process", "element": "Security & Immigration",
         "item": "Instruksi dan prosedur keamanan serta imigrasi disampaikan dengan jelas, sopan, dan mudah dipahami oleh penumpang", "weight": 0.95},
        {"id": "ps11", "pillar": "Process", "element": "Security & Immigration",
         "item": "Jalur khusus (fast track, crew, disabilitas, diplomat) dikelola dengan baik dan tidak mengganggu alur penumpang reguler", "weight": 0.85},

        # 5. Boarding & Departure Processes
        {"id": "ps12", "pillar": "Process", "element": "Boarding & Departure",
         "item": "Proses boarding dilakukan secara terorganisir, tepat waktu, dan sesuai urutan yang telah diumumkan", "weight": 1.0},
        {"id": "ps13", "pillar": "Process", "element": "Boarding & Departure",
         "item": "Pengumuman di gate jelas, tepat waktu, dan memberikan informasi yang cukup bagi penumpang", "weight": 0.9},

        # 6. Baggage Claim & Post-Arrival Processes
        {"id": "ps14", "pillar": "Process", "element": "Baggage Claim & Arrival",
         "item": "Waktu tunggu bagasi di baggage claim dalam batas wajar dengan informasi status bagasi yang jelas dan akurat", "weight": 0.95},
        {"id": "ps15", "pillar": "Process", "element": "Baggage Claim & Arrival",
         "item": "Proses klaim bagasi hilang/rusak/terlambat jelas, responsif, dan mudah diakses oleh penumpang", "weight": 0.85},
    ]


def render_audit_form(context):
    st.markdown("## ✍️ My Inspection")
    
    # Check if we are in Edit mode
    editing_audit = st.session_state.get("editing_audit")
    editing_audit_id = st.session_state.get("editing_audit_id")
    
    if editing_audit:
        st.warning("✏️ **Mode Edit Aktif** — Perubahan akan langsung tersimpan saat klik tombol di bawah.")
        st.caption("Perubahan yang kamu buat akan langsung tersimpan saat kamu klik tombol submit di bawah.")
        if st.button("❌ Batalkan Edit & Kembali ke Mode Baru", type="secondary", key="cancel_edit_btn"):
            st.session_state.pop("editing_audit", None)
            st.session_state.pop("editing_audit_id", None)
            st.rerun()
    
    if not context["rag_ready"]:
        st.warning("⚠️ Knowledge Base belum siap. Beberapa rekomendasi mungkin kurang akurat.")
    
    # Use data from editing_audit if available, otherwise use sidebar context
    auditor_name = editing_audit.get("auditor", context['auditor']) if editing_audit else context['auditor']
    audit_date = editing_audit.get("audit_date", context['audit_date']) if editing_audit else context['audit_date']
    
    terminal_info = context.get("terminal", "")
    display_location = f"{context['airport']} - {terminal_info}" if terminal_info else context['airport']
    st.markdown(f"**{display_location}** • Inspector: **{auditor_name}** • {audit_date}")
    
    checklist = get_base_checklist_items()
    
    # Group by pillar
    pillars = ["People", "Process", "Premises"]
    pillar_data = {p: [item for item in checklist if item["pillar"] == p] for p in pillars}
    
    # Pre-fill defaults from editing inspection (used as widget defaults)
    prefilled_scores = {}
    prefilled_comments = {}
    if editing_audit:
        old_scores = editing_audit.get("scores", {})
        old_comments = editing_audit.get("comments", {})
        for item in checklist:
            prefilled_scores[item["id"]] = old_scores.get(item["id"], 4)
            prefilled_comments[item["id"]] = old_comments.get(item["id"], "")
    
    # === WRAP IN FORM: ini kunci agar tidak lag saat isi form ===
    with st.form(key="audit_form", clear_on_submit=False):
        st.caption("Isi semua skor & komentar di bawah. Klik tombol besar di paling bawah untuk menyimpan. Tidak akan ada loading di tengah pengisian.")
        
        scores = {}
        comments = {}
        evidence_files = {}

        pillar_icons = {"People": "👥", "Process": "⚙️", "Premises": "🏢"}

        # ========== PEOPLE PILLAR ==========
        st.markdown('<div class="pillar-section people-pillar">', unsafe_allow_html=True)

        # Custom prominent colored title for the main pillar (exactly like Panduan Detail style)
        st.markdown(
            '<div class="pillar-title-header people">👥 People</div>',
            unsafe_allow_html=True
        )

        with st.expander(" ", expanded=True):
            
            # Prominent colored Panduan Detail header (pillar-specific)
            st.markdown(
                '<div style="background: linear-gradient(90deg, #1E40AF 0%, #3B82F6 100%); '
                'color: white; padding: 12px 18px; border-radius: 10px; margin: 10px 0 6px 0; '
                'font-size: 1.05rem; font-weight: 700; box-shadow: 0 2px 8px rgba(30, 64, 175, 0.25); '
                'display: flex; align-items: center; gap: 10px;">'
                '📖 <span>Panduan Detail People Pillar</span>'
                '</div>',
                unsafe_allow_html=True
            )
            
            with st.expander("Klik untuk lihat panduan lengkap dari Playbook 4", expanded=False):
                st.caption("Standar diambil langsung dari CX Guidance for People Pillar.")

                with st.expander("1. General Wellness & Personal Hygiene"):
                    st.markdown("""
                    - Kebersihan badan: tidak ada bau badan, bau mulut, atau keringat berlebihan.
                    - Kuku pendek, bersih, dan rapi (tidak ada kotoran atau cat kuku mencolok).
                    - Wajah segar dan tidak berminyak berlebihan. Gigi bersih dan nafas segar.
                    """)

                with st.expander("2. Rambut & Personal Presentation"):
                    st.markdown("""
                    **Pria:** Rambut pendek & rapi (model Fade/Taper yang halus). Dilarang punk, spiky, skin-cropped berlebihan.
                    
                    **Wanita:** Rambut panjang disanggul rapi (Chignon/French Twist). Rambut pendek tidak menyentuh bahu. Warna natural.
                    
                    **Make-up & Parfum:** Natural dan profesional, tidak berlebihan. Parfum wangi ringan.
                    """)

                with st.expander("3. Seragam & Kelengkapan"):
                    st.markdown("""
                    - Seragam lengkap, rapi, disetrika, tidak kusut.
                    - Name tag/ID card terpasang benar dan terlihat jelas.
                    - Dasi/Scarf rapi, sepatu pantofel hitam formal bersih.
                    - Rok atau celana sesuai ketentuan panjangnya.
                    """)

                with st.expander("4. Aksesoris & Atribut"):
                    st.markdown("""
                    - Maksimal 5 titik aksesoris di seluruh tubuh.
                    - Cincin maksimal 1 buah, ukuran kecil.
                    - Kalung harus di dalam pakaian (tidak terlihat).
                    - Kerudung (jika memakai) rapi dan menutupi seluruh rambut + leher.
                    """)

                with st.expander("5. Sikap Tubuh & Ekspresi"):
                    st.markdown("""
                    - Berdiri tegak, bahu rileks, siap melayani.
                    - Menampilkan senyum tulus dan ramah.
                    - Kontak mata natural saat berbicara.
                    - Dilarang: membungkuk, silang tangan, wajah datar/cemberut.
                    """)

                with st.expander("6. Komunikasi & Etika Profesional"):
                    st.markdown("""
                    - Menggunakan dua tangan saat menerima/menyerahkan dokumen.
                    - Telapak tangan terbuka saat menunjuk arah.
                    - Tidak menggunakan ponsel/earphone di area publik.
                    - Tidak makan/minum di tempat terlarang saat berseragam.
                    """)

            # === Full People Checklist Rendering ===
            people_categories = {
                "General Wellness & Personal Hygiene": ["p01", "p02", "p03"],
                "Rambut & Personal Presentation": ["p04", "p05", "p06"],
                "Seragam & Kelengkapan": ["p07", "p08", "p09"],
                "Aksesoris & Atribut": ["p10", "p11"],
                "Sikap Tubuh & Ekspresi": ["p12", "p13"],
                "Komunikasi & Etika Profesional": ["p14", "p15"],
            }
            
            items_by_id = {item["id"]: item for item in pillar_data["People"]}
            
            def get_cat_avg(cat_ids):
                vals = []
                for iid in cat_ids:
                    val = st.session_state.get(f"score_{iid}")
                    if val is None:
                        val = prefilled_scores.get(iid, 4)
                    vals.append(val)
                return round(sum(vals) / len(vals), 1) if vals else 0

            for cat_name, id_list in people_categories.items():
                avg = get_cat_avg(id_list)
                stars = "★" * int(round(avg)) + "☆" * (5 - int(round(avg)))
                
                if avg >= 4.0:
                    score_class = "score-high"
                elif avg >= 3.0:
                    score_class = "score-medium"
                else:
                    score_class = "score-low"
                
                st.markdown(
                    f'<div class="category-card category-card-people">'
                    f'<span class="category-title">▸ {cat_name}</span>'
                    f'<span class="{score_class}">Rata-rata: {avg} {stars}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                for iid in id_list:
                    item = items_by_id[iid]
                    cur = st.session_state.get(f"score_{iid}", prefilled_scores.get(iid, 4))
                    
                    with st.expander(f"{item['item']} | Skor saat ini: {cur}", expanded=False):
                        st.markdown(f"**{item['item']}**")
                        st.caption(f"Elemen: **{item['element']}**")
                        
                        sc = st.select_slider("Skor", options=[1,2,3,4,5], value=cur, key=f"score_{iid}")
                        scores[iid] = sc
                        
                        comments[iid] = st.text_area(
                            "Komentar / Temuan", 
                            value=prefilled_comments.get(iid, ""), 
                            key=f"comment_{iid}", 
                            height=60
                        )
                        
                        evidence_files[iid] = st.file_uploader(
                            "Foto Bukti (opsional)", 
                            type=["jpg", "png"], 
                            key=f"ev_{iid}"
                        )

        st.markdown('</div>', unsafe_allow_html=True)  # close people-pillar colored wrapper

        # ========== PROCESS PILLAR ==========
        st.markdown('<div class="pillar-section process-pillar">', unsafe_allow_html=True)

        # Custom prominent colored title for the main pillar (exactly like Panduan Detail style)
        st.markdown(
            '<div class="pillar-title-header process">⚙️ Process</div>',
            unsafe_allow_html=True
        )

        with st.expander(" ", expanded=False):
            # Prominent colored Panduan Detail header (pillar-specific)
            st.markdown(
                '<div style="background: linear-gradient(90deg, #3730A3 0%, #6366F1 100%); '
                'color: white; padding: 12px 18px; border-radius: 10px; margin: 10px 0 6px 0; '
                'font-size: 1.05rem; font-weight: 700; box-shadow: 0 2px 8px rgba(55, 48, 163, 0.25); '
                'display: flex; align-items: center; gap: 10px;">'
                '📖 <span>Panduan Detail Process Pillar</span>'
                '</div>',
                unsafe_allow_html=True
            )
            
            with st.expander("Klik untuk lihat panduan lengkap dari Playbook 2", expanded=False):
                st.caption("Standar diambil langsung dari CX Guidance for Process Pillar.")

                with st.expander("1. Pre-Journey & Digital Processes"):
                    st.markdown("""
                    - Website dan aplikasi mobile mudah digunakan dan informatif.
                    - Proses check-in online berjalan lancar.
                    - Informasi penerbangan real-time akurat.
                    - Notifikasi dan pengingat dikirim tepat waktu.
                    """)

                with st.expander("2. Arrival & Ground Transportation Processes"):
                    st.markdown("""
                    - Proses turun dari pesawat dan menuju terminal efisien.
                    - Transportasi darat (taksi, bus, kereta) mudah diakses.
                    - Antrian di imigrasi/keamanan dikelola dengan baik.
                    - Staf ramah dan membantu penumpang yang bingung.
                    """)

                with st.expander("3. Check-in & Baggage Processes"):
                    st.markdown("""
                    - Antrian check-in tertata rapi dan bergerak cepat.
                    - Staf check-in sopan, efisien, dan informatif.
                    - Proses penimbangan dan penanganan bagasi hati-hati.
                    - Self check-in kiosk berfungsi dengan baik.
                    """)

                with st.expander("4. Security, Immigration & Customs Processes"):
                    st.markdown("""
                    - Proses pemeriksaan keamanan dilakukan dengan sopan.
                    - Antrian security, imigrasi, dan customs dikelola dengan baik.
                    - Petugas ramah dan memberikan penjelasan jika diperlukan.
                    - Proses berjalan lancar tanpa penumpukan berlebihan.
                    """)

                with st.expander("5. Boarding & Departure Processes"):
                    st.markdown("""
                    - Proses boarding tertib dan sesuai jadwal.
                    - Pengumuman jelas dan mudah dipahami.
                    - Staf boarding membantu penumpang prioritas dengan baik.
                    - Gate area nyaman dan tidak terlalu padat.
                    """)

                with st.expander("6. Baggage Claim & Post-Arrival Processes"):
                    st.markdown("""
                    - Bagasi datang tepat waktu dan sesuai penerbangan.
                    - Area klaim bagasi bersih dan tidak ramai berlebihan.
                    - Informasi mengenai bagasi yang tertinggal jelas.
                    - Proses keluar terminal berjalan lancar.
                    """)

            # Full Process rendering
            process_categories = {
                "Pre-Journey & Digital Processes": ["ps01", "ps02", "ps03"],
                "Arrival & Ground Transportation Processes": ["ps04", "ps05"],
                "Check-in & Baggage Processes": ["ps06", "ps07", "ps08"],
                "Security, Immigration & Customs Processes": ["ps09", "ps10", "ps11"],
                "Boarding & Departure Processes": ["ps12", "ps13"],
                "Baggage Claim & Post-Arrival Processes": ["ps14", "ps15"],
            }
            
            items_by_id = {item["id"]: item for item in pillar_data["Process"]}
            
            def get_proc_avg(cat_ids):
                vals = []
                for iid in cat_ids:
                    val = st.session_state.get(f"score_{iid}")
                    if val is None:
                        val = prefilled_scores.get(iid, 4)
                    vals.append(val)
                return round(sum(vals) / len(vals), 1) if vals else 0

            for cat_name, id_list in process_categories.items():
                avg = get_proc_avg(id_list)
                stars = "★" * int(round(avg)) + "☆" * (5 - int(round(avg)))
                
                if avg >= 4.0:
                    score_class = "score-high"
                elif avg >= 3.0:
                    score_class = "score-medium"
                else:
                    score_class = "score-low"
                
                st.markdown(
                    f'<div class="category-card category-card-process">'
                    f'<span class="category-title">▸ {cat_name}</span>'
                    f'<span class="{score_class}">Rata-rata: {avg} {stars}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                for iid in id_list:
                    item = items_by_id[iid]
                    cur = st.session_state.get(f"score_{iid}", prefilled_scores.get(iid, 4))
                    
                    with st.expander(f"{item['item']} | Skor saat ini: {cur}", expanded=False):
                        st.markdown(f"**{item['item']}**")
                        st.caption(f"Elemen: **{item['element']}**")
                        sc = st.select_slider("Skor", [1,2,3,4,5], value=cur, key=f"score_{iid}")
                        scores[iid] = sc
                        comments[iid] = st.text_area("Komentar", value=prefilled_comments.get(iid,""), key=f"comment_{iid}", height=60)
                        evidence_files[iid] = st.file_uploader("Foto Bukti", type=["jpg","png"], key=f"ev_{iid}")

        st.markdown('</div>', unsafe_allow_html=True)  # close process-pillar colored wrapper

        # ========== PREMISES PILLAR ==========
        st.markdown('<div class="pillar-section premises-pillar">', unsafe_allow_html=True)

        # Custom prominent colored title for the main pillar (exactly like Panduan Detail style)
        st.markdown(
            '<div class="pillar-title-header premises">🏢 Premises</div>',
            unsafe_allow_html=True
        )

        with st.expander(" ", expanded=False):
            # Prominent colored Panduan Detail header (pillar-specific)
            st.markdown(
                '<div style="background: linear-gradient(90deg, #065F46 0%, #10B981 100%); '
                'color: white; padding: 12px 18px; border-radius: 10px; margin: 10px 0 6px 0; '
                'font-size: 1.05rem; font-weight: 700; box-shadow: 0 2px 8px rgba(6, 95, 70, 0.25); '
                'display: flex; align-items: center; gap: 10px;">'
                '📖 <span>Panduan Detail Premises Pillar</span>'
                '</div>',
                unsafe_allow_html=True
            )
            
            with st.expander("Klik untuk lihat panduan lengkap dari Playbook 3", expanded=False):
                st.caption("Standar diambil langsung dari CX Guidance for Premises Pillar.")

                with st.expander("1. Public Spaces & Outdoor Areas"):
                    st.markdown("""
                    - Kebersihan lantai, dinding, dan langit-langit (tidak ada noda, debu, atau sampah).
                    - Pencahayaan cukup dan merata, tidak ada lampu mati atau redup berlebihan.
                    - Tidak ada bau tidak sedap, asap rokok, atau bau makanan yang mengganggu.
                    - Kursi dan area duduk dalam kondisi baik, tidak rusak atau kotor.
                    """)

                with st.expander("2. Parking & Curbside / Arrival Facilities"):
                    st.markdown("""
                    - Area drop-off dan pick-up tertata rapi, tidak ada kendaraan parkir sembarangan.
                    - Trotoar dan jalur pejalan kaki bersih serta mudah dilalui.
                    - Tanda petunjuk arah ke terminal jelas dan tidak rusak.
                    - Area taksi dan transportasi umum terorganisir dengan baik.
                    """)

                with st.expander("3. Customer Facilities"):
                    st.markdown("""
                    - Toilet bersih, kering, tidak ada bau, dan perlengkapan lengkap (sabun, tissue, hand dryer).
                    - Musholla / prayer room nyaman, bersih, dan memiliki fasilitas wudhu yang memadai.
                    - Area makan dan retail bersih serta memiliki tempat sampah yang cukup.
                    - WiFi berfungsi dengan baik dan mudah diakses.
                    """)

                with st.expander("4. Special Needs & Inclusive Facilities"):
                    st.markdown("""
                    - Tersedia jalur khusus untuk penyandang disabilitas (ramp, lift, toilet accessible).
                    - Tanda petunjuk untuk tuna netra (tactile paving) dalam kondisi baik.
                    - Tersedia kursi roda dan bantuan staf jika diperlukan.
                    - Area untuk keluarga dengan anak kecil (nursing room) bersih dan nyaman.
                    """)

                with st.expander("5. Safety & Security Facilities"):
                    st.markdown("""
                    - CCTV berfungsi dan area terasa aman.
                    - Tanda darurat, jalur evakuasi, dan alat pemadam api terlihat jelas.
                    - Tidak ada area gelap atau rawan di dalam terminal.
                    - Prosedur keamanan dilakukan dengan sopan dan efisien.
                    """)

                with st.expander("6. Wayfinding, Signage & Information Systems"):
                    st.markdown("""
                    - Papan petunjuk arah (signage) jelas, konsisten, dan mudah dibaca.
                    - Peta terminal dan informasi penerbangan (FIDS) akurat dan up-to-date.
                    - Informasi dalam bahasa Indonesia dan Inggris tersedia.
                    - Staf informasi mudah ditemukan dan membantu dengan ramah.
                    """)

            # Full Premises rendering
            premises_categories = {
                "Public Spaces & Outdoor Areas": ["pr01", "pr02", "pr03"],
                "Parking & Curbside / Arrival Facilities": ["pr04", "pr05", "pr06"],
                "Customer Facilities": ["pr07", "pr08", "pr09"],
                "Special Needs & Inclusive Facilities": ["pr10", "pr11", "pr12"],
                "Safety & Security Facilities": ["pr13", "pr14", "pr15"],
                "Wayfinding, Signage & Information Systems": ["pr16", "pr17", "pr18"],
            }
            
            items_by_id = {item["id"]: item for item in pillar_data["Premises"]}
            
            def get_prem_avg(cat_ids):
                vals = []
                for iid in cat_ids:
                    val = st.session_state.get(f"score_{iid}")
                    if val is None:
                        val = prefilled_scores.get(iid, 4)
                    vals.append(val)
                return round(sum(vals) / len(vals), 1) if vals else 0

            for cat_name, id_list in premises_categories.items():
                avg = get_prem_avg(id_list)
                stars = "★" * int(round(avg)) + "☆" * (5 - int(round(avg)))
                
                if avg >= 4.0:
                    score_class = "score-high"
                elif avg >= 3.0:
                    score_class = "score-medium"
                else:
                    score_class = "score-low"
                
                st.markdown(
                    f'<div class="category-card category-card-premises">'
                    f'<span class="category-title">▸ {cat_name}</span>'
                    f'<span class="{score_class}">Rata-rata: {avg} {stars}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                for iid in id_list:
                    item = items_by_id[iid]
                    cur = st.session_state.get(f"score_{iid}", prefilled_scores.get(iid, 4))
                    
                    with st.expander(f"{item['item']} | Skor saat ini: {cur}", expanded=False):
                        st.markdown(f"**{item['item']}**")
                        st.caption(f"Elemen: **{item['element']}**")
                        sc = st.select_slider("Skor", [1,2,3,4,5], value=cur, key=f"score_{iid}")
                        scores[iid] = sc
                        comments[iid] = st.text_area("Komentar", value=prefilled_comments.get(iid,""), key=f"comment_{iid}", height=60)
                        evidence_files[iid] = st.file_uploader("Foto Bukti", type=["jpg","png"], key=f"ev_{iid}")
            
            # [Sisa kode lama yang rusak sudah dibersihkan di sini]
        
        st.markdown('</div>', unsafe_allow_html=True)  # close premises-pillar colored wrapper

        # Live ringkasan di dalam form (hanya update saat submit)
        pillar_scores = {}
        for pillar in pillars:
            pillar_items = pillar_data[pillar]
            total = sum(scores.get(i['id'], 3) * i.get('weight', 1.0) for i in pillar_items)
            max_possible = sum(5 * i.get('weight', 1.0) for i in pillar_items)
            pillar_scores[pillar] = round((total / max_possible) * 5, 2) if max_possible > 0 else 0
        
        overall = round(sum(pillar_scores.values()) / 3, 2)
        
        st.markdown("### 📊 Ringkasan Skor (akan tersimpan saat submit)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("People", f"{pillar_scores['People']:.2f}")
        col2.metric("Process", f"{pillar_scores['Process']:.2f}")
        col3.metric("Premises", f"{pillar_scores['Premises']:.2f}")
        col4.metric("OVERALL", f"{overall:.2f}")
        
        is_editing = bool(editing_audit)
        button_label = "💾 Update Inspection Sekarang" if is_editing else "💾 Simpan Inspection Lengkap & Generate Rekomendasi"
        
        submitted = st.form_submit_button(button_label, type="primary", width="stretch")
    
    # === LOGIC SETELAH SUBMIT (hanya jalan sekali saat tombol besar diklik) ===
    if submitted:
        if is_editing:
            updated_record = {
                "scores": scores,
                "comments": comments,
                "pillar_scores": pillar_scores,
                "overall_score": overall,
                "recommendations": generate_recommendations(checklist, scores, comments, pillar_scores),
            }
            success = update_audit(editing_audit_id, updated_record)
            if success:
                st.success("✅ Inspection berhasil diperbarui!")
                st.session_state.pop("editing_audit", None)
                st.session_state.pop("editing_audit_id", None)
                get_cached_audits.clear()
                st.rerun()
            else:
                st.error("Gagal memperbarui inspection.")
        else:
            audit_record = {
                "audit_id": generate_audit_id(),
                "airport": context["airport"],
                "terminal": context.get("terminal", ""),
                "auditor": context["auditor"],
                "audit_date": context["audit_date"],
                "checklist_items": checklist,
                "scores": scores,
                "comments": comments,
                "pillar_scores": pillar_scores,
                "overall_score": overall,
                "recommendations": generate_recommendations(checklist, scores, comments, pillar_scores),
            }
            
            # Save evidence files safely (skip items with no upload)
            evidence_dir = get_reports_dir() / "evidence" / audit_record["audit_id"]
            saved_any_evidence = False

            for item_id, file in evidence_files.items():
                if file is not None:
                    if not saved_any_evidence:
                        evidence_dir.mkdir(parents=True, exist_ok=True)
                        saved_any_evidence = True
                    ext = file.name.split(".")[-1]
                    dest = evidence_dir / f"{item_id}.{ext}"
                    with open(dest, "wb") as f:
                        f.write(file.getbuffer())

            if saved_any_evidence:
                audit_record["evidence_folder"] = str(evidence_dir)
            
            audit_id = save_audit(audit_record)
            get_cached_audits.clear()
            st.success("✅ Inspection berhasil disimpan!")
            st.balloons()
            st.session_state["last_audit_id"] = audit_id
            st.rerun()


def generate_recommendations(checklist, scores, comments, pillar_scores):
    """Simple rule-based recommendation generator (can be upgraded with LLM later)."""
    recs = []
    for item in checklist:
        score = scores.get(item["id"], 3)
        if score <= 2:
            recs.append({
                "pillar": item["pillar"],
                "item": item["item"],
                "score": score,
                "suggestion": f"Perlu perbaikan mendesak pada: {item['element']}. " + 
                              (comments.get(item["id"]) or "Lakukan pelatihan ulang dan monitoring ketat.")
            })
        elif score == 3:
            recs.append({
                "pillar": item["pillar"],
                "item": item["item"],
                "score": score,
                "suggestion": f"Tingkatkan konsistensi pada {item['element']}."
            })
    return recs


@st.cache_data(ttl=5)
def get_cached_audits():
    """Cache the list of all inspections for a short time (5s) to improve performance while avoiding stale data after deletions."""
    return list_all_audits()


@st.cache_data(ttl=60)
def compute_element_scores(audits: list) -> dict:
    """
    Compute average score per element (e.g., "People - Safe Space").
    Used for more detailed breakdown in Dashboard.
    Cached for 60 seconds to improve performance.
    """
    from collections import defaultdict

    element_totals = defaultdict(float)
    element_counts = defaultdict(int)

    for audit in audits:
        checklist = audit.get("checklist_items", [])
        scores = audit.get("scores", {})

        for item in checklist:
            pillar = item.get("pillar", "")
            element = item.get("element", "")
            item_id = item.get("id")
            weight = item.get("weight", 1.0)

            if item_id in scores:
                key = f"{pillar} - {element}"
                element_totals[key] += scores[item_id] * weight
                element_counts[key] += weight

    element_averages = {}
    for key in element_totals:
        if element_counts[key] > 0:
            element_averages[key] = round(element_totals[key] / element_counts[key], 2)

    return element_averages


# =============================================================================
# KNOWLEDGE BASE (Real RAG)
# =============================================================================

def render_knowledge_base(context):
    st.markdown("## 📚 Knowledge Base — Tanya Playbook CX InJourney")

    # AI / RAG features have been disabled (per user request "hapus saja Ai nya")
    # to keep the app lightweight and fast for the main Daily Service QC use case.
    # No attempt to load heavy packages (chromadb etc.) is made.
    st.info("Fitur Knowledge Base (AI/RAG) dinonaktifkan untuk menjaga performa dan stabilitas aplikasi.")
    st.markdown(
        "Fokus utama aplikasi ini adalah **📋 Daily Service QC (Harian)** di tab terakhir, "
        "yang mendukung pencatatan cepat, foto bukti, dan export PDF/Excel yang rapih."
    )
    st.caption("Jika suatu saat butuh fitur AI lagi, kita bisa tambahkan kembali dengan dependencies terpisah.")
    return


# =============================================================================
# LAPORAN & EXPORT
# =============================================================================

def render_reports_tab(context):
    st.markdown("## 📄 Laporan & Riwayat Inspection")
    
    audits = get_cached_audits()
    
    if not audits:
        st.info("Belum ada inspection yang disimpan. Buat inspection pertama di tab 'My Inspection'.")
        return

    tab_list, tab_detail, tab_manage = st.tabs([
        "📋 Daftar & Cari Inspection", 
        "📊 Detail & Export", 
        "🗑️ Manajemen Data"
    ])

    with tab_list:
        search_term = st.text_input("🔍 Cari Inspection (Lokasi atau Inspector)", placeholder="Ketik untuk filter...")
        filtered_audits = [a for a in audits if search_term.lower() in str(a).lower()] if search_term else audits

        if filtered_audits:
            df = pd.DataFrame(filtered_audits)
            if "terminal" in df.columns:
                df["Lokasi"] = df.apply(
                    lambda row: f"{row['airport']} - {row['terminal']}" if row.get("terminal") else row["airport"], 
                    axis=1
                )
                display_cols = ["Lokasi", "overall_score", "auditor", "audit_date"]
                df_display = df[display_cols].rename(columns={
                    "auditor": "Inspector",
                    "overall_score": "Overall Score",
                    "audit_date": "Inspection Date"
                })
                st.dataframe(df_display, width="stretch", hide_index=True)
            else:
                st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.warning("Tidak ada inspection yang cocok dengan pencarian.")

    with tab_detail:
        # Build friendly options for the selectbox (hide raw technical ID from user)
        id_to_label = {}
        options = []
        for a in filtered_audits:
            loc = f"{a.get('airport', '')}"
            if a.get("terminal"):
                loc += f" - {a['terminal']}"
            date = a.get("audit_date", "")
            score = a.get("overall_score", 0)
            label = f"{loc} | {date} | Skor: {score}"
            id_to_label[label] = a["audit_id"]
            options.append(label)

        selected_label = st.selectbox("Pilih Inspection untuk Detail/Export", options) if options else None
        selected_id = id_to_label.get(selected_label) if selected_label else None
        
        if selected_id:
            audit = load_audit(selected_id)
            if audit:
                loc = f"{audit.get('airport', '')}"
                if audit.get("terminal"):
                    loc += f" - {audit['terminal']}"
                date_str = audit.get("audit_date", "")
                st.markdown(f"### Detail Inspection — {loc} ({date_str})")

                st.markdown("**Skor per Elemen**")
                elem_scores = compute_element_scores([audit])
                if elem_scores:
                    elem_df_single = pd.DataFrame(
                        list(elem_scores.items()), 
                        columns=["Elemen", "Skor"]
                    ).sort_values("Skor", ascending=False)
                    st.dataframe(elem_df_single, width="stretch", hide_index=True)
                else:
                    st.caption("Data elemen tidak tersedia.")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("⬇️ Export PDF Profesional", width="stretch", key=f"export_pdf_{selected_id}"):
                        try:
                            pdf_bytes = generate_pdf_report(audit)
                            st.download_button("Download Laporan PDF", data=pdf_bytes, file_name=f"InJourney_CX_Inspection_{selected_id}.pdf", mime="application/pdf")
                        except Exception as e:
                            st.error(f"Gagal membuat PDF: {e}")
                with c2:
                    if st.button("⬇️ Export Excel Detail", width="stretch", key=f"export_excel_{selected_id}"):
                        try:
                            excel_bytes = generate_excel_report(audit)
                            st.download_button("Download Laporan Excel", data=excel_bytes, file_name=f"InJourney_CX_Inspection_{selected_id}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        except Exception as e:
                            st.error(f"Gagal membuat Excel: {e}")
        else:
            st.info("Pilih inspection dari daftar di tab pertama.")

    with tab_manage:
        st.caption("⚠️ Fitur ini bersifat destruktif. Gunakan dengan hati-hati.")

        enable_fast_delete = st.checkbox("Tampilkan tombol hapus cepat", key="enable_fast_delete")
        if enable_fast_delete:
            for audit in filtered_audits:
                filepath = AUDITS_DIR / f"{audit['audit_id']}.json"
                if not filepath.exists():
                    continue  # Skip audits that no longer exist on disk

                col1, col2, col3 = st.columns([5, 2, 1])
                with col1:
                    loc = f"{audit['airport']} - {audit['terminal']}" if audit.get("terminal") else audit["airport"]
                    st.write(f"**{loc}**")
                with col2:
                    st.write(f"Skor: {audit['overall_score']} | {audit['audit_date']}")
                with col3:
                    if st.button("🗑️ Hapus", key=f"fast_del_{audit['audit_id']}", type="secondary"):
                        if delete_audit(audit['audit_id']):
                            get_cached_audits.clear()
                            st.success(f"✅ Inspection berhasil dihapus")
                            st.rerun()
                        else:
                            st.error("Gagal menghapus file. Coba refresh halaman.")
        else:
            st.info("Centang untuk menampilkan tombol hapus cepat.")

        st.markdown("---")
        with st.expander("🔥 Mode Super Cepat (Hapus langsung + auto backup) — Klik untuk membuka daftar", expanded=False):
            st.warning("Perhatian: tombol di bawah langsung menghapus tanpa konfirmasi tambahan. Data akan di-backup otomatis.")
            if not filtered_audits:
                st.info("Tidak ada inspection.")
            else:
                for audit in filtered_audits:
                    filepath = AUDITS_DIR / f"{audit['audit_id']}.json"
                    if not filepath.exists():
                        continue

                    col1, col2 = st.columns([6, 1])
                    with col1:
                        loc = f"{audit['airport']} - {audit['terminal']}" if audit.get("terminal") else audit["airport"]
                        st.write(f"{loc}")
                    with col2:
                        if st.button("🔥 Hapus Langsung", key=f"super_del_{audit['audit_id']}", type="primary"):
                            backup_path = create_backup(audit['audit_id'])
                            if delete_audit(audit['audit_id']):
                                get_cached_audits.clear()
                                st.success(f"✅ Dihapus. Backup tersimpan.")
                                st.rerun()
                            else:
                                st.error("Gagal menghapus file.")

def generate_pdf_report(audit: dict) -> bytes:
    """
    Professional InJourney-branded PDF Report.
    Includes header, inspection metadata, pillar scores table, detailed findings,
    recommendations, and signature area.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
        HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=1.6*cm, 
        leftMargin=1.6*cm,
        topMargin=1.4*cm, 
        bottomMargin=1.8*cm
    )

    # Colors
    IJ_BLUE = colors.HexColor("#003366")
    IJ_TEAL = colors.HexColor("#00A8A8")
    IJ_LIGHT_TEAL = colors.HexColor("#E0F7F7")
    IJ_GRAY = colors.HexColor("#64748B")

    styles = getSampleStyleSheet()

    # Custom styles - centered titles (user request)
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=IJ_BLUE,
        spaceAfter=2*mm,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=IJ_TEAL,
        alignment=TA_CENTER,
        spaceAfter=3*mm,
        fontName='Helvetica-Bold'
    )

    header_style = ParagraphStyle(
        'HeaderInfo',
        parent=styles['Normal'],
        fontSize=9,
        textColor=IJ_GRAY,
        spaceAfter=1*mm
    )

    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=IJ_BLUE,
        spaceBefore=4*mm,
        spaceAfter=2*mm,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )

    story = []

    # ========== HEADER: Logo di kanan atas, teks tetap CENTER (sesuai permintaan) ==========
    logo_path = Path(__file__).parent / "assets" / "injourney_airports_logo.png"
    if logo_path.exists():
        from reportlab.platypus import Image
        from reportlab.lib.units import cm
        logo_img = Image(str(logo_path), width=4.5*cm, height=1.35*cm)

        # Logo di kanan atas
        logo_table = Table([[logo_img]], colWidths=[17*cm])
        logo_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(logo_table)
        story.append(Spacer(1, 3*mm))

    # Judul tetap CENTER (seperti sebelumnya)
    story.append(Paragraph("INJOURNEY AIRPORTS", title_style))
    story.append(Paragraph("LAPORAN INSPECTION CUSTOMER EXPERIENCE QUALITY CONTROL", subtitle_style))
    
    # Horizontal line
    story.append(HRFlowable(width="100%", thickness=2, color=IJ_TEAL, spaceAfter=4*mm))

    # Inspection Info Box - sesuai permintaan: Inspector → Tanggal (dengan hari) → Bandara/Terminal (paling bawah)
    location = audit.get("airport", "-")
    if audit.get("terminal"):
        location += f" - {audit['terminal']}"

    # Format tanggal + nama hari dalam Bahasa Indonesia
    raw_date = audit.get("audit_date", "")
    formatted_date = raw_date
    try:
        from datetime import datetime
        if raw_date:
            dt = datetime.strptime(str(raw_date), "%Y-%m-%d")
            hari_id = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
            bulan_id = [
                "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                "Juli", "Agustus", "September", "Oktober", "November", "Desember"
            ]
            nama_hari = hari_id[dt.weekday()]
            nama_bulan = bulan_id[dt.month - 1]
            formatted_date = f"{nama_hari}, {dt.day} {nama_bulan} {dt.year}"
    except Exception:
        formatted_date = raw_date or "-"

    info_data = [
        [Paragraph("<b>Inspector</b>", normal_style), audit.get("auditor", "-")],
        [Paragraph("<b>Tanggal</b>", normal_style), formatted_date],
        [Paragraph("<b>Bandara / Terminal</b>", normal_style), location],
    ]
    
    info_table = Table(info_data, colWidths=[4.5*cm, 12*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), IJ_LIGHT_TEAL),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#1E293B")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 5*mm))

    # ========== OVERALL SCORE ==========
    overall = audit.get("overall_score", 0)
    score_color = "#10B981" if overall >= 4.0 else ("#F59E0B" if overall >= 3.0 else "#EF4444")
    
    story.append(Paragraph("SKOR KESELURUHAN", section_style))
    
    overall_data = [
        [Paragraph("<b>OVERALL CX SCORE</b>", ParagraphStyle('', fontSize=12, textColor=colors.white, alignment=TA_CENTER)),
         Paragraph(f"<b>{overall:.2f}</b> / 5.00", ParagraphStyle('', fontSize=16, textColor=colors.white, alignment=TA_CENTER, fontName='Helvetica-Bold'))]
    ]
    overall_table = Table(overall_data, colWidths=[9*cm, 7.5*cm])
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(score_color)),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(overall_table)
    story.append(Spacer(1, 4*mm))

    # ========== PILLAR SCORES ==========
    story.append(Paragraph("SKOR PER PILAR", section_style))
    
    pillar_scores = audit.get("pillar_scores", {})
    pillar_data = [
        ["Pilar", "Skor", "Kategori"],
    ]
    for pillar in ["People", "Process", "Premises"]:
        score = pillar_scores.get(pillar, 0)
        if score >= 4.2:
            kategori = "Sangat Baik"
        elif score >= 3.5:
            kategori = "Baik"
        elif score >= 2.5:
            kategori = "Cukup"
        else:
            kategori = "Perlu Perbaikan"
        pillar_data.append([pillar, f"{score:.2f}", kategori])
    
    pillar_table = Table(pillar_data, colWidths=[5*cm, 3*cm, 8.5*cm])
    pillar_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), IJ_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),   # Skor column - CENTER
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),   # Kategori column - CENTER
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0, -1), (-1, -1), IJ_LIGHT_TEAL),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(pillar_table)
    story.append(Spacer(1, 5*mm))

    # ========== RECOMMENDATIONS ==========
    recommendations = audit.get("recommendations", [])
    if recommendations:
        story.append(Paragraph("REKOMENDASI PERBAIKAN", section_style))
        
        rec_data = [["No", "Pilar", "Rekomendasi"]]
        for idx, r in enumerate(recommendations[:10], 1):
            rec_data.append([
                str(idx),
                r.get("pillar", "-"),
                Paragraph(r.get("suggestion", "-"), normal_style)
            ])
        
        rec_table = Table(rec_data, colWidths=[1*cm, 2.2*cm, 13.3*cm])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), IJ_TEAL),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(rec_table)
    
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=IJ_GRAY, spaceAfter=4*mm))

    # ========== LAMPIRAN: STANDAR CX PLAYBOOK YANG RELEVAN ==========
    pillar_scores = audit.get("pillar_scores", {})
    weak_pillars = [p for p, s in pillar_scores.items() if s < 4.0]

    if weak_pillars:
        story.append(Paragraph("LAMPIRAN: STANDAR CX PLAYBOOK YANG RELEVAN", section_style))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(
            "Berikut adalah standar yang seharusnya dipenuhi berdasarkan CX Playbook InJourney. "
            "Standar ini menjadi acuan perbaikan untuk area yang dinilai masih perlu ditingkatkan.",
            normal_style
        ))
        story.append(Spacer(1, 4*mm))

        for pillar in weak_pillars:
            standards = PLAYBOOK_STANDARDS.get(pillar, {})
            if standards:
                story.append(Paragraph(f"<b>{pillar}</b>", normal_style))
                story.append(Spacer(1, 2*mm))
                for cat, text in standards.items():
                    story.append(Paragraph(f"<b>• {cat}</b>", normal_style))
                    story.append(Paragraph(text, normal_style))
                    story.append(Spacer(1, 3*mm))
                story.append(Spacer(1, 3*mm))

    # ========== SIGNATURE AREA (tanpa Inspector, Supervisor/Manager, dan Tanggal) ==========
    sig_data = [
        ["_______________________________"],
    ]
    
    sig_table = Table(sig_data, colWidths=[16*cm])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('TEXTCOLOR', (0, 0), (-1, -1), IJ_GRAY),
    ]))
    story.append(sig_table)

    # Footer note
    story.append(Spacer(1, 6*mm))
    footer_style = ParagraphStyle('Footer', fontSize=7, textColor=IJ_GRAY, alignment=TA_CENTER)
    story.append(Paragraph(
        "Dokumen ini dibuat secara otomatis oleh InJourney Airports CX Quality Control System<br/>"
        "Berdasarkan InJourney Airports Customer Experience Transformation Concept Playbooks",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# (duplicate generate_daily_qc_excel removed - authoritative version is after main())



def generate_daily_qc_pdf(daily_data: dict) -> bytes:
    """
    Modern, professional, and audit-ready PDF for Daily Service QC.
    Clean layout with InJourney branding, metrics, color-coded tables, and clear sections.
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
        HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    buffer = BytesIO()

    # Use A4 portrait for professional daily report
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=1.8*cm,
        bottomMargin=1.5*cm
    )

    # Brand Colors (matching the app)
    IJ_BLUE = colors.HexColor("#003366")
    IJ_TEAL = colors.HexColor("#00A8A8")
    IJ_LIGHT = colors.HexColor("#E0F7F7")
    IJ_GRAY = colors.HexColor("#64748B")
    IJ_DARK = colors.HexColor("#1E293B")
    GREEN = colors.HexColor("#10B981")
    YELLOW = colors.HexColor("#F59E0B")
    RED = colors.HexColor("#EF4444")

    styles = getSampleStyleSheet()

    # Custom modern styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=IJ_BLUE,
        alignment=TA_CENTER,
        spaceAfter=2*mm,
        fontName='Helvetica-Bold',
        leading=20
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=IJ_TEAL,
        alignment=TA_CENTER,
        spaceAfter=4*mm,
        fontName='Helvetica-Bold'
    )

    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=IJ_BLUE,
        spaceBefore=5*mm,
        spaceAfter=2*mm,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=IJ_DARK
    )

    small_style = ParagraphStyle(
        'Small',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        textColor=IJ_GRAY
    )

    story = []

    # Header: Logo di pojok kanan atas (elegan), teks di bawahnya center
    logo_path = Path(__file__).parent / "assets" / "injourney_airports_logo.png"
    if logo_path.exists():
        from reportlab.platypus import Image
        from reportlab.lib.units import cm
        logo_img = Image(str(logo_path), width=4.2*cm, height=1.25*cm)

        # Logo di kanan atas
        logo_table = Table([[logo_img]], colWidths=[17*cm])
        logo_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(logo_table)
        story.append(Spacer(1, 2*mm))

    # Removed "INJOURNEY AIRPORTS" text as requested
    # Main title enlarged: Service Quality Control System
    story.append(Paragraph("<b>Service Quality Control System</b>", title_style))
    story.append(Paragraph("Soekarno-Hatta International Airport", ParagraphStyle('Airport', parent=subtitle_style, fontSize=12, spaceAfter=3*mm)))
    story.append(HRFlowable(width="100%", thickness=1, color=IJ_TEAL, spaceAfter=4*mm))

    # Subtitle
    story.append(Paragraph("LAPORAN HARIAN SERVICE QUALITY CONTROL", subtitle_style))
    story.append(Paragraph("Daily Operational Report — Siap untuk Audit", ParagraphStyle('SubReport', parent=subtitle_style, fontSize=9, spaceAfter=2*mm)))
    story.append(HRFlowable(width="100%", thickness=1, color=IJ_TEAL, spaceAfter=4*mm))

    # Metadata - centered for neat modern look
    meta = [
        ["Tanggal", daily_data.get("date", "-")],
        ["Inspector", daily_data.get("inspector", "-")],
        ["Lokasi", daily_data.get("location", "-")],
        ["Waktu Generate", datetime.now().strftime("%d %b %Y %H:%M")]
    ]
    meta_table = Table(meta, colWidths=[3.5*cm, 13*cm])
    meta_table.hAlign = 'CENTER'
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), IJ_TEAL),
        ('TEXTCOLOR', (1, 0), (1, -1), IJ_DARK),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('LINEBELOW', (0, 0), (-1, -2), 0.3, colors.HexColor("#E2E8F0")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5*mm))

    fc = daily_data.get("facility_checks", [])
    comps = daily_data.get("complaints", [])
    iss = daily_data.get("issues", [])

    # === KEY METRICS ===
    story.append(Paragraph("RINGKASAN KINERJA HARI INI", section_style))

    # Calculate metrics
    total_areas = len(fc)
    baik = len([x for x in fc if x.get("status") == "Baik"])
    readiness = round((baik / total_areas * 100)) if total_areas > 0 else 0

    open_issues = len([x for x in iss if x.get("status") != "Closed"])
    resolved_complaints = len([x for x in comps if x.get("status") == "Resolved"])

    metrics_data = [
        ["Facility Readiness", f"{readiness}%", f"{baik}/{total_areas} area Baik"],
        ["Keluhan Masuk", str(len(comps)), f"{resolved_complaints} sudah ditangani"],
        ["Issue Terbuka", str(open_issues), "Memerlukan tindak lanjut"],
    ]

    metrics_table = Table(metrics_data, colWidths=[5*cm, 3*cm, 8.5*cm])
    metrics_table.hAlign = 'CENTER'
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), IJ_LIGHT),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (1, 0), (1, 0), GREEN if readiness >= 80 else (YELLOW if readiness >= 60 else RED)),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 6*mm))

    # === FACILITY READINESS TABLE ===
    if fc:
        story.append(Paragraph("1. PENGECEKAN KESIAPAN FASILITAS", section_style))

        fac_data = [["No", "Area / Lokasi", "Status", "Catatan"]]
        for i, item in enumerate(fc, 1):
            status = item.get("status", "-")
            color = GREEN if status == "Baik" else (YELLOW if "Minor" in status else RED)
            fac_data.append([
                str(i),
                item.get("area", "-"),
                Paragraph(f'<font color="{color.hexval()}"><b>{status}</b></font>', normal_style),
                Paragraph(item.get("notes", "-")[:80] + ("..." if len(item.get("notes", "")) > 80 else ""), small_style)
            ])

        fac_table = Table(fac_data, colWidths=[0.7*cm, 4.5*cm, 2*cm, 6.5*cm])
        fac_table.hAlign = 'CENTER'
        fac_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), IJ_TEAL),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(fac_table)
        story.append(Spacer(1, 5*mm))

    # === COMPLAINTS TABLE ===
    if comps:
        story.append(Paragraph("2. LOG KELUHAN PELANGGAN", section_style))

        comp_data = [["No", "Area", "Kategori", "Deskripsi", "Status"]]
        for i, c in enumerate(comps, 1):
            status = c.get("status", "-")
            color = GREEN if status == "Resolved" else (YELLOW if "Progress" in status else RED)
            comp_data.append([
                str(i),
                c.get("area", "-")[:18],
                c.get("category", "-"),
                Paragraph(c.get("description", "-")[:60], small_style),
                Paragraph(f'<font color="{color.hexval()}"><b>{status}</b></font>', small_style)
            ])

        comp_table = Table(comp_data, colWidths=[0.7*cm, 3.2*cm, 2.8*cm, 7.5*cm, 2.5*cm])
        comp_table.hAlign = 'CENTER'
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), IJ_TEAL),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 5*mm))

    # === ISSUES TABLE ===
    if iss:
        story.append(Paragraph("3. ISSUES, ROOT CAUSE & TINDAK LANJUT", section_style))

        issue_data = [["No", "Area", "Masalah", "Root Cause", "Status", "PIC / Due"]]
        for i, item in enumerate(iss, 1):
            status = item.get("status", "-")
            color = GREEN if status == "Closed" else (YELLOW if "Progress" in status else RED)
            issue_data.append([
                str(i),
                item.get("area", "-")[:14],
                Paragraph(item.get("description", "-")[:45], small_style),
                item.get("root_cause", "-")[:18],
                Paragraph(f'<font color="{color.hexval()}"><b>{status}</b></font>', small_style),
                f"{item.get('pic', '-')[:10]}\n{item.get('due_date', '-')}"
            ])

        issue_table = Table(issue_data, colWidths=[0.7*cm, 2.8*cm, 5.5*cm, 3.2*cm, 2*cm, 3*cm])
        issue_table.hAlign = 'CENTER'
        issue_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), IJ_TEAL),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (-2, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(issue_table)

    # Lampiran Foto — more prominent and neatly centered for "rapih" requirement
    photo_items = [item for item in fc if item.get("photo_data")]
    if photo_items:
        story.append(Spacer(1, 5*mm))
        story.append(HRFlowable(width="60%", thickness=0.8, color=IJ_TEAL, spaceBefore=2*mm, spaceAfter=2*mm))
        story.append(Paragraph("<b>LAMPIRAN FOTO BUKTI</b>", ParagraphStyle('PhotoHeader', parent=section_style, alignment=TA_CENTER, fontSize=10)))
        story.append(Spacer(1, 2*mm))
        for idx, item in enumerate(photo_items, 1):
            try:
                img = Image(BytesIO(base64.b64decode(item["photo_data"])), width=5.2*cm, height=3.7*cm)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Paragraph(f"<i>Foto {idx}: {item.get('area', '')}</i>", ParagraphStyle('Pcap', parent=small_style, alignment=TA_CENTER)))
                story.append(Spacer(1, 3*mm))
            except:
                pass

    # Footer
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=IJ_GRAY))
    footer_text = f"""
    <font size="7" color="#64748B">
    Laporan ini dibuat secara otomatis oleh <b>InJourney Airports CX Quality Control System</b>.<br/>
    Data bersifat internal dan dapat digunakan sebagai bukti untuk keperluan audit internal/eksternal.<br/>
    Dicetak pada: {datetime.now().strftime("%d %B %Y pukul %H:%M")}
    </font>
    """
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', alignment=TA_CENTER, leading=10)))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# DAILY SERVICE QC (HARIAN)
# =============================================================================


def render_dashboard(context):
    """Main overview dashboard for CX Quality Control — restored modern look."""
    # Nice branded header
    st.markdown("""
    <div class="injourney-header">
        <h1>📊 CX Quality Control Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

    audits = get_cached_audits() or []

    # === TOP METRICS (styled cards) ===
    total = len(audits)

    if total > 0:
        avg_score = round(sum(a.get("overall_score", 0) for a in audits) / total, 2)
        this_month = sum(1 for a in audits if str(a.get("audit_date", ""))[:7] == datetime.now().strftime("%Y-%m"))
    else:
        avg_score = 0.0
        this_month = 0

    # Daily QC quick stats from session (if user has been using the harian module)
    daily = st.session_state.get("daily_qc", {})
    daily_fc = len(daily.get("facility_checks", []))
    daily_iss = len([i for i in daily.get("issues", []) if i.get("status") != "Closed"])
    daily_ready = 0
    if daily_fc > 0:
        baik = len([x for x in daily.get("facility_checks", []) if x.get("status") == "Baik"])
        daily_ready = round((baik / daily_fc) * 100)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="cx-card">
            <div style="font-size:0.85rem; color:#64748B;">Total Inspeksi</div>
            <div style="font-size:2rem; font-weight:800; color:#003366;">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        score_class = "score-high" if avg_score >= 4.2 else ("score-medium" if avg_score >= 3.5 else "score-low")
        st.markdown(f"""
        <div class="cx-card">
            <div style="font-size:0.85rem; color:#64748B;">Rata-rata Skor</div>
            <div style="font-size:2rem; font-weight:800;"><span class="{score_class}">{avg_score:.2f}</span> <span style="font-size:1rem; color:#64748B;">/ 5</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="cx-card">
            <div style="font-size:0.85rem; color:#64748B;">Inspeksi Bulan Ini</div>
            <div style="font-size:2rem; font-weight:800; color:#003366;">{this_month}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        ready_color = "#10B981" if daily_ready >= 80 else ("#F59E0B" if daily_ready >= 60 else "#EF4444")
        st.markdown(f"""
        <div class="cx-card">
            <div style="font-size:0.85rem; color:#64748B;">Facility Readiness Hari Ini</div>
            <div style="font-size:2rem; font-weight:800; color:{ready_color};">{daily_ready}%</div>
            <div style="font-size:0.75rem; color:#64748B;">{daily_fc} area • {daily_iss} issue terbuka</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not audits:
        st.markdown("""
        <div class="cx-card">
            <h3>Belum ada data inspeksi lengkap</h3>
            <p>Mulai dengan tab <b>✍️ My Inspection</b> untuk pengecekan pilar lengkap, atau gunakan <b>📋 Daily Service QC (Harian)</b> untuk pencatatan cepat harian + foto.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # === PILLAR PERFORMANCE (using existing pillar styling) ===
    st.markdown("### 📈 Performa Rata-rata per Pilar")

    pillars = ["People", "Process", "Premises"]
    pillar_avgs = {}
    for p in pillars:
        scores = [a.get("pillar_scores", {}).get(p, 0) for a in audits if a.get("pillar_scores")]
        pillar_avgs[p] = round(sum(scores) / len(scores), 2) if scores else 0.0

    pcols = st.columns(3)
    pillar_info = [
        ("People", "👥", "people", pillar_avgs["People"]),
        ("Process", "⚙️", "process", pillar_avgs["Process"]),
        ("Premises", "🏢", "premises", pillar_avgs["Premises"]),
    ]

    for i, (name, emoji, cls, val) in enumerate(pillar_info):
        with pcols[i]:
            badge = f'<span class="pillar-badge pillar-{cls}">{name}</span>'
            color = "#1E40AF" if cls == "people" else ("#3730A3" if cls == "process" else "#065F46")
            st.markdown(f"""
            <div class="cx-card" style="text-align:center;">
                <div style="font-size:1.6rem; margin-bottom:4px;">{emoji}</div>
                <div style="margin-bottom:6px;">{badge}</div>
                <div style="font-size:2.1rem; font-weight:800; color:{color};">{val:.2f}</div>
                <div style="font-size:0.75rem; color:#64748B;">rata-rata dari {len(audits)} inspeksi</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # === RECENT INSPECTIONS (styled cards) ===
    st.markdown("### 📋 Inspeksi Terbaru")

    recent = sorted(audits, key=lambda x: x.get("audit_date", ""), reverse=True)[:6]

    for audit in recent:
        date = audit.get("audit_date", "-")
        airport = audit.get("airport", "-")
        terminal = audit.get("terminal", "")
        loc = f"{airport} - {terminal}" if terminal else airport
        score = audit.get("overall_score", 0)
        auditor = audit.get("auditor", audit.get("inspector", "-"))
        pillar_scores = audit.get("pillar_scores", {})

        # Score styling
        if score >= 4.2:
            score_html = f'<span class="score-high">{score:.2f}</span>'
        elif score >= 3.5:
            score_html = f'<span class="score-medium">{score:.2f}</span>'
        else:
            score_html = f'<span class="score-low">{score:.2f}</span>'

        # Mini pillar badges
        p_badges = ""
        for p, s in pillar_scores.items():
            pcls = p.lower()
            p_badges += f'<span class="pillar-badge pillar-{pcls}" style="margin-right:4px;">{p[:3]} {s:.1f}</span>'

        st.markdown(f"""
        <div class="cx-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-weight:700; font-size:1.05rem; color:#003366;">{loc}</div>
                    <div style="font-size:0.8rem; color:#64748B;">{date} • {auditor}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.35rem; font-weight:800;">{score_html}</div>
                    <div style="margin-top:4px;">{p_badges}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Data dari riwayat inspeksi lengkap. Untuk kegiatan harian cepat + foto bukti, buka tab **📋 Daily Service QC (Harian)**.")


# =============================================================================
# DAILY QC PERSISTENCE HELPERS
# Data hanya hilang jika user explicitly hapus (tombol Hapus atau Mulai Hari Baru)
# =============================================================================

def _get_daily_qc_file_path(date_str: str, inspector: str) -> Path:
    """Return path for persisted daily qc data."""
    safe_name = inspector.replace(" ", "_").replace("/", "_").replace("\\", "_")
    reports = get_reports_dir() / "daily_qc"
    reports.mkdir(parents=True, exist_ok=True)
    return reports / f"{date_str}_{safe_name}.json"


def load_daily_qc(date_str: str, inspector: str) -> dict | None:
    """Load daily qc data from disk if exists."""
    path = _get_daily_qc_file_path(date_str, inspector)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure required keys
                if "facility_checks" not in data:
                    data["facility_checks"] = []
                if "complaints" not in data:
                    data["complaints"] = []
                if "issues" not in data:
                    data["issues"] = []
                return data
        except Exception:
            pass
    return None


def save_daily_qc(data: dict):
    """Persist current daily qc data to disk."""
    if not data or not data.get("date") or not data.get("inspector"):
        return
    path = _get_daily_qc_file_path(data["date"], data["inspector"])
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Non-fatal
        print(f"[Daily QC] Gagal simpan: {e}")


def delete_daily_qc_file(date_str: str, inspector: str):
    """Delete persisted file (used by 'Mulai Hari Baru')."""
    path = _get_daily_qc_file_path(date_str, inspector)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def render_daily_service_qc(context):
    """
    Daily operational module for Airport Service Quality Control Staff (CGK).
    Focus: fast fieldwork + audit-grade records.
    Data is automatically persisted to disk and only cleared when user explicitly deletes it
    (individual "Hapus" buttons or "Mulai Hari Baru").
    """
    st.markdown("## 📋 Daily Service QC (Harian)")
    st.caption("Catat kegiatan harian secara cepat. Data tersimpan otomatis dan hanya akan hilang jika Anda hapus secara manual (tombol Hapus atau 'Mulai Hari Baru').")

    # Prominent stable link for sharing & PWA
    st.success("🔗 Link Permanen (bisa dibagikan & install di HP): https://injourney-cx-daily-qc.streamlit.app")

    st.caption("⏱️ Loading pertama setelah idle di free tier bisa 15-40 detik (cold start). Setelah itu biasanya jauh lebih cepat. App ini sudah di-slim supaya lebih ringan.")

    # Mobile / PWA usage tip - iPhone focused. Now using the real stable URL.
    with st.expander("📱 Install sebagai App di iPhone (Bisa dari Mana Saja)", expanded=True):
        st.markdown("""
        **✅ Link Stabil Permanen (24/7, gratis, siap pakai):**

        **https://injourney-cx-daily-qc.streamlit.app**

        **Langkah Install di iPhone (Safari):**
        1. Buka link di atas pakai **Safari** (bukan Chrome/Firefox).
        2. Ketuk tombol **Share** (kotak dengan panah ke atas).
        3. Scroll ke bawah → pilih **"Add to Home Screen"**.
        4. Ganti nama jadi **"CX Daily QC"** → Add.

        Selesai! Icon akan muncul di home screen seperti app asli. Buka dari situ = full screen, cepat, dan bisa ambil foto bukti dari galeri HP, serta **siapa saja** yang punya link bisa pakai (asalkan ada internet).

        **PENTING — Data di Cloud (Free Tier):**
        Penyimpanan bersifat sementara. Data Daily Service QC (facility, complaints, issues + foto) **hanya aman kalau di-Export**.
        - Tiap selesai hari, scroll ke bawah → tekan **"Export Daily Data (JSON for backup)"**.
        - Besok / setelah app restart → pakai **"Import Daily Data JSON"** untuk restore.
        - Data **hanya hilang** kalau kamu tekan "Mulai Hari Baru" atau tombol Hapus.

        **Performa:** Loading pertama setelah idle (free tier) bisa 15-40 detik (cold start). Setelah itu biasanya cepat. App sudah di-slim dari dependency berat.

        Link ini stabil selamanya. Tidak perlu laptop nyala. Bisa dibagikan ke tim.
        """)

    # Ngrok logic removed — app runs on permanent stable URL above.

    today = str(datetime.now().date())
    live_inspector = context.get("inspector", "Inspector")
    live_location = context.get("location", "CGK - Soekarno-Hatta")

    # === PERSISTENCE: Load from disk if available for today ===
    # Data hanya hilang kalau user explicitly delete
    if "daily_qc" not in st.session_state or st.session_state.daily_qc.get("date") != today:
        loaded = load_daily_qc(today, live_inspector)
        if loaded:
            st.session_state.daily_qc = loaded
        else:
            st.session_state.daily_qc = {
                "date": today,
                "inspector": live_inspector,
                "location": live_location,
                "facility_checks": [],
                "complaints": [],
                "issues": []
            }

    daily = st.session_state.daily_qc

    # Sync live inspector and location from sidebar on EVERY render
    # This makes changes in sidebar immediately visible in Daily Service QC header
    if daily.get("inspector") != live_inspector or daily.get("location") != live_location:
        daily["inspector"] = live_inspector
        daily["location"] = live_location
        save_daily_qc(daily)

    current_inspector = daily.get("inspector", live_inspector)
    current_location = daily.get("location", live_location)

    # Quick header info - always reflects current sidebar selection (important for mobile)
    # Use two columns for better readability (avoids truncation on long location names)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Tanggal", daily.get("date"))
        st.metric("Inspector", current_inspector)
    with c2:
        st.markdown("**Lokasi Saat Ini**")
        st.info(current_location, icon="📍")

    st.divider()

    # Four practical inner tabs
    tab_fac, tab_comp, tab_iss, tab_sum = st.tabs([
        "🔧 Facility Readiness",
        "😠 Complaints Log",
        "⚠️ Issues & Follow-up (RCA)",
        "📊 Ringkasan & Export"
    ])

    # ========== FACILITY READINESS ==========
    with tab_fac:
        st.subheader("Pengecekan Kesiapan Fasilitas")
        st.caption("Isi untuk setiap area yang Anda cek hari ini. Status: Baik / Minor Issue / Major Issue. Foto sangat dianjurkan sebagai bukti.")

        # Simple area list (self-contained)
        facility_areas = [
            "Entrance / Drop Off", "Check-in / Self Check-in", "Security Check", "Boarding Gate",
            "Toilet Umum", "Toilet Difabel", "Prayer Room", "Retail / F&B Area",
            "Baggage Claim", "Information Counter", "Lost & Found", "Public Seating",
            "Lift / Escalator", "Signage & Wayfinding", "Parking Area", "Other"
        ]

        with st.form("add_facility_check", clear_on_submit=True):
            area = st.selectbox("Area / Lokasi", facility_areas, key="fac_area")
            status = st.selectbox("Status", ["Baik", "Minor Issue", "Major Issue"], key="fac_status")
            notes = st.text_area("Catatan / Temuan", placeholder="Contoh: Lantai basah di dekat pintu masuk toilet wanita", height=80, key="fac_notes")

            # Foto bukti via galeri (camera direct removed to keep app lighter on mobile)
            st.markdown("**Foto Bukti** (opsional)")
            photo_file = st.file_uploader("Pilih foto dari galeri HP", type=["png", "jpg", "jpeg"], key="fac_photo_file", label_visibility="collapsed")

            if st.form_submit_button("➕ Tambah / Update Pengecekan Area Ini", type="primary"):
                photo = photo_file
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "area": area,
                    "status": status,
                    "notes": notes.strip(),
                    "photo_name": getattr(photo, 'name', 'foto_bukti.jpg') if photo else None,
                    "photo_data": base64.b64encode(photo.getvalue()).decode() if photo else None
                }
                daily["facility_checks"] = [e for e in daily["facility_checks"] if e["area"] != area]
                daily["facility_checks"].append(entry)
                save_daily_qc(daily)
                st.success(f"✅ {area} dicatat sebagai {status}")
                st.rerun()

        # Current list with photo preview
        if daily["facility_checks"]:
            st.markdown("**Pengecekan Hari Ini**")
            for idx, item in enumerate(daily["facility_checks"]):
                emoji = "🟢" if item["status"] == "Baik" else ("🟡" if "Minor" in item["status"] else "🔴")
                with st.expander(f"{emoji} {item['area']} — {item['status']}", expanded=False):
                    st.write(f"**Catatan:** {item['notes'] or '-'}")
                    if item.get("photo_data"):
                        st.image(base64.b64decode(item["photo_data"]), caption=item.get("photo_name"), width=280)
                    if st.button("Hapus", key=f"del_fac_{idx}"):
                        daily["facility_checks"].pop(idx)
                        save_daily_qc(daily)
                        st.rerun()
        else:
            st.info("Belum ada pengecekan fasilitas hari ini.")

    # ========== COMPLAINTS ==========
    with tab_comp:
        st.subheader("Log Keluhan Pelanggan")
        st.caption("Catat keluhan yang diterima hari ini (dari penumpang atau observasi langsung).")

        complaint_cats = ["Cleanliness", "Facility", "Staff Attitude", "Information", "Queue / Waiting Time", "Signage", "Other"]

        with st.form("add_complaint", clear_on_submit=True):
            c_area = st.selectbox("Area Terkait", facility_areas, key="comp_area")
            c_cat = st.selectbox("Kategori", complaint_cats, key="comp_cat")
            c_desc = st.text_area("Deskripsi Keluhan", height=70, key="comp_desc")
            c_action = st.text_input("Tindakan yang Sudah Dilakukan Hari Ini", key="comp_action")
            c_status = st.selectbox("Status Penanganan", ["Open", "In Progress", "Resolved"], key="comp_status")

            if st.form_submit_button("➕ Catat Keluhan", type="primary"):
                comp = {
                    "timestamp": datetime.now().isoformat(),
                    "area": c_area,
                    "category": c_cat,
                    "description": c_desc.strip(),
                    "action_today": c_action.strip(),
                    "status": c_status
                }
                daily["complaints"].append(comp)
                save_daily_qc(daily)
                st.success("Keluhan dicatat.")
                st.rerun()

        if daily["complaints"]:
            for i, c in enumerate(daily["complaints"]):
                emoji = "🟢" if c["status"] == "Resolved" else ("🟡" if c["status"] == "In Progress" else "🔴")
                with st.expander(f"{emoji} [{c['area']}] {c['category']} — {c['status']}", expanded=False):
                    st.write(f"**Deskripsi:** {c['description']}")
                    st.write(f"**Tindakan hari ini:** {c['action_today'] or '-'}")
                    if st.button("Hapus", key=f"del_comp_{i}"):
                        daily["complaints"].pop(i)
                        save_daily_qc(daily)
                        st.rerun()
        else:
            st.info("Belum ada keluhan tercatat hari ini.")

    # ========== ISSUES + RCA ==========
    with tab_iss:
        st.subheader("⚠️ Issues, Root Cause & Tindak Lanjut")
        st.caption("Catat masalah yang ditemukan beserta root cause dan rencana tindak lanjutnya. Data ini penting untuk audit dan perbaikan operasional.")

        issue_cats = ["Cleanliness", "Facility Damage", "Staff / Service", "Process / Flow", "Information / Signage", "Safety", "Other"]
        rca_options = ["Belum diketahui", "Kurangnya cleaning", "Kerusakan fasilitas", "Proses tidak efisien", 
                       "Kurang koordinasi", "Kurang training staff", "Sistem / equipment error", "Lainnya"]

        # Add new issue form
        with st.form("add_issue", clear_on_submit=True):
            i_area = st.selectbox("Area", facility_areas, key="issue_area")
            i_desc = st.text_area("Deskripsi Masalah / Temuan", height=70, key="issue_desc")
            i_cat = st.selectbox("Kategori Masalah", issue_cats, key="issue_cat")
            i_rca = st.selectbox("Root Cause (sementara)", rca_options, key="issue_rca")
            i_immediate = st.text_input("Tindakan Segera yang Sudah Dilakukan", key="issue_immediate")
            i_pic = st.text_input("Penanggung Jawab (PIC)", key="issue_pic")
            i_due = st.date_input("Target Penyelesaian", value=datetime.now().date(), key="issue_due")
            i_status = st.selectbox("Status", ["Open", "In Progress", "Closed"], key="issue_status")

            submitted = st.form_submit_button("➕ Tambah Issue + RCA", type="primary")

            if submitted:
                new_issue = {
                    "timestamp": datetime.now().isoformat(),
                    "area": i_area,
                    "description": i_desc.strip(),
                    "category": i_cat,
                    "root_cause": i_rca,
                    "immediate_action": i_immediate.strip(),
                    "pic": i_pic.strip(),
                    "due_date": str(i_due),
                    "status": i_status
                }
                daily["issues"].append(new_issue)
                save_daily_qc(daily)
                st.success("Issue berhasil dicatat.")
                st.rerun()

        st.divider()

        # List of issues (simple, no AI)
        if daily["issues"]:
            st.markdown("### Daftar Issue Hari Ini")

            for idx, iss in enumerate(daily["issues"]):
                status_emoji = "🟢" if iss["status"] == "Closed" else ("🟡" if iss["status"] == "In Progress" else "🔴")
                header = f"{status_emoji} [{iss['area']}] {iss['category']} — {iss['status']}"

                with st.container(border=True):
                    st.markdown(f"**{header}**")
                    st.write(f"**Deskripsi:** {iss['description']}")
                    st.write(f"**Root Cause (saat ini):** {iss['root_cause']}")
                    st.write(f"**Tindakan Segera:** {iss['immediate_action'] or '-'}")
                    st.write(f"**PIC:** {iss['pic'] or '-'}  |  **Due:** {iss['due_date']}")

                    if st.button("Hapus Issue", key=f"del_iss_{idx}", type="secondary"):
                        daily["issues"].pop(idx)
                        save_daily_qc(daily)
                        st.rerun()

                    st.markdown("")  # spacing

        else:
            st.info("Belum ada issue. Tambahkan melalui form di atas.")

    # ========== SUMMARY & EXPORT ==========
    with tab_sum:
        st.subheader("Ringkasan Harian & Export (Siap Audit)")

        fc = daily.get("facility_checks", [])
        comps = daily.get("complaints", [])
        iss = daily.get("issues", [])

        col1, col2, col3 = st.columns(3)
        col1.metric("Area Dicek", len(fc))
        col2.metric("Keluhan Masuk", len(comps))
        col3.metric("Issue Terbuka", len([i for i in iss if i.get("status") != "Closed"]))

        if fc:
            baik = len([x for x in fc if x.get("status") == "Baik"])
            readiness = round((baik / len(fc)) * 100) if fc else 0
            st.metric("Facility Readiness Hari Ini", f"{readiness}%")

        st.markdown("### Ringkasan Singkat")
        summary_text = f"""Daily Service QC - {daily.get('date')}
Inspector: {daily.get('inspector')}
Lokasi: {daily.get('location')}

Facility Checks: {len(fc)} area (Baik: {len([x for x in fc if x.get('status')=='Baik'])})
Complaints: {len(comps)}
Open Issues: {len([i for i in iss if i.get('status') != 'Closed'])}"""
        st.text_area("Ringkasan", value=summary_text, height=140)

        st.divider()

        # Professional exports (the ones we fixed earlier)
        if st.button("📄 Generate & Download PDF Report (Modern & Ready for Audit)", type="primary", use_container_width=True):
            if not (fc or comps or iss):
                st.warning("Belum ada data.")
            else:
                pdf_bytes = generate_daily_qc_pdf(daily)
                fname = f"Daily_Service_QC_Report_{daily.get('date')}_{daily.get('inspector','').replace(' ', '_')}.pdf"
                st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=fname, mime="application/pdf", use_container_width=True)

        if st.button("📊 Download Excel Report (Multi-Sheet, Professional)", type="primary", use_container_width=True):
            if not (fc or comps or iss):
                st.warning("Belum ada data.")
            else:
                excel_bytes = generate_daily_qc_excel(daily)
                fname = f"Daily_Service_QC_Report_{daily.get('date')}_{daily.get('inspector','').replace(' ', '_')}.xlsx"
                st.download_button("⬇️ Download Excel", data=excel_bytes, file_name=fname, 
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        # Cloud-friendly backup: Export/Import daily data as JSON (for Streamlit Cloud where local files are ephemeral)
        st.markdown("### Backup / Restore Daily Data (for cloud deployment)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Daily Data (JSON for backup)", use_container_width=True):
                if not (fc or comps or iss):
                    st.warning("No data to export.")
                else:
                    data_str = json.dumps(daily, indent=2, default=str)
                    st.download_button("⬇️ Download daily_data.json", data=data_str, file_name=f"daily_data_{daily.get('date')}.json", mime="application/json", use_container_width=True)
        with col2:
            uploaded = st.file_uploader("Import Daily Data JSON (restore backup)", type="json", key="daily_import")
            if uploaded is not None:
                try:
                    imported = json.load(uploaded)
                    st.session_state.daily_qc = imported
                    st.success("Daily data restored from backup!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to import: {e}")

        if st.button("🔄 Mulai Hari Baru (Clear Data)", type="secondary"):
            # Delete persisted file so data truly disappears only on explicit clear
            delete_daily_qc_file(today, live_inspector)
            st.session_state.daily_qc = {
                "date": str(datetime.now().date()),
                "inspector": live_inspector,
                "location": live_location,
                "facility_checks": [],
                "complaints": [],
                "issues": []
            }
            st.rerun()


def main():
    apply_branding()
    context = render_sidebar()
    render_header()
    render_environment_banner()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "✍️ My Inspection",
        "📚 Knowledge Base",
        "📄 Laporan",
        "📋 Daily Service QC (Harian)"
    ])
    
    with tab1:
        render_dashboard(context)
    
    with tab2:
        render_audit_form(context)
    
    with tab3:
        render_knowledge_base(context)
    
    with tab4:
        render_reports_tab(context)
    
    with tab5:
        render_daily_service_qc(context)
    
    # Footer
    st.markdown("""
    <div class="app-footer">
        © 2025 InJourney Airports • CX Quality Control System — """ + APP_VERSION + """<br>
        Berdasarkan Official CX Transformation Playbooks
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


def generate_daily_qc_excel(daily_data: dict) -> bytes:
    """
    Modern, professional, and attractive Excel report for Daily Service QC.
    Multiple sheets with branding colors, conditional formatting, and clean layout.
    Ready for management and audit use.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage
    import base64
    from io import BytesIO

    wb = Workbook()

    # Brand colors (matching the app and PDF)
    IJ_BLUE = "003366"
    IJ_TEAL = "00A8A8"
    IJ_LIGHT = "E0F7F7"
    GREEN = "10B981"
    YELLOW = "F59E0B"
    RED = "EF4444"
    WHITE = "FFFFFF"
    LIGHT_GRAY = "F8FAFC"
    DARK = "1E293B"

    header_fill = PatternFill(start_color=IJ_TEAL, end_color=IJ_TEAL, fill_type="solid")
    header_font = Font(bold=True, color=WHITE, size=11)
    title_font = Font(bold=True, color=IJ_BLUE, size=16)
    subtitle_font = Font(bold=True, color=IJ_TEAL, size=11)
    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )
    center_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
    left_align = Alignment(horizontal='left', vertical='center', wrap_text=True)

    fc = daily_data.get("facility_checks", [])
    comps = daily_data.get("complaints", [])
    iss = daily_data.get("issues", [])

    # ========== SHEET 1: Ringkasan ==========
    ws = wb.active
    ws.title = "Ringkasan"

    ws.merge_cells('A1:F1')
    ws['A1'] = "LAPORAN HARIAN SERVICE QUALITY CONTROL"
    ws['A1'].font = title_font
    ws['A1'].alignment = Alignment(horizontal='center', vertical='center')

    ws.merge_cells('A2:F2')
    ws['A2'] = "Daily Operational Report — Modern & Professional"
    ws['A2'].font = subtitle_font
    ws['A2'].alignment = Alignment(horizontal='center')

    # Metadata
    ws['A4'] = "Tanggal"
    ws['B4'] = daily_data.get("date", "-")
    ws['A5'] = "Inspector"
    ws['B5'] = daily_data.get("inspector", "-")
    ws['A6'] = "Lokasi"
    ws['B6'] = daily_data.get("location", "-")
    ws['A7'] = "Waktu Generate"
    ws['B7'] = datetime.now().strftime("%d %b %Y %H:%M")

    for row in range(4, 8):
        ws[f'A{row}'].font = Font(bold=True, color=IJ_TEAL)
        ws[f'A{row}'].fill = PatternFill(start_color=IJ_LIGHT, end_color=IJ_LIGHT, fill_type="solid")

    # Key Metrics
    total_areas = len(fc)
    baik = len([x for x in fc if x.get("status") == "Baik"])
    readiness = round((baik / total_areas * 100)) if total_areas > 0 else 0
    open_issues = len([x for x in iss if x.get("status") != "Closed"])
    resolved_complaints = len([x for x in comps if x.get("status") == "Resolved"])

    ws['A9'] = "METRIK UTAMA"
    ws['A9'].font = Font(bold=True, color=IJ_BLUE, size=12)
    ws.merge_cells('A9:C9')

    metrics = [
        ("Facility Readiness", f"{readiness}%", f"{baik}/{total_areas} area Baik"),
        ("Keluhan Masuk", len(comps), f"{resolved_complaints} sudah ditangani"),
        ("Issue Terbuka", open_issues, "Memerlukan tindak lanjut"),
    ]

    for i, (label, value, note) in enumerate(metrics, start=10):
        ws[f'A{i}'] = label
        ws[f'B{i}'] = value
        ws[f'C{i}'] = note
        ws[f'A{i}'].font = Font(bold=True)
        ws[f'B{i}'].alignment = center_align

    if readiness >= 80:
        ws['B10'].fill = PatternFill(start_color=GREEN, end_color=GREEN, fill_type="solid")
        ws['B10'].font = Font(bold=True, color=WHITE)
    elif readiness >= 60:
        ws['B10'].fill = PatternFill(start_color=YELLOW, end_color=YELLOW, fill_type="solid")
        ws['B10'].font = Font(bold=True, color=DARK)
    else:
        ws['B10'].fill = PatternFill(start_color=RED, end_color=RED, fill_type="solid")
        ws['B10'].font = Font(bold=True, color=WHITE)

    ws.column_dimensions['A'].width = 22
    ws.column_dimensions['B'].width = 18
    ws.column_dimensions['C'].width = 28

    # ========== SHEET 2: Facility Readiness ==========
    ws2 = wb.create_sheet("Facility Readiness")

    ws2.merge_cells('A1:E1')
    ws2['A1'] = f"FACILITY READINESS - {daily_data.get('date')}"
    ws2['A1'].font = title_font
    ws2['A1'].alignment = Alignment(horizontal='center')

    ws2.merge_cells('A2:E2')
    ws2['A2'] = "PERHATIAN: Foto bukti embedded. Lihat KOLOM G (thumbnail kecil di sebelah kanan tabel) — scroll ke bawah untuk 'LAMPIRAN FOTO BUKTI' versi besar. Jika gambar tidak tampil, buka file ini di Microsoft Excel → klik 'Enable Editing' / 'Enable Content' jika muncul peringatan."
    ws2['A2'].font = Font(size=8, italic=True, color="666666")
    ws2['A2'].alignment = Alignment(horizontal='left', wrap_text=True)

    headers = ["No", "Area / Lokasi", "Status", "Catatan", "Foto"]
    for col, header in enumerate(headers, 1):
        cell = ws2.cell(row=3, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = thin_border

    for i, item in enumerate(fc, 1):
        row = i + 3
        ws2.cell(row=row, column=1, value=i).alignment = center_align
        ws2.cell(row=row, column=2, value=item.get("area", "-"))
        status_cell = ws2.cell(row=row, column=3, value=item.get("status", "-"))
        status_cell.alignment = center_align

        if item.get("status") == "Baik":
            status_cell.fill = PatternFill(start_color=GREEN, end_color=GREEN, fill_type="solid")
            status_cell.font = Font(bold=True, color=WHITE)
        elif "Minor" in item.get("status", ""):
            status_cell.fill = PatternFill(start_color=YELLOW, end_color=YELLOW, fill_type="solid")
            status_cell.font = Font(bold=True, color=DARK)
        else:
            status_cell.fill = PatternFill(start_color=RED, end_color=RED, fill_type="solid")
            status_cell.font = Font(bold=True, color=WHITE)

        ws2.cell(row=row, column=4, value=item.get("notes", "-"))
        foto_text = "Lihat lampiran di bawah" if item.get("photo_data") else "Tidak ada foto"
        ws2.cell(row=row, column=5, value=foto_text)

        for col in range(1, 6):
            ws2.cell(row=row, column=col).border = thin_border
            ws2.cell(row=row, column=col).alignment = left_align if col in [2,4] else center_align

    # === THUMBNAILS IN COLUMN G (immediate visibility next to each row) ===
    # This ensures photos "tampil" right when the user opens the Excel file.
    for row_idx, item in enumerate(fc, start=4):
        if item.get("photo_data"):
            try:
                img_bytes = base64.b64decode(item["photo_data"])
                img = XLImage(BytesIO(img_bytes))
                img.width = 52
                img.height = 40
                ws2.add_image(img, f"G{row_idx}")
            except Exception:
                pass

    # === LAMPIRAN FOTO BUKTI (larger versions at the bottom for detailed review) ===
    last_data_row = 3 + len(fc) + 2
    has_photos = any(item.get("photo_data") for item in fc)
    if has_photos:
        ws2.cell(row=last_data_row, column=1, value="LAMPIRAN FOTO BUKTI (Versi Besar — Scroll ke Bawah)").font = Font(bold=True, color=IJ_BLUE, size=11)
        ws2.merge_cells(start_row=last_data_row, start_column=1, end_row=last_data_row, end_column=5)
        last_data_row += 2
        pidx = 1
        for item in fc:
            if item.get("photo_data"):
                try:
                    img_bytes = base64.b64decode(item["photo_data"])
                    img = XLImage(BytesIO(img_bytes))
                    img.width = 130
                    img.height = 98
                    ws2.add_image(img, f"A{last_data_row}")
                    ws2.cell(row=last_data_row, column=4, value=f"Foto {pidx}: {item.get('area', '')}").font = Font(size=9, italic=True)
                    ws2.row_dimensions[last_data_row].height = 103
                    last_data_row += 7
                    pidx += 1
                except Exception:
                    pass

    ws2.column_dimensions['A'].width = 5
    ws2.column_dimensions['B'].width = 32
    ws2.column_dimensions['C'].width = 14
    ws2.column_dimensions['D'].width = 50
    ws2.column_dimensions['E'].width = 22
    ws2.column_dimensions['G'].width = 10   # dedicated space for quick thumbnails beside the table

    # ========== SHEET 3: Complaints ==========
    ws3 = wb.create_sheet("Complaints")

    ws3.merge_cells('A1:F1')
    ws3['A1'] = f"KELUHAN PELANGGAN - {daily_data.get('date')}"
    ws3['A1'].font = title_font
    ws3['A1'].alignment = Alignment(horizontal='center')

    headers = ["No", "Area", "Kategori", "Deskripsi", "Tindakan Hari Ini", "Status"]
    for col, header in enumerate(headers, 1):
        cell = ws3.cell(row=3, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = thin_border

    for i, c in enumerate(comps, 1):
        row = i + 3
        ws3.cell(row=row, column=1, value=i).alignment = center_align
        ws3.cell(row=row, column=2, value=c.get("area", "-"))
        ws3.cell(row=row, column=3, value=c.get("category", "-"))
        ws3.cell(row=row, column=4, value=c.get("description", "-"))
        ws3.cell(row=row, column=5, value=c.get("action_today", "-"))

        status_cell = ws3.cell(row=row, column=6, value=c.get("status", "-"))
        status_cell.alignment = center_align
        status_cell.border = thin_border

        if c.get("status") == "Resolved":
            status_cell.fill = PatternFill(start_color=GREEN, end_color=GREEN, fill_type="solid")
            status_cell.font = Font(bold=True, color=WHITE)
        elif "Progress" in c.get("status", ""):
            status_cell.fill = PatternFill(start_color=YELLOW, end_color=YELLOW, fill_type="solid")
            status_cell.font = Font(bold=True, color=DARK)
        else:
            status_cell.fill = PatternFill(start_color=RED, end_color=RED, fill_type="solid")
            status_cell.font = Font(bold=True, color=WHITE)

        for col in range(1, 6):
            ws3.cell(row=row, column=col).border = thin_border
            ws3.cell(row=row, column=col).alignment = left_align

    ws3.column_dimensions['A'].width = 5
    ws3.column_dimensions['B'].width = 25
    ws3.column_dimensions['C'].width = 16
    ws3.column_dimensions['D'].width = 45
    ws3.column_dimensions['E'].width = 30
    ws3.column_dimensions['F'].width = 14

    # ========== SHEET 4: Issues & Follow-up ==========
    ws4 = wb.create_sheet("Issues & Follow-up")

    ws4.merge_cells('A1:H1')
    ws4['A1'] = f"ISSUES, ROOT CAUSE & TINDAK LANJUT - {daily_data.get('date')}"
    ws4['A1'].font = title_font
    ws4['A1'].alignment = Alignment(horizontal='center')

    headers = ["No", "Area", "Deskripsi Masalah", "Kategori", "Root Cause", "Tindakan Segera", "PIC / Due Date", "Status"]
    for col, header in enumerate(headers, 1):
        cell = ws4.cell(row=3, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = thin_border

    for i, item in enumerate(iss, 1):
        row = i + 3
        ws4.cell(row=row, column=1, value=i).alignment = center_align
        ws4.cell(row=row, column=2, value=item.get("area", "-"))
        ws4.cell(row=row, column=3, value=item.get("description", "-"))
        ws4.cell(row=row, column=4, value=item.get("category", "-"))
        ws4.cell(row=row, column=5, value=item.get("root_cause", "-"))
        ws4.cell(row=row, column=6, value=item.get("immediate_action", "-"))
        ws4.cell(row=row, column=7, value=f"{item.get('pic', '-')}\n{item.get('due_date', '-')}")
        ws4.cell(row=row, column=7).alignment = Alignment(wrap_text=True, vertical='center')

        status_cell = ws4.cell(row=row, column=8, value=item.get("status", "-"))
        status_cell.alignment = center_align
        status_cell.border = thin_border

        if item.get("status") == "Closed":
            status_cell.fill = PatternFill(start_color=GREEN, end_color=GREEN, fill_type="solid")
            status_cell.font = Font(bold=True, color=WHITE)
        elif "Progress" in item.get("status", ""):
            status_cell.fill = PatternFill(start_color=YELLOW, end_color=YELLOW, fill_type="solid")
            status_cell.font = Font(bold=True, color=DARK)
        else:
            status_cell.fill = PatternFill(start_color=RED, end_color=RED, fill_type="solid")
            status_cell.font = Font(bold=True, color=WHITE)

        for col in range(1, 8):
            ws4.cell(row=row, column=col).border = thin_border
            ws4.cell(row=row, column=col).alignment = left_align if col not in [1, 8] else center_align

    ws4.column_dimensions['A'].width = 5
    ws4.column_dimensions['B'].width = 22
    ws4.column_dimensions['C'].width = 38
    ws4.column_dimensions['D'].width = 18
    ws4.column_dimensions['E'].width = 20
    ws4.column_dimensions['F'].width = 28
    ws4.column_dimensions['G'].width = 18
    ws4.column_dimensions['H'].width = 14

    # Freeze panes
    ws.freeze_panes = 'A4'
    ws2.freeze_panes = 'A4'
    ws3.freeze_panes = 'A4'
    ws4.freeze_panes = 'A4'

    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()

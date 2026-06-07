#!/bin/bash

# =====================================================
# InJourney Airports CX Quality Control
# Easy Run Script
# =====================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "✈️  InJourney Airports CX Quality Control"
echo "========================================"
echo ""

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment tidak ditemukan!"
    echo "   Silakan jalankan dulu:"
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔄 Mengaktifkan virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit belum terinstall di virtual environment."
    echo "   Silakan jalankan: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Virtual environment aktif"
echo "🚀 Menjalankan aplikasi..."
echo ""

# Get local IP for network access (macOS)
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

echo "📱 Untuk akses dari iPhone / HP / orang lain di jaringan yang SAMA:"
echo "   Buka Safari di iPhone lalu masukkan:"
echo "   http://${IP}:8501"
echo ""
echo "   Setelah terbuka, ketuk Share → 'Add to Home Screen' untuk install sebagai App."
echo ""

echo "🌍 PUBLIC LINK (siapa saja yang punya internet bisa buka):"
echo ""
echo "   Cara TERMUDAH (rekomendasi):"
echo "   1. Pastikan ngrok terinstall (sekali saja):"
echo "      brew install ngrok"
echo "   2. Jalankan:"
echo "      ./run.sh --public"
echo "   3. Script akan otomatis jalankan Streamlit + ngrok tunnel."
echo "   4. Copy link HTTPS yang muncul dan kirim ke orang."
echo ""
echo "   Penerima buka link di browser atau di iPhone Safari → Share → Add to Home Screen."
echo ""
echo "   ⚠️ PENTING:"
echo "   - Link ngrok berubah setiap restart (gratis)."
echo "   - Laptop harus tetap nyala."
echo "   - Belum ada password."
echo ""
echo "   Mau link PERMANEN + 24/7 tanpa laptop nyala? Bilang 'deploy'."
echo ""

# Support easy public sharing with ngrok

# Note: The --public mode starts Streamlit + ngrok and tries to print a clean public HTTPS link.
if [ "$1" = "--public" ]; then
    if ! command -v ngrok &> /dev/null; then
        echo "❌ ngrok belum terinstall."
        echo ""
        echo "   Langkah instalasi lengkap (Mac):"
        echo "   1. brew install ngrok"
        echo "   2. Buka https://ngrok.com dan daftar akun gratis (penting!)"
        echo "   3. Setelah login, kamu akan dapat authtoken."
        echo "   4. Jalankan di terminal:"
        echo "      ngrok config add-authtoken <token-yang-kamu-dapat>"
        echo ""
        echo "   5. Baru jalankan lagi: ./run.sh --public"
        echo ""
        echo "   Setelah itu script akan kasih public link https://...ngrok.io"
        exit 1
    fi

    echo "🚀 Menjalankan dalam mode PUBLIC (ngrok)..."
    echo "   Streamlit akan berjalan di background."
    echo ""

    # Start Streamlit in background
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true &
    STREAMLIT_PID=$!

    # Give Streamlit a moment to start
    sleep 4

    echo "🌐 Memulai ngrok tunnel..."
    echo "   (Tekan Ctrl+C untuk berhenti semua)"
    echo ""

    # Start ngrok in background, log to file for parsing
    NGROK_LOG="/tmp/ngrok_public.log"
    rm -f "$NGROK_LOG"
    ngrok http 8501 --log=stdout > "$NGROK_LOG" 2>&1 &
    NGROK_PID=$!

    # Poll ngrok local API for the public URL (more reliable)
    PUBLIC_URL=""
    for i in $(seq 1 12); do
        sleep 1
        if [ -f "$NGROK_LOG" ] && grep -q "started tunnel" "$NGROK_LOG"; then
            # Try to extract from log or use API
            if curl -s http://127.0.0.1:4040/api/tunnels >/dev/null 2>&1; then
                PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c '
import sys, json
try:
    data = json.load(sys.stdin)
    for t in data.get("tunnels", []):
        pu = t.get("public_url", "")
        if pu.startswith("https://"):
            print(pu)
            break
except:
    pass
' 2>/dev/null || true)
            fi
            if [ -n "$PUBLIC_URL" ]; then
                break
            fi
        fi
    done

    if [ -n "$PUBLIC_URL" ]; then
        echo ""
        echo "✅ PUBLIC LINK SIAP (bagikan link ini):"
        echo "   $PUBLIC_URL"
        echo ""
        echo "   Penerima bisa buka di browser biasa."
        echo "   Di iPhone (Safari): buka link → Share → 'Add to Home Screen'."
        echo "   Akan terinstall sebagai app (PWA) dengan ikon."
        echo ""
        echo "   ⚠️  Link ini akan berubah kalau ngrok di-restart."
        echo "   Laptop harus tetap menyala."
        echo ""

        # Save current public URL so the app can display it nicely
        echo "$PUBLIC_URL" > "$PROJECT_DIR/.current_public_url"
    else
        echo "⚠️  Gagal mendapatkan public URL otomatis."
        echo "   Cek output ngrok atau jalankan manual: ngrok http 8501"
        echo ""
    fi

    # Wait for ngrok (keeps script running)
    wait $NGROK_PID 2>/dev/null || true

    # Cleanup on exit
    kill $STREAMLIT_PID 2>/dev/null || true
    exit 0
fi

# Normal local / same-network run
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

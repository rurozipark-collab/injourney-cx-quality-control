# InJourney Airports CX Quality Control

Profesional web-based audit tool untuk mengevaluasi kualitas layanan bandara berdasarkan **6 dokumen resmi InJourney Airports Customer Experience Transformation Concept**.

## Fitur Utama (v1.3)

### Audit & Penilaian
- **Audit Interaktif** — Checklist berdasarkan 3 pilar (People, Premises, Process)
- Skor 1–5 + komentar per item
- Upload foto bukti
- Perhitungan skor otomatis per pilar & overall

### Laporan & Export
- **Export PDF Profesional** — Dengan branding InJourney, tabel skor, rekomendasi, dan area tanda tangan
- **Export Excel** — Multi-sheet + conditional formatting warna skor
- Riwayat audit lengkap

### Knowledge Base (RAG)
- Pencarian cerdas ke dalam 6 dokumen Playbook resmi
- Saat ini berjalan dalam mode lokal (tanpa API Key)

### Manajemen Data
- Hapus audit satu per satu (aman)
- Hapus cepat dengan checkbox risiko
- **Hapus Super Cepat (1 klik)** dengan backup otomatis
- Hapus semua data dengan konfirmasi
- Sistem backup otomatis sebelum penghapusan

### Lainnya
- Dashboard dengan grafik performa
- Pencarian & filter di daftar laporan
- Panduan penggunaan lengkap di dalam aplikasi
- Dukungan mode lokal (tanpa OpenAI Key)

## Struktur Proyek

```
injourney-cx-quality-control/
├── app.py
├── run.sh                  # Script mudah untuk menjalankan
├── requirements.txt
├── .env.example
├── README.md
├── data/                   # Tempat 6 PDF Playbook
├── reports/
│   ├── audits/             # Data audit (JSON)
│   ├── evidence/           # Foto bukti
│   └── backups/            # Backup otomatis sebelum hapus
├── chroma_db/              # Vector database RAG
└── utils/
    ├── audit_manager.py
    ├── rag_engine.py
    ├── pdf_processor.py
    └── llm.py
```

## Cara Menjalankan

1. Buat virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Opsional) Siapkan OpenAI Key
```bash
cp .env.example .env
```
Edit file `.env` dan isi `OPENAI_API_KEY` jika ingin menggunakan fitur chat AI.

3. Jalankan aplikasi
```bash
./run.sh
```
atau
```bash
source .venv/bin/activate
streamlit run app.py
```

## Status Saat Ini

- Aplikasi berjalan dalam **Mode Lokal** (tanpa OpenAI Key)
- Semua fitur audit, laporan, dan manajemen data sudah aktif
- RAG Knowledge Base sudah bisa digunakan untuk pencarian
- Fitur chat otomatis dengan LLM belum tersedia (opsional)

## Rekomendasi

- Untuk penggunaan sehari-hari, mode lokal sudah sangat cukup.
- Jika suatu saat ingin mengaktifkan chat AI, cukup isi OpenAI API Key di `.env` lalu restart aplikasi.

## Lisensi & Catatan

Aplikasi ini dibuat khusus untuk kebutuhan internal InJourney Airports berdasarkan dokumen resmi Customer Experience Transformation Concept.

© 2025 InJourney Airports

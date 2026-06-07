# Deploy ke Cloud (Streamlit Community Cloud - Gratis & Tanpa Kartu)

Ini cara terbaik untuk mendapatkan URL publik stabil permanen, tanpa perlu laptop nyala terus, dan bisa diinstall sebagai app di iPhone (PWA).

## Keuntungan
- URL stabil seperti https://yourname-injourney-cx-qc.streamlit.app
- Gratis (basic).
- Tidak perlu kartu kredit (berbeda dengan Render yang minta verifikasi kartu seperti di screenshot kamu).
- Siapa saja dengan internet bisa buka via link.
- PWA di iPhone: Buka URL di Safari → Share → Add to Home Screen. Akan jadi app full screen dengan ikon.
- Data Daily Service QC bisa di-backup/restore via JSON (fitur sudah ditambahkan di Ringkasan & Export).

## Keterbatasan (Free Tier)
- App akan "sleep" setelah idle ~15-30 menit (otomatis bangun saat dibuka, delay 10-30 detik).
- Penyimpanan file lokal (reports/, daily JSON, foto evidence, chroma_db) **tidak persistent**. Hilang saat restart/redeploy.
  - Solusi: Gunakan tombol "Export Daily Data (JSON for backup)" di Ringkasan & Export untuk backup data harian kamu.
  - Setelah deploy atau restart, pakai "Import Daily Data JSON" untuk restore.
- Jika butuh persistence penuh otomatis (tanpa manual backup), nanti kita integrasikan Supabase (gratis Postgres + Storage).

## Langkah Deploy

1. **Push code ke GitHub**
   - Pastikan semua file penting di-repo: app.py, requirements.txt, .streamlit/ (config.toml), static/ (manifest.json, service-worker.js), data/ (PDF playbooks jika ingin RAG), dll.
   - Repo bisa public atau private (untuk Streamlit Cloud, public lebih mudah).

2. **Deploy di Streamlit Community Cloud**
   - Buka https://share.streamlit.io
   - Login dengan GitHub.
   - Klik "New app" atau "Deploy a public app from GitHub".
   - Pilih repo kamu.
   - Branch: main (atau yang dipakai).
   - Main file path: `app.py` (atau `injourney-cx-quality-control/app.py` jika di subfolder - sesuaikan struktur repo agar app.py mudah diakses).
   - App name: pilih nama bagus, misal "injourney-cx-quality-control".
   - Advanced settings (jika perlu):
     - Secrets: Tambahkan jika pakai OpenAI untuk RAG, contoh:
       ```
       OPENAI_API_KEY = "sk-..."
       ```
   - Klik Deploy.

3. **Setelah Deploy**
   - Dapat URL publik stabil, contoh: https://your-username-injourney-cx-quality-control.streamlit.app
   - App akan build (beberapa menit pertama).
   - RAG/Knowledge Base akan rebuild dari PDF di repo (cepat).
   - Daily Service QC: Gunakan export/import JSON untuk data harian.

4. **Gunakan di iPhone sebagai App (PWA)**
   - Buka URL stabil di **Safari** (bukan browser lain).
   - Ketuk Share (kotak panah ke atas).
   - Scroll ke bawah → "Add to Home Screen".
   - Beri nama (misal "CX Daily QC") → Add.
   - Icon akan muncul di home screen. Buka dari situ = full screen app (manifest sudah disiapkan).
   - Kamera, foto bukti, export PDF/Excel, backup JSON semua akan jalan.

5. **Share dengan Orang Lain**
   - Kirim URL stabil.
   - Mereka buka di browser atau install sebagai PWA di HP.
   - Data daily per inspector (pakai nama di sidebar) - gunakan export/import untuk share/backup antar orang/sesi.

## Tips Tambahan
- **Secrets**: Di dashboard Streamlit Cloud app kamu > Settings > Secrets. Tambah OPENAI_API_KEY jika RAG dipakai.
- **Update Code**: Setelah perubahan, push ke GitHub, Streamlit Cloud akan auto redeploy (atau manual).
- **Custom Domain**: Bisa tambah nanti di settings (butuh domain sendiri).
- **Untuk Data Persistence Penuh**: Jika ingin daily data otomatis tersimpan tanpa manual export (multi user, across restarts), kita bisa tambah Supabase integration (gratis tier). Bilang saja "tambah Supabase", saya implement (ganti JSON file dengan table + storage untuk foto).
- **RAG**: Akan rebuild dari PDF di repo setiap start (ok untuk sekarang).
- **Local Development**: Tetap pakai `./run.sh` atau `./run.sh --public` (ngrok) untuk testing.

## Troubleshooting
- App error on deploy: Cek logs di Streamlit Cloud. Pastikan requirements.txt lengkap, tidak ada local path hardcode (kita sudah support DATA_DIR).
- PWA tidak install di iPhone: Pastikan URL HTTPS, buka di Safari, manifest accessible (static/manifest.json harus ada di repo).
- Data daily hilang: Gunakan backup JSON.
- RAG tidak jalan: Cek jika butuh OpenAI key di secrets.

Setelah deploy, kasih tahu URL-nya. Saya bisa update mobile tip di app untuk langsung pakai URL cloud, atau tambah fitur lain.

Mau lanjut ke Supabase untuk data daily sekarang, atau deploy dulu?

Push ke GitHub dan deploy via share.streamlit.io sekarang juga bisa. Saya siap bantu jika ada error.
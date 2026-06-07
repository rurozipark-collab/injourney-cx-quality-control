# Dockerfile for InJourney Airports CX Quality Control (Streamlit)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (for pdfplumber, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for persistent data (will be mounted as volume in cloud)
RUN mkdir -p /data/reports/daily_qc /data/reports/audits /data/reports/evidence /data/reports/backups /data/chroma_db

# Expose Streamlit default port
EXPOSE 8501

# Default to using /data for persistent files (override with DATA_DIR env if needed)
ENV DATA_DIR=/data

# Run the app
# Use --server.headless true for cloud
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
# ====================================================
# Dockerfile - NusantaraAI Application
# Base image: Python 3.12 slim
# Port: 5000
# ====================================================

FROM python:3.12-slim

LABEL maintainer="NusantaraAI Team"
LABEL description="Sistem Rekomendasi Wisata Indonesia berbasis ML + LLM"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements terlebih dahulu (Docker cache layer optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code aplikasi
COPY app/ ./app/
COPY app.py .

# Buat direktori untuk cache model
RUN mkdir -p /tmp/models

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Non-root user untuk keamanan
RUN useradd -m -u 1000 appuser
USER appuser

# Jalankan aplikasi Flask
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]

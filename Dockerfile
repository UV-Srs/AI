# =============================================================================
# CPU DOCKERFILE - UNIFIED PPE & 3D FACE RECOGNITION SYSTEM
# Optimized for pure CPU execution with PyTorch CPU distribution
# =============================================================================

FROM python:3.11-slim

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# Install native system dependencies for OpenCV, dlib, PostgreSQL, and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU distribution
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy \
    pandas \
    matplotlib \
    Pillow \
    PyYAML \
    tqdm \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    psycopg2-binary \
    python-dotenv \
    websockets \
    mediapipe \
    dlib

# Copy application code and models
COPY backend/ /app/backend/
COPY Model/ /app/Model/

# Create directories for persistent runtime data
RUN mkdir -p /app/data/media /app/Datasets/Learned /app/runs

# Expose FastAPI backend port
EXPOSE 8001

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Launch backend runner
CMD ["python", "backend/run.py"]

# =============================================================================
# AIVerifySnap ML Service - Docker Image
# =============================================================================
# Multi-stage build for optimized production deployment
# Supports both CPU and GPU inference

# -----------------------------------------------------------------------------
# Stage 1: Base Python image with dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Install dependencies
# -----------------------------------------------------------------------------
FROM base as dependencies

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python dependencies
# For GPU support, modify this to use the CUDA wheel index
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Production image
# -----------------------------------------------------------------------------
FROM dependencies as production

# Copy application code
COPY app/ ./app/

# Create models directory for trained weights
RUN mkdir -p models && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the service port
EXPOSE 8000

# Environment variables for configuration
# These can be overridden at runtime
ENV MODEL_BACKEND=aiverifynet \
    MODEL_PATH=models/aiverifynet.pth \
    ALLOW_UNTRAINED=1 \
    MAX_IMAGE_SIDE=1024 \
    DEVICE=auto \
    CORS_ORIGINS="http://localhost:3000,http://localhost:8080" \
    ELA_QUALITY=90 \
    INPUT_SIZE=224 \
    CONFIDENCE_THRESHOLD=0.5

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Run the application with Uvicorn
# Using multiple workers for production performance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# -----------------------------------------------------------------------------
# Stage 4: Development image (optional)
# -----------------------------------------------------------------------------
FROM production as development

# Switch to root for installing dev dependencies
USER root

# Copy and install dev dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Switch back to non-root user
USER appuser

# Override CMD for development with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


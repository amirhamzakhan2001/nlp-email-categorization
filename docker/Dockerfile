# ======================================================
# Base image (CUDA optional)
# If GPU available → CUDA used
# If not → CPU fallback works automatically
# ======================================================
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# ------------------------------------------------------
# Environment variables
# ------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HuggingFace cache (optional override by .env)
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_CACHE=/hf_cache
ENV TORCH_HOME=/hf_cache

# MLflow (can be overridden)
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# ------------------------------------------------------
# System dependencies
# ------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------
# Working directory
# ------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------
# Copy project code
# ------------------------------------------------------
COPY src/ src/
COPY models/ models/
COPY airflow/ airflow/        # optional, safe
COPY docker/ docker/

# ------------------------------------------------------
# Create runtime directories (safe)
# ------------------------------------------------------
RUN mkdir -p \
    data/raw \
    data/processed \
    artifacts \
    outputs/logs \
    outputs/metrics \
    hf_cache

# ------------------------------------------------------
# Default entrypoint
# (Airflow / scripts override this)
# ------------------------------------------------------
ENTRYPOINT ["python", "-m"]
CMD ["src.pipelines.full_pipeline"]
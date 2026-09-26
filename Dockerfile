# ==============================================================================
# Multi-stage Dockerfile for Hugging Face Spaces (or any Docker container host)
# Stage 1: Build the React frontend production bundle
# Stage 2: Serve FastAPI backend + PyTorch ML model + React static files on port 7860
# ==============================================================================

# --- Stage 1: Frontend Build ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Production Python & PyTorch Environment ---
FROM python:3.10-slim

# Set environment variables for Python and Hugging Face Spaces
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0

# Set up a new non-root user (Hugging Face Spaces enforces UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install minimal system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER user

# Upgrade pip and install lightweight CPU-only PyTorch (avoids 2.5GB CUDA overhead)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install deployment dependencies
COPY --chown=user:user requirements-deploy.txt $HOME/app/
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy backend Python code and configurations
COPY --chown=user:user app.py config.py inference.py preprocessing.py $HOME/app/
COPY --chown=user:user configs/ $HOME/app/configs/
COPY --chown=user:user evaluation_plots/ $HOME/app/evaluation_plots/
COPY --chown=user:user assets/ $HOME/app/assets/

# Copy trained model weights
COPY --chown=user:user saved_models/ $HOME/app/saved_models/

# Copy compiled frontend from Stage 1 into the location expected by app.py
COPY --chown=user:user --from=frontend-builder /app/frontend/dist $HOME/app/frontend/dist

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI backend with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

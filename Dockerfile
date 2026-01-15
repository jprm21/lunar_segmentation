FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ─────────────────────────────
# Variables de entorno
# ─────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Costa_Rica
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# ─────────────────────────────
# Dependencias del sistema
# ─────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nano \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────
# NumPy compatible con PyTorch 2.1
# ─────────────────────────────
RUN pip install --no-cache-dir "numpy<2"

# ─────────────────────────────
# Dependencias Python del proyecto
# ─────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────
# Default
# ─────────────────────────────
CMD ["/bin/bash"]


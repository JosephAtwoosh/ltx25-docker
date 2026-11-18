# syntax=docker/dockerfile:1.7
#
# LTX-2.5 inference image (CUDA 13.2).
#
# Why CUDA 13.2: Lightricks pins ltx-core to torch 2.13.0 + cu132, and the
# `natten` extra (used for the fast diffusion-VAE decode) ships only as
# natten==0.21.7+torch2130cu132. A cu128/cu124 image resolves to a torch the
# natten wheel cannot link against. The host driver must support CUDA 13.2
# (R580+). Check with: docker run --rm --gpus all <image> nvidia-smi
#
# Weights are NOT baked in: they are ~71 GiB and gated on Hugging Face. They are
# mounted at /models (see README / scripts/download-models.sh).

ARG CUDA_VERSION=13.2.0
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------
# Stage 1: builder — resolve the uv workspace and build the virtualenv
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.9.30
# Pin the upstream source so the image is reproducible. Bump deliberately.
ARG LTX_REPO=https://github.com/Lightricks/LTX-2.git
ARG LTX_REF=main

ENV PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/ltx2/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv from the official installer (no pip/torch pulled into the system Python).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
    && install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv --version

WORKDIR /opt/ltx2
RUN git init -q . \
    && git remote add origin "${LTX_REPO}" \
    && git fetch -q --depth 1 origin "${LTX_REF}" \
    && git checkout -q FETCH_HEAD \
    && test -f pyproject.toml \
    && echo "LTX-2 checked out at $(git rev-parse --short HEAD)"

# The upstream repo does NOT commit a uv.lock (verified against the GitHub
# contents API), so `uv sync --frozen` / `--locked` cannot be used on a fresh
# clone -- it fails with "No lockfile found". Plain `uv sync` resolves from
# pyproject.toml, whose pins carry the constraints that matter:
#   torch==2.13.0 from the cu132 index, natten==0.21.7+torch2130cu132,
#   transformers <5.15, nvidia-cudnn-cu13==9.24.0.43.
# The `natten` extra is Linux-only and is the production diffusion-VAE decode
# path; without it decode falls back to Triton/eager (much slower), so it is not
# optional here.
RUN uv sync --extra natten \
    && /opt/ltx2/.venv/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__} (CUDA build {torch.version.cuda})")
PY

# ---------------------------------------------------------------------------
# Stage 2: runtime — CUDA libs + ffmpeg, no compilers
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Reduces fragmentation / OOM on the big 22B transformer (upstream advice).
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Off, or 12B Gemma tokenizers spawn a process per core.
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/.hf \
    UV_PROJECT_ENVIRONMENT=/opt/ltx2/.venv \
    PATH=/opt/ltx2/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the resolved workspace and virtualenv from the builder.
COPY --from=builder /opt/ltx2 /opt/ltx2

# Keep NATTEN's extension importable even though the builder's CUDA toolkit is
# not in this stage: it only needs the runtime libs that ship in the CUDA image.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

COPY --chmod=0755 scripts/entrypoint.sh /usr/local/bin/ltx2-entrypoint
COPY --chmod=0755 scripts/doctor.sh /usr/local/bin/ltx2-doctor
COPY --chmod=0755 scripts/run-distilled.sh /usr/local/bin/ltx2-run
COPY --chmod=0644 scripts/check_libs.py /usr/local/bin/ltx2-check-libs

# Weights live outside the image; outputs are written to /output.
RUN mkdir -p /models /output /opt/ltx2/output
VOLUME ["/models", "/output"]

WORKDIR /opt/ltx2
ENTRYPOINT ["/usr/local/bin/ltx2-entrypoint"]
CMD ["doctor"]

#!/usr/bin/env bash
# Preflight for LTX-2.5 inside the container. Read-only: never downloads weights.
#
# Exit codes: 0 = ready, 2 = at least one blocking problem, 3 = ready with warnings.
set -uo pipefail

RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; RST=$'\033[0m'
if [ ! -t 1 ]; then RED=""; YEL=""; GRN=""; DIM=""; RST=""; fi

FAILS=0
WARNS=0
ok()   { printf '  %sok%s   %s\n'  "$GRN" "$RST" "$*"; }
warn() { printf '  %s!!%s   %s\n'  "$YEL" "$RST" "$*"; WARNS=$((WARNS+1)); }
fail() { printf '  %sXX%s   %s\n'  "$RED" "$RST" "$*"; FAILS=$((FAILS+1)); }
info() { printf '  %s--%s   %s\n'  "$DIM" "$RST" "$*"; }

MODELS_DIR="${LTX_MODELS_DIR:-/models}"

echo "LTX-2.5 container preflight"
echo "==========================="

# ---------------------------------------------------------------- GPU / driver
echo
echo "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found in the container — start the container with --gpus all (Docker Desktop / NVIDIA Container Toolkit)."
else
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
  VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
  if [ -z "${GPU_NAME}" ]; then
    fail "nvidia-smi is present but reported no GPU. Check the --gpus flag and the host driver."
  else
    ok "GPU: ${GPU_NAME}"
    info "driver ${DRIVER:-unknown}"
    # The image is built on CUDA 13.2. Its matching branch is R595, but CUDA
    # minor-version compatibility lets any 13.x app run on a driver >= 580.
    DRV_MAJOR=$(printf '%s' "${DRIVER:-0}" | cut -d. -f1)
    if [ "${DRV_MAJOR:-0}" -ge 580 ] 2>/dev/null; then
      ok "driver ${DRIVER} can run CUDA 13.2 (13.x runs on any driver >= 580; R595 is the matching branch)"
    else
      fail "driver ${DRIVER:-unknown} is older than 580, so it cannot run a CUDA 13.2 image. Update the NVIDIA driver (or pick a cloud image with R595+)."
    fi
    if [ -n "${VRAM_MIB:-}" ]; then
      VRAM_GB=$(( VRAM_MIB / 1024 ))
      info "VRAM: ${VRAM_GB} GiB"
      if [ "${VRAM_GB}" -ge 32 ]; then
        ok "VRAM meets the official 32 GB minimum"
      elif [ "${VRAM_GB}" -ge 22 ]; then
        warn "VRAM ${VRAM_GB} GiB is below the official 32 GB minimum; the run script uses --quantization fp8-cast --offload cpu and modest resolutions to fit."
      else
        fail "VRAM ${VRAM_GB} GiB is too small for the 22B transformer even in FP8 (~21 GiB of weights alone)."
      fi
    fi
  fi
fi

# --------------------------------------------------------------- libraries
echo
echo "Libraries"
if [ -f /usr/local/bin/ltx2-check-libs ]; then
  python /usr/local/bin/ltx2-check-libs || FAILS=$((FAILS+1))
else
  python "${LTX_SCRIPTS_DIR:-/opt/ltx2-docker/scripts}/check_libs.py" || FAILS=$((FAILS+1))
fi

# ------------------------------------------------------------------ toolchain
# Triton compiles its CUDA launcher with a C compiler the first time a Triton
# kernel runs, and needs Python.h to build it as a CPython extension. torch 2.13
# routes bmm_outer_product (Gemma-4 RoPE, i.e. the very first forward pass of the
# text encoder) through Triton, so without these the first generation dies with
# "Failed to find C compiler" or "Python.h: No such file or directory" -- after
# the weights have been downloaded.
echo
echo "Toolchain (Triton needs these at runtime)"
CC_BIN="$(command -v cc || command -v gcc || true)"
if [ -n "${CC_BIN}" ]; then
  ok "C compiler: $(printf '%s' "${CC_BIN}") ($("${CC_BIN}" --version 2>/dev/null | head -n1))"
else
  fail "no C compiler (cc/gcc). Triton compiles its CUDA launcher at runtime, so every generation fails at the first op. Install gcc (build-essential)."
fi
PY_HDR="$(python -c 'import os,sysconfig; print(os.path.join(sysconfig.get_paths()["include"], "Python.h"))' 2>/dev/null)"
if [ -n "${PY_HDR}" ] && [ -f "${PY_HDR}" ]; then
  ok "Python.h: ${PY_HDR}"
else
  fail "Python.h missing (python3-dev). Triton cannot compile its launcher as a CPython extension; generation fails at the first op."
fi

# -------------------------------------------------------------------- memory
echo
echo "System memory"
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "${RAM_GB}" -ge 60 ]; then
  ok "${RAM_GB} GiB RAM"
elif [ "${RAM_GB}" -ge 30 ]; then
  ok "${RAM_GB} GiB RAM (enough for --offload cpu, which holds the 12B text encoder in host memory)"
else
  fail "${RAM_GB} GiB RAM is short: --offload cpu keeps the 12B text encoder in host memory (~26 GB) alongside the process."
fi

# --------------------------------------------------------------------- disks
echo
echo "Storage"
if [ -d "${MODELS_DIR}" ]; then
  FREE_GB=$(df -Pk "${MODELS_DIR}" | awk 'NR==2 {printf "%d", $4/1024/1024}')
  if [ "${FREE_GB}" -ge 90 ]; then
    ok "${FREE_GB} GiB free under ${MODELS_DIR}"
  elif [ "${FREE_GB}" -ge 30 ]; then
    warn "${FREE_GB} GiB free under ${MODELS_DIR}; the distilled set is ~71 GiB (plus ~100 GB of headroom recommended)"
  else
    fail "${FREE_GB} GiB free under ${MODELS_DIR}; need ~71 GiB for weights"
  fi

  # ------------------------------------------------------------- weight files
  echo
  echo "Weights (${MODELS_DIR})"
  check_file() { # path, min-bytes, label
    local f="$1" min="$2" label="$3"
    if [ ! -f "$f" ]; then
      fail "missing ${label}: ${f}"
    else
      local sz
      sz=$(stat -c%s "$f")
      if [ "$sz" -lt "$min" ]; then
        fail "${label} is truncated (${sz} bytes): ${f} — re-run the download"
      else
        ok "${label} ($(numfmt --to=iec --suffix=B "$sz" 2>/dev/null || echo "${sz}B"))"
      fi
    fi
  }
  check_file "${MODELS_DIR}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" 40000000000 "distilled transformer (42 GB)"
  check_file "${MODELS_DIR}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" 25000000000 "Gemma 4 12B text encoder (26 GB)"
  check_file "${MODELS_DIR}/vae/ltx-2.5-video-vae-bf16.safetensors" 1000000000 "video VAE"
  check_file "${MODELS_DIR}/vae/ltx-2.5-audio-vae-bf16.safetensors" 300000000 "audio VAE"
  check_file "${MODELS_DIR}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" 900000000 "spatial upscaler"
  echo
  info "Run 'ltx2-run --help' (or 'docker compose run --rm ltx2 run --help') for the distilled CLI."
else
  fail "${MODELS_DIR} does not exist. Mount the models volume: -v <host>/models:/models"
fi

# ---------------------------------------------------------------- conclusion
echo
echo "==========================="
if [ "${FAILS}" -gt 0 ]; then
  printf '%sNOT READY%s: %d blocking problem(s), %d warning(s).\n' "$RED" "$RST" "$FAILS" "$WARNS"
  exit 2
elif [ "${WARNS}" -gt 0 ]; then
  printf '%sREADY (with warnings)%s: %d warning(s).\n' "$YEL" "$RST" "$WARNS"
  exit 3
else
  printf '%sREADY%s: all checks passed.\n' "$GRN" "$RST"
  exit 0
fi

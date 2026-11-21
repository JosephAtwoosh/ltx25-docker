#!/usr/bin/env bash
# Download the LTX-2.5 weights needed by the distilled pipeline (~71 GiB).
#
# The weights are gated: accept the terms at
#   https://huggingface.co/Lightricks/LTX-2.5
# and pass a Read token with "read gated repos" enabled.
#
#   export HF_TOKEN=hf_xxx
#   ./scripts/download-models.sh                 # distilled set (~71 GiB)
#   INCLUDE_DFR=1 ./scripts/download-models.sh   # + DFR detailing IC-LoRA (+0.3 GiB)
#   INCLUDE_DURATION_HEAD=1 ...                  # + auto-duration head (+4 MB)
#   INCLUDE_A2VID=1 ...                          # + full dev transformer + distilled
#                                                #   LoRA for a2vid (+~51 GiB)
#
# Env: HF_TOKEN, MODELS_DIR, HF_CLI (default: hf, falls back to huggingface-cli),
#      MODEL_REPO, HF_MAX_WORKERS, INCLUDE_DFR, INCLUDE_DURATION_HEAD, INCLUDE_A2VID
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models/ltx-2.5}"
MODEL_REPO="${MODEL_REPO:-Lightricks/LTX-2.5}"

if [ -z "${HF_TOKEN:-}" ]; then
  cat >&2 <<'EOF'
error: HF_TOKEN is not set.

  1. Accept the model terms: https://huggingface.co/Lightricks/LTX-2.5
  2. Create a Read token: https://huggingface.co/settings/tokens
     (fine-grained tokens need the "read gated repos" scope)
  3. export HF_TOKEN=hf_...
EOF
  exit 1
fi

# Resolve the CLI. Newer huggingface_hub ships `hf`; older ships `huggingface-cli`.
HF_BIN=""
for candidate in "${HF_CLI:-}" hf huggingface-cli; do
  [ -n "$candidate" ] || continue
  if command -v "$candidate" >/dev/null 2>&1; then HF_BIN="$candidate"; break; fi
done
[ -n "$HF_BIN" ] || { echo "error: neither 'hf' nor 'huggingface-cli' found. pip install -U huggingface_hub[hf]" >&2; exit 1; }

mkdir -p "${MODELS_DIR}"

# The distilled pipeline needs exactly these five components.
FILES=(
  "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
  "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
  "vae/ltx-2.5-video-vae-bf16.safetensors"
  "vae/ltx-2.5-audio-vae-bf16.safetensors"
  "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)
if [ "${INCLUDE_DURATION_HEAD:-0}" = "1" ]; then
  FILES+=("model_patches/ltx-2.5-duration-head-bf16.safetensors")
fi
# a2vid (audio-to-video) stage 1 runs the FULL transformer, not the distilled one,
# and stage 2 refines with the distilled LoRA. Both are required by
# ltx_pipelines.a2vid_two_stage; without them the wrapper refuses to start.
if [ "${INCLUDE_A2VID:-0}" = "1" ]; then
  FILES+=(
    "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
  )
fi

echo "Downloading LTX-2.5 -> ${MODELS_DIR}"
echo "  cli:    $HF_BIN"
echo "  repo:   ${MODEL_REPO}"
echo "  files:  ${#FILES[@]}"
if [ -n "${HF_MAX_WORKERS:-}" ]; then
  echo "  workers: ${HF_MAX_WORKERS}"
fi

ARGS=(download "${MODEL_REPO}" --local-dir "${MODELS_DIR}")
for f in "${FILES[@]}"; do ARGS+=(--include "$f"); done
if [ -n "${HF_MAX_WORKERS:-}" ]; then
  ARGS+=(--max-workers "${HF_MAX_WORKERS}")
fi

if [ "$HF_BIN" = "huggingface-cli" ]; then
  # huggingface-cli also needs an explicit token env var; it reads HF_TOKEN.
  "$HF_BIN" "${ARGS[@]:1}"
else
  "$HF_BIN" "${ARGS[@]}"
fi

# DFR's detailing IC-LoRA lives in its own repository.
if [ "${INCLUDE_DFR:-0}" = "1" ]; then
  echo
  echo "Downloading DFR detailing IC-LoRA -> ${MODELS_DIR}/loras"
  "$HF_BIN" download Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler \
    ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors \
    --local-dir "${MODELS_DIR}/loras"
fi

echo
echo "Done. Sizes:"
du -sh "${MODELS_DIR}"/* 2>/dev/null || true
echo
echo "Next: ./scripts/run.sh   (or: docker compose build && docker compose run --rm ltx2 run)"

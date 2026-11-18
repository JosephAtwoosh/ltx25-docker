#!/usr/bin/env bash
# LTX-2.5 DistilledPipeline wrapper, tuned for a 24 GB GPU (RTX 3090/4090).
#
# Defaults are the "it fits" profile:
#   --quantization fp8-cast  downcasts the 42 GB bf16 transformer to ~21 GB on load
#   --offload cpu            keeps the 26 GB Gemma-4 text encoder in host RAM
#   768x1280, 121 frames     stage 1 runs at this size, stage 2 upsamples 2x
#
# Usage:
#   ltx2-run --help
#   ltx2-run                                  # defaults
#   ltx2-run --prompt "a cat on a beach"      # literal prompt
#   ltx2-run --prompt-file /output/prompt.txt
#   ltx2-run --height 1024 --width 1536 --enhance-prompt
# Any unrecognised flag is forwarded to ltx_pipelines.distilled verbatim.
#
# Env: LTX_MODELS_DIR, LTX_OUTPUT_DIR, HEIGHT, WIDTH, NUM_FRAMES, SEED,
#      QUANTIZATION (fp8-cast|fp8-scaled-mm|none), OFFLOAD (cpu|disk|none), PROMPT
set -euo pipefail

MODELS_DIR="${LTX_MODELS_DIR:-/models}"
OUTPUT_DIR="${LTX_OUTPUT_DIR:-/output}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-121}"
SEED="${SEED:-42}"
QUANTIZATION="${QUANTIZATION:-fp8-cast}"
OFFLOAD="${OFFLOAD:-cpu}"
PROMPT_ARG="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
LTX-2.5 distilled text/image-to-video, 24 GB GPU profile.

Usage: ltx2-run [options] [-- <extra ltx_pipelines.distilled flags>]

Options:
  -h, --help              show this help
  --prompt TEXT           prompt to generate from
  --prompt-file PATH      read the prompt from a file
  --out PATH              exact output file path (default: /output/ltx25-<time>-<WxH>-<N>f.mp4)
  --height N              output height, divisible by 64 (default: 768)
  --width N               output width, divisible by 64 (default: 1280)
  --num-frames N          8k+1 frames, 24 fps (default: 121 = ~5 s)
  --seed N                random seed (default: 42)
  --quantization NAME     fp8-cast | fp8-scaled-mm | none (default: fp8-cast)
  --offload MODE          cpu | disk | none (default: cpu)

Anything else is passed straight to the pipeline, e.g.:
  --enhance-prompt, --lora PATH [STRENGTH], --chunk-pixel-frames 97
EOF
}

EXACT_OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)      usage; exit 0 ;;
    --prompt)       PROMPT_ARG="${2:?--prompt needs a value}"; shift 2 ;;
    --prompt-file)  PROMPT_FILE="${2:?--prompt-file needs a value}"; shift 2 ;;
    --out)          EXACT_OUT="${2:?--out needs a value}"; shift 2 ;;
    --height)       HEIGHT="${2:?--height needs a value}"; shift 2 ;;
    --width)        WIDTH="${2:?--width needs a value}"; shift 2 ;;
    --num-frames)   NUM_FRAMES="${2:?--num-frames needs a value}"; shift 2 ;;
    --seed)         SEED="${2:?--seed needs a value}"; shift 2 ;;
    --quantization) QUANTIZATION="${2:?--quantization needs a value}"; shift 2 ;;
    --offload)      OFFLOAD="${2:?--offload needs a value}"; shift 2 ;;
    --)             shift; break ;;
    *)              break ;;
  esac
done
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------- validation
[ -d "${MODELS_DIR}" ] || die "models directory not found: ${MODELS_DIR} (mount it with -v <host>/models:/models)"
mkdir -p "${OUTPUT_DIR}" || die "cannot write to output directory: ${OUTPUT_DIR}"

if [ $((HEIGHT % 64)) -ne 0 ] || [ $((WIDTH % 64)) -ne 0 ]; then
  die "height/width must be divisible by 64 (got ${HEIGHT}x${WIDTH}); try 512x896, 768x1280, 1024x1536"
fi
if [ $(((NUM_FRAMES - 1) % 8)) -ne 0 ]; then
  die "num-frames must be on the 8k+1 grid (97, 121, 193, ...), got ${NUM_FRAMES}"
fi

TRANSFORMER="${MODELS_DIR}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER="${MODELS_DIR}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE="${MODELS_DIR}/vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE="${MODELS_DIR}/vae/ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER="${MODELS_DIR}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

for f in "${TRANSFORMER}" "${TEXT_ENCODER}" "${VIDEO_VAE}" "${AUDIO_VAE}" "${UPSCALER}"; do
  [ -f "$f" ] || die "weight file missing: $f — run scripts/download-models.sh (or the downloader service)"
done

# ------------------------------------------------------------------- prompt
DEFAULT_PROMPT="A medium close-up shot of a woman with silver hair standing on a rooftop at dusk. She turns her head slowly toward the camera as wind moves her hair, city lights blurring behind her. The camera pushes in gently. Ambient city noise and a soft, low synth note are audible. Cinematic lighting, shallow depth of field."
if [ -n "${PROMPT_FILE}" ]; then
  [ -f "${PROMPT_FILE}" ] || die "prompt file not found: ${PROMPT_FILE}"
  PROMPT_ARG="$(cat "${PROMPT_FILE}")"
fi
if [ -z "${PROMPT_ARG}" ]; then
  PROMPT_ARG="${DEFAULT_PROMPT}"
fi

# ------------------------------------------------------------------- output
if [ -n "${EXACT_OUT}" ]; then
  OUT="${EXACT_OUT}"
else
  STAMP="$(date +%Y%m%d-%H%M%S)"
  OUT="${OUTPUT_DIR}/ltx25-${STAMP}-${WIDTH}x${HEIGHT}-${NUM_FRAMES}f.mp4"
fi

echo "LTX-2.5 distilled"
echo "  output       ${OUT}"
echo "  size         ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames @ 24 fps (~$(awk -v n="${NUM_FRAMES}" 'BEGIN{printf "%.1f", n/24}') s)"
echo "  quantization ${QUANTIZATION}   offload ${OFFLOAD}   seed ${SEED}"
echo
echo "  prompt       ${PROMPT_ARG:0:160}$([ ${#PROMPT_ARG} -gt 160 ] && echo '...')"
echo

CMD=(
  python -m ltx_pipelines.distilled
  --transformer-path       "${TRANSFORMER}"
  --text-encoder-path      "${TEXT_ENCODER}"
  --video-vae-path         "${VIDEO_VAE}"
  --audio-vae-path         "${AUDIO_VAE}"
  --spatial-upsampler-path "${UPSCALER}"
  --height "${HEIGHT}"
  --width  "${WIDTH}"
  --num-frames "${NUM_FRAMES}"
  --seed "${SEED}"
  --prompt "${PROMPT_ARG}"
  --output-path "${OUT}"
)

case "${QUANTIZATION}" in
  none) ;;
  *) CMD+=(--quantization "${QUANTIZATION}") ;;
esac
case "${OFFLOAD}" in
  none) ;;
  *) CMD+=(--offload "${OFFLOAD}") ;;
esac

CMD+=("${EXTRA_ARGS[@]}")

# Native exec so SIGTERM/SIGINT reach Python; the pipeline is long-running.
exec "${CMD[@]}"

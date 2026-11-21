#!/usr/bin/env bash
# LTX-2.5 A2VidPipelineTwoStage wrapper -- audio-driven video.
#
# This module is NOT part of the repo's own tooling (the repo only wraps the
# distilled pipeline). It exists so the a2vid path can be exercised against the
# already-built image without rebuilding: mount this directory and run
#   docker run --rm --gpus all ... -v $PWD/scripts:/work/scripts:ro \
#     ghcr.io/josephatwoosh/ltx25:cu132 bash /work/scripts/run-a2vid.sh --audio /audio/clip.wav
#
# Required weights (beyond the distilled set):
#   diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors   (~42 GB, the FULL model)
#   loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors           (~8.9 GB, stage-2 refiner)
#
# Usage:
#   ltx2-a2vid --audio /audio/clip.wav
#   ltx2-a2vid --audio /audio/clip.wav --num-frames 97 --width 1024 --height 1536
#   ltx2-a2vid --audio /audio/song.mp3 --audio-max-duration 8
#   ltx2-a2vid --audio /audio/clip.wav -- <extra ltx_pipelines.a2vid_two_stage flags>
#
# Env: LTX_MODELS_DIR, LTX_OUTPUT_DIR, AUDIO, HEIGHT, WIDTH, NUM_FRAMES, SEED,
#      QUANTIZATION (fp8-cast|fp8-scaled-mm|none), OFFLOAD (cpu|disk|none), PROMPT
set -euo pipefail

MODELS_DIR="${LTX_MODELS_DIR:-/models}"
OUTPUT_DIR="${LTX_OUTPUT_DIR:-/output}"
HEIGHT="${HEIGHT:-768}"          # stage-2 height; stage 1 runs at HEIGHT/2
WIDTH="${WIDTH:-1280}"           # stage-2 width;  stage 1 runs at WIDTH/2
NUM_FRAMES="${NUM_FRAMES:-121}"  # 8k+1 grid
FRAME_RATE="${FRAME_RATE:-24}"   # 24 fps default; Wan S2V ran 16
SEED="${SEED:-42}"
QUANTIZATION="${QUANTIZATION:-fp8-cast}"
OFFLOAD="${OFFLOAD:-cpu}"
AUDIO="${AUDIO:-}"
AUDIO_MAX_DURATION="${AUDIO_MAX_DURATION:-}"
PROMPT_ARG="${PROMPT:-}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
LTX-2.5 audio-to-video (A2VidPipelineTwoStage).

Usage: ltx2-a2vid --audio PATH [options] [-- <extra pipeline flags>]

Options:
  -h, --help              show this help
  --audio PATH            audio file to drive the video (wav/mp3/flac)
  --audio-max-duration S  derive length from the audio instead of --num-frames
                          (mutually exclusive with --num-frames)
  --audio-start-time S    read the audio from this offset (default 0)
  --image PATH [FRAME [STRENGTH]]
                          still to condition on; repeatable. Default frame 0,
                          strength 1.0. At frame 0 the still REPLACES the first
                          frame (a start image); later frames act as keyframe guides.
  --prompt TEXT           prompt to generate from
  --out PATH              exact output path (default /output/a2vid-<time>.mp4)
  --height N              stage-2 height, /64 (default 768; stage 1 = 384)
  --width N               stage-2 width,  /64 (default 1280; stage 1 = 640)
  --num-frames N          8k+1 frames (default 121 = ~5 s at 24 fps)
  --frame-rate FPS        output frame rate (default 24); scales the clip length
  --seed N                random seed (default 42)
  --quantization NAME     fp8-cast | fp8-scaled-mm | none
  --offload MODE          cpu | disk | none
EOF
}

EXACT_OUT=""
AUDIO_START=""
IMAGE_ARGV=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)             usage; exit 0 ;;
    --audio)               AUDIO="${2:?--audio needs a value}"; shift 2 ;;
    --audio-max-duration)  AUDIO_MAX_DURATION="${2:?--audio-max-duration needs a value}"; shift 2 ;;
    --audio-start-time)    AUDIO_START="${2:?--audio-start-time needs a value}"; shift 2 ;;
    --image)
      img="${2:?--image needs a path}"; shift 2
      [ -f "$img" ] || die "image file not found: ${img}"
      fr=0; st=1.0
      # Optional trailing FRAME (all digits) then STRENGTH (a number), so
      # `--image start.jpg` and `--image start.jpg 0 0.9` both work.
      if [ $# -gt 0 ] && [ -n "${1##*[!0-9]*}" ]; then fr="$1"; shift; fi
      if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then st="$1"; shift; fi
      IMAGE_ARGV+=(--image "$img" "$fr" "$st")
      ;;
    --prompt)              PROMPT_ARG="${2:?--prompt needs a value}"; shift 2 ;;
    --out)                 EXACT_OUT="${2:?--out needs a value}"; shift 2 ;;
    --height)              HEIGHT="${2:?--height needs a value}"; shift 2 ;;
    --width)               WIDTH="${2:?--width needs a value}"; shift 2 ;;
    --num-frames)          NUM_FRAMES="${2:?--num-frames needs a value}"; shift 2 ;;
    --frame-rate)          FRAME_RATE="${2:?--frame-rate needs a value}"; shift 2 ;;
    --seed)                SEED="${2:?--seed needs a value}"; shift 2 ;;
    --quantization)        QUANTIZATION="${2:?--quantization needs a value}"; shift 2 ;;
    --offload)             OFFLOAD="${2:?--offload needs a value}"; shift 2 ;;
    --)                    shift; break ;;
    *)                     break ;;
  esac
done
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------- validation
[ -n "${AUDIO}" ] || die "--audio is required (path to the driving audio file)"
[ -f "${AUDIO}" ] || die "audio file not found: ${AUDIO} (mount a directory, e.g. -v ./audio:/audio)"
[ -d "${MODELS_DIR}" ] || die "models directory not found: ${MODELS_DIR}"
mkdir -p "${OUTPUT_DIR}" || die "cannot write to output directory: ${OUTPUT_DIR}"

if [ $((HEIGHT % 64)) -ne 0 ] || [ $((WIDTH % 64)) -ne 0 ]; then
  die "height/width must be divisible by 64 (got ${WIDTH}x${HEIGHT}); try 768x1280 or 1024x1536"
fi
if [ -n "${AUDIO_MAX_DURATION}" ]; then
  case "${AUDIO_MAX_DURATION}" in
    ''|*[!0-9.]*) die "--audio-max-duration must be a number of seconds" ;;
  esac
else
  if [ $(((NUM_FRAMES - 1) % 8)) -ne 0 ]; then
    die "num-frames must be on the 8k+1 grid (97, 121, 193, ...), got ${NUM_FRAMES}"
  fi
fi

TRANSFORMER="${MODELS_DIR}/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
TEXT_ENCODER="${MODELS_DIR}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE="${MODELS_DIR}/vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE="${MODELS_DIR}/vae/ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER="${MODELS_DIR}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
DISTILLED_LORA="${MODELS_DIR}/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"

for f in "${TRANSFORMER}" "${TEXT_ENCODER}" "${VIDEO_VAE}" "${AUDIO_VAE}" "${UPSCALER}" "${DISTILLED_LORA}"; do
  [ -f "$f" ] || die "weight file missing: $f (a2vid needs the FULL dev transformer and the distilled LoRA, not the distilled transformer)"
done

# ------------------------------------------------------------------- prompt
DEFAULT_PROMPT="A performer on a dimly lit stage reacts to the music, moving in time with the beat as colored lights sweep across a haze-filled room. The camera slowly pushes in. Cinematic, high detail."
if [ -z "${PROMPT_ARG}" ]; then
  PROMPT_ARG="${DEFAULT_PROMPT}"
fi

# ------------------------------------------------------------------- output
if [ -n "${EXACT_OUT}" ]; then
  OUT="${EXACT_OUT}"
else
  OUT="${OUTPUT_DIR}/a2vid-$(date +%Y%m%d-%H%M%S)-${WIDTH}x${HEIGHT}.mp4"
fi

echo "LTX-2.5 a2vid (two-stage, audio-driven)"
echo "  audio        ${AUDIO}${AUDIO_MAX_DURATION:+ (max ${AUDIO_MAX_DURATION}s)}"
IMG_COUNT=$(( ${#IMAGE_ARGV[@]} / 4 ))
if [ "${IMG_COUNT}" -gt 0 ]; then
  echo "  image        ${IMG_COUNT} still(s); first: ${IMAGE_ARGV[1]} (frame ${IMAGE_ARGV[2]}, strength ${IMAGE_ARGV[3]})"
fi
echo "  output       ${OUT}"
echo "  size         ${WIDTH}x${HEIGHT} final (stage 1 denoises at $((WIDTH / 2))x$((HEIGHT / 2)))"
echo "  quantization ${QUANTIZATION}   offload ${OFFLOAD}   seed ${SEED}"
echo

CMD=(
  python -m ltx_pipelines.a2vid_two_stage
  --transformer-path       "${TRANSFORMER}"
  --text-encoder-path      "${TEXT_ENCODER}"
  --video-vae-path         "${VIDEO_VAE}"
  --audio-vae-path         "${AUDIO_VAE}"
  --spatial-upsampler-path "${UPSCALER}"
  --distilled-lora         "${DISTILLED_LORA}"
  --audio-path             "${AUDIO}"
  --height "${HEIGHT}"
  --width  "${WIDTH}"
  --frame-rate "${FRAME_RATE}"
  --seed "${SEED}"
  --prompt "${PROMPT_ARG}"
  --output-path "${OUT}"
)

if [ -n "${AUDIO_MAX_DURATION}" ]; then
  CMD+=(--audio-max-duration "${AUDIO_MAX_DURATION}")
else
  CMD+=(--num-frames "${NUM_FRAMES}")
fi
if [ -n "${AUDIO_START}" ]; then
  CMD+=(--audio-start-time "${AUDIO_START}")
fi
if [ ${#IMAGE_ARGV[@]} -gt 0 ]; then
  CMD+=("${IMAGE_ARGV[@]}")
fi

case "${QUANTIZATION}" in
  none) ;;
  *) CMD+=(--quantization "${QUANTIZATION}") ;;
esac
case "${OFFLOAD}" in
  none) ;;
  *) CMD+=(--offload "${OFFLOAD}") ;;
esac

CMD+=("${EXTRA_ARGS[@]}")

# Native exec so SIGTERM/SIGINT reach Python; generations run for minutes.
exec "${CMD[@]}"

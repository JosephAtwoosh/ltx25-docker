#!/usr/bin/env bash
# Convenience wrapper around docker compose for LTX-2.5.
#
#   ./scripts/run.sh              # preflight report (doctor)
#   ./scripts/run.sh build        # build the image
#   ./scripts/run.sh download     # fetch weights (requires HF_TOKEN)
#   ./scripts/run.sh run          # generate with the 24 GB defaults
#   ./scripts/run.sh a2vid --audio /audio/clip.wav   # audio-driven video
#   ./scripts/run.sh run --height 1024 --width 1536 --num-frames 97
#   ./scripts/run.sh shell        # interactive shell in the image
#   ./scripts/run.sh <cmd...>     # anything else is passed to the container verbatim
#
# Note: Git Bash rewrites leading-slash arguments. This script sets
# MSYS_NO_PATHCONV for that reason; keep using it rather than raw docker commands.
set -euo pipefail

cd "$(dirname "$0")/.."
export MSYS_NO_PATHCONV=1
export COMPOSE_CONVERT_WINDOWS_PATHS="${COMPOSE_CONVERT_WINDOWS_PATHS:-1}"

command -v docker >/dev/null 2>&1 || { echo "error: docker not found on PATH" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "error: 'docker compose' (v2) is required" >&2; exit 1; }

cmd="${1:-doctor}"
shift || true

case "$cmd" in
  build)
    exec docker compose build "$@"
    ;;
  download)
    : "${HF_TOKEN:?export HF_TOKEN to a Read token with gated-repo access}"
    exec docker compose --profile download run --rm downloader "$@"
    ;;
  doctor)
    exec docker compose run --rm ltx2 "$@"
    ;;
  shell)
    exec docker compose run --rm ltx2 shell "$@"
    ;;
  run)
    exec docker compose run --rm ltx2 run "$@"
    ;;
  *)
    exec docker compose run --rm ltx2 "$cmd" "$@"
    ;;
esac

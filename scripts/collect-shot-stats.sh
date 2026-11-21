#!/usr/bin/env bash
# Run ONE a2vid shot and write a JSON manifest describing how it went.
#
#   bash collect-shot-stats.sh --stats /output/sc01_001.json \
#     --out /output/sc01_001.mp4 \
#     --audio /audio/sc01_001/audio.wav \
#     --image /audio/sc01_001/start.png \
#     --prompt "..." --width 768 --height 512 --audio-max-duration 12
#
# Everything except --stats is forwarded verbatim to run-a2vid.sh, so this is a
# drop-in: add --stats and a fixed --out and the run is instrumented.
#
# A2VID_BIN overrides which wrapper is invoked (default: run-a2vid.sh beside this
# script). Sampler + manifest need nvidia-smi, ffprobe and python3; all three are
# in the image. The manifest is written even when the render fails.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
A2VID="${A2VID_BIN:-${HERE}/run-a2vid.sh}"
STATS=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --stats) STATS="${2:?--stats needs a path}"; shift 2 ;;
    *)       ARGS+=("$1"); shift ;;
  esac
done
[ -n "${STATS}" ] || STATS="/output/a2vid-stats-$(date +%Y%m%d-%H%M%S).json"

# ... locate the output file so we can hash and probe it afterwards.
OUT_FILE=""
i=0
while [ ${i} -lt ${#ARGS[@]} ]; do
  if [ "${ARGS[${i}]}" = "--out" ]; then OUT_FILE="${ARGS[$((i + 1))]:-}"; fi
  i=$((i + 1))
done

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
VRAM_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"

# Peak-usage sampler: 2 s cadence is cheap and enough to catch the stage peaks.
VRAM_CSV="$(mktemp)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 >"${VRAM_CSV}" 2>/dev/null &
SAMPLER=$!
trap 'kill ${SAMPLER} 2>/dev/null || true' EXIT

START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0="$(date +%s)"
set +e
bash "${A2VID}" "${ARGS[@]}"
RC=$?
set -e
T1="$(date +%s)"
kill ${SAMPLER} 2>/dev/null || true
wait ${SAMPLER} 2>/dev/null || true
trap - EXIT

WALL=$(( T1 - T0 ))
PEAK_MIB="$(awk 'BEGIN{m=0}{gsub(/[^0-9]/,"");if($1+0>m)m=$1+0}END{print m+0}' "${VRAM_CSV}" 2>/dev/null || echo 0)"

# ... output probe
OUT_BYTES=0; OUT_SHA=""; OUT_W=0; OUT_H=0; OUT_FPS=0; OUT_FRAMES=0; OUT_SECS=0
if [ -n "${OUT_FILE}" ] && [ -f "${OUT_FILE}" ]; then
  OUT_BYTES="$(stat -c%s "${OUT_FILE}")"
  OUT_SHA="$(sha256sum "${OUT_FILE}" | cut -d' ' -f1)"
  read -r OUT_W OUT_H OUT_FPS OUT_FRAMES < <(
    ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=width,height,r_frame_rate,nb_read_frames \
      -of csv=p=0 "${OUT_FILE}" 2>/dev/null | tr ',' ' '
  ) || true
  OUT_SECS="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${OUT_FILE}" 2>/dev/null || echo 0)"
fi
# r_frame_rate arrives as "24/1"; normalise, and never divide by zero.
OUT_FPS="$(awk -F/ -v f="${OUT_FPS:-0}" 'BEGIN{ if (f ~ /\//) split(f,a,"/"); else a[1]=f; print (a[2]+0>0 ? a[1]/a[2] : a[1]+0) }' 2>/dev/null || echo 0)"

PY="${PY:-}"
if [ -z "${PY}" ]; then
  # The venv interpreter has torch; the system python3 does not necessarily.
  if [ -x /opt/ltx2/.venv/bin/python ]; then PY=/opt/ltx2/.venv/bin/python; else PY=python3; fi
fi

STATS_PATH="${STATS}" START_ISO="${START_ISO}" WALL="${WALL}" RC="${RC}" \
PEAK_MIB="${PEAK_MIB}" VRAM_TOTAL="${VRAM_TOTAL}" GPU_NAME="${GPU_NAME}" DRIVER="${DRIVER}" \
OUT_FILE="${OUT_FILE}" OUT_BYTES="${OUT_BYTES}" OUT_SHA="${OUT_SHA}" \
OUT_W="${OUT_W}" OUT_H="${OUT_H}" OUT_FPS="${OUT_FPS}" OUT_FRAMES="${OUT_FRAMES}" OUT_SECS="${OUT_SECS}" \
CMD="${A2VID} ${ARGS[*]}" STATS_DIR="$(dirname "${STATS}")" \
"${PY}" - <<'PY'
import json, os, platform

try:
    import torch
    torch_version, cuda_version = torch.__version__, torch.version.cuda
    cuda_available = torch.cuda.is_available()
except Exception:  # torch missing/broken must not cost us the whole manifest
    torch_version, cuda_version, cuda_available = None, None, None

e = os.environ
frames = int(float(e.get("OUT_FRAMES") or 0))
secs = float(e.get("OUT_SECS") or 0)
wall = int(e.get("WALL") or 0)
manifest = {
    "schema": "ltx25-a2vid/1",
    "created_at": e["START_ISO"],
    "command": e["CMD"],
    "exit_code": int(e.get("RC") or 0),
    "timing": {
        "start_iso": e["START_ISO"],
        "wall_seconds": wall,
        "video_seconds": round(secs, 3),
        "realtime_factor": round(wall / secs, 2) if secs else None,
    },
    "peak": {
        "vram_mib": int(float(e.get("PEAK_MIB") or 0)),
        "vram_total_mib": int(float(e.get("VRAM_TOTAL") or 0)),
    },
    "gpu": {
        "name": e.get("GPU_NAME") or None,
        "driver": e.get("DRIVER") or None,
        "torch": torch_version,
        "cuda": cuda_version,
        "cuda_available": cuda_available,
    },
    "output": {
        "file": e.get("OUT_FILE") or None,
        "bytes": int(e.get("OUT_BYTES") or 0),
        "sha256": e.get("OUT_SHA") or None,
        "width": int(float(e.get("OUT_W") or 0)),
        "height": int(float(e.get("OUT_H") or 0)),
        "fps": float(e.get("OUT_FPS") or 0),
        "frames": frames,
        "duration_s": round(secs, 3),
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
    },
}
path = e["STATS_PATH"]
os.makedirs(e.get("STATS_DIR") or ".", exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(f"\nstats -> {path}")
PY

echo
echo "== summary =="
echo "  exit        ${RC}"
echo "  wall        ${WALL}s"
echo "  peak VRAM   ${PEAK_MIB} MiB of ${VRAM_TOTAL:-?}"
echo "  output      ${OUT_FILE:-none} (${OUT_BYTES} bytes, ${OUT_FRAMES} frames @ ${OUT_FPS} fps, ${OUT_SECS}s)"
echo "  stats       ${STATS}"
echo
exit "${RC}"

#!/usr/bin/env bash
# Entrypoint for the LTX-2.5 image.
#
#   <no args>        -> doctor (preflight report)
#   doctor           -> preflight report
#   shell            -> interactive bash
#   run [args...]    -> ltx2-run (distilled text-to-video) with args appended
#   <anything else>  -> executed as-is (e.g. python -m ltx_pipelines.dfr_pipeline ...)
set -euo pipefail

cd /opt/ltx2

case "${1:-doctor}" in
  doctor)
    exec /usr/local/bin/ltx2-doctor
    ;;
  shell|bash)
    shift || true
    exec /bin/bash "$@"
    ;;
  run)
    shift
    exec /usr/local/bin/ltx2-run "$@"
    ;;
  *)
    exec "$@"
    ;;
esac

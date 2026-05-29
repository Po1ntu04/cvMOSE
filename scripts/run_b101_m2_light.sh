#!/usr/bin/env bash
set -euo pipefail
# M2-light standalone ablation.  It reuses the M2 runner but writes to distinct
# roots and softens the memory-write gate.  The output masks are still SAM2's
# per-frame predictions; only memory-write decisions and audit states change.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
export PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m2_light}"
export SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m2_light}"
export ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_m2_light.zip}"
export AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m2_light_latest.json}"
export AUDIT_DIR="${AUDIT_DIR:-$MOSE_WORKSPACE/homework/logs/m2_light_by_video}"
export LOG_DIR="${LOG_DIR:-$MOSE_WORKSPACE/homework/logs/m2_light_parallel}"
LIGHT_ARGS=(
  --m2-reference last_reliable_on_empty
  --m2-no-update-prev-on-unreliable
  --m2-obj-thr 0.25
  --m2-stability-thr 0.50
  --m2-min-area-pixels 1
  --m2-min-ratio 0.08
  --m2-max-ratio 8.0
  --m2-max-motion-px-floor 60
  --m2-motion-area-scale 5.0
)
EXTRA_FROM_CALLER="${M2_EXTRA_ARGS:-}"
LIGHT_EXTRA="${LIGHT_ARGS[*]}"
if [[ -n "$EXTRA_FROM_CALLER" ]]; then
  export M2_EXTRA_ARGS="$LIGHT_EXTRA $EXTRA_FROM_CALLER"
else
  export M2_EXTRA_ARGS="$LIGHT_EXTRA"
fi
exec "$SCRIPT_DIR/run_b101_m2_memory_gate.sh"

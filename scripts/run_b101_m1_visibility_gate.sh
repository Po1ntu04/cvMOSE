#!/usr/bin/env bash
set -euo pipefail
SSH_KEY="${B101_SSH_KEY:-$HOME/.ssh/id_ed25519_b101}"
REMOTE_HOST="${B101_HOST:-yuzhixiang@b101.guhk.cc}"
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
CONDA_ENV="${CONDA_ENV:-mose_sam2}"
RAW_PRED_ROOT="${RAW_PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_b101}"
OUT_PRED_ROOT="${OUT_PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m1_visibility}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m1_visibility}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_m1_visibility.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m1_visibility_gate_latest.json}"
DRY_RUN="${DRY_RUN:-0}"
GATE_EXTRA_ARGS="${GATE_EXTRA_ARGS:-}"

REMOTE_DRY=()
if [[ "$DRY_RUN" == "1" ]]; then
  REMOTE_DRY=(--dry-run)
fi

GATE_ARGS=()
if [[ -n "$GATE_EXTRA_ARGS" ]]; then
  # Simple space-separated extra flags, e.g.
  # GATE_EXTRA_ARGS="--strict-gap 6 --reappear-dist-frac 0.14"
  read -r -a GATE_ARGS <<< "$GATE_EXTRA_ARGS"
fi

ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$REMOTE_HOST" bash -s -- \
  "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$RAW_PRED_ROOT" "$OUT_PRED_ROOT" "$SUBMIT_ROOT" "$ZIP_PATH" "$AUDIT_JSON" "${REMOTE_DRY[@]}" "${GATE_ARGS[@]}" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; RAW_PRED_ROOT="$4"; OUT_PRED_ROOT="$5"; SUBMIT_ROOT="$6"; ZIP_PATH="$7"; AUDIT_JSON="$8"; shift 8
cd "$CODE_ROOT"
conda run -n "$ENV_NAME" python tools/apply_visibility_gate.py \
  --workspace "$WORKSPACE" \
  --raw-pred-root "$RAW_PRED_ROOT" \
  --out-pred-root "$OUT_PRED_ROOT" \
  --audit-json "$AUDIT_JSON" \
  "$@"
if [[ " $* " != *" --dry-run "* ]]; then
  conda run -n "$ENV_NAME" python tools/build_submission.py \
    --workspace "$WORKSPACE" \
    --pred-root "$OUT_PRED_ROOT" \
    --submit-root "$SUBMIT_ROOT" \
    --zip-path "$ZIP_PATH" \
    --overwrite
fi
REMOTE

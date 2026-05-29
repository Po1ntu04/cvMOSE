#!/usr/bin/env bash
set -euo pipefail
# Training-free M1.1 with frozen-SAM2 backward cycle consistency verification.
SSH_KEY="${B101_SSH_KEY:-$HOME/.ssh/id_ed25519_b101}"
REMOTE_HOST="${B101_HOST:-yuzhixiang@b101.guhk.cc}"
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
CONDA_ENV="${CONDA_ENV:-mose_sam2}"
RAW_PRED_ROOT="${RAW_PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_b101}"
OUT_PRED_ROOT="${OUT_PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m11_cycle}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m11_cycle}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_m11_cycle.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m11_cycle_gate_latest.json}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-433}"
EXPECTED_PNGS="${EXPECTED_PNGS:-66526}"
GATE_EXTRA_ARGS="${GATE_EXTRA_ARGS:---cycle-verify}"
REMOTE_CUDA_VISIBLE_DEVICES="${REMOTE_CUDA_VISIBLE_DEVICES:-4}"

GATE_ARGS=()
if [[ -n "$GATE_EXTRA_ARGS" ]]; then
  read -r -a GATE_ARGS <<< "$GATE_EXTRA_ARGS"
fi

ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$REMOTE_HOST" bash -s -- \
  "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$RAW_PRED_ROOT" "$OUT_PRED_ROOT" "$SUBMIT_ROOT" "$ZIP_PATH" "$AUDIT_JSON" "$DRY_RUN" "$EXPECTED_VIDEOS" "$EXPECTED_PNGS" "$REMOTE_CUDA_VISIBLE_DEVICES" "${GATE_ARGS[@]}" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$12"
export TQDM_DISABLE=1
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; RAW_PRED_ROOT="$4"; OUT_PRED_ROOT="$5"; SUBMIT_ROOT="$6"; ZIP_PATH="$7"; AUDIT_JSON="$8"; DRY_RUN_FLAG="$9"; EXPECTED_VIDEOS="${10}"; EXPECTED_PNGS="${11}"; shift 12
cd "$CODE_ROOT"
GATE_DRY=()
if [[ "$DRY_RUN_FLAG" == "1" ]]; then
  GATE_DRY=(--dry-run)
fi
conda run -n "$ENV_NAME" python tools/apply_tracklet_gate.py \
  --workspace "$WORKSPACE" \
  --raw-pred-root "$RAW_PRED_ROOT" \
  --out-pred-root "$OUT_PRED_ROOT" \
  --audit-json "$AUDIT_JSON" \
  "${GATE_DRY[@]}" \
  "$@"
if [[ "$DRY_RUN_FLAG" != "1" ]]; then
  conda run -n "$ENV_NAME" python tools/build_submission.py \
    --workspace "$WORKSPACE" \
    --pred-root "$OUT_PRED_ROOT" \
    --submit-root "$SUBMIT_ROOT" \
    --zip-path "$ZIP_PATH" \
    --overwrite
  python3 - "$ZIP_PATH" "$EXPECTED_VIDEOS" "$EXPECTED_PNGS" <<'PY'
import os
import sys
import zipfile
zip_path, expected_videos, expected_pngs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    dirs = {n.split('/')[0] for n in names if '/' in n}
    pngs = [n for n in names if n.endswith('.png')]
    bad = zf.testzip()
if len(dirs) != expected_videos or len(pngs) != expected_pngs or bad is not None:
    raise SystemExit(f"invalid submission zip: videos={len(dirs)} pngs={len(pngs)} testzip_bad={bad}")
print(f"validated_zip={zip_path} videos={len(dirs)} pngs={len(pngs)} size={os.path.getsize(zip_path)}")
PY
fi
REMOTE

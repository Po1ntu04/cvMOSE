#!/usr/bin/env bash
set -euo pipefail
SSH_KEY="${B101_SSH_KEY:-$HOME/.ssh/id_ed25519_b101}"
REMOTE_HOST="${B101_HOST:-yuzhixiang@b101.guhk.cc}"
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
CONDA_ENV="${CONDA_ENV:-mose_sam2}"
GPUS="${GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_parallel}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_sam2_parallel.zip}"
LOG_DIR="${LOG_DIR:-$MOSE_WORKSPACE/homework/logs/parallel_sam2}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

REMOTE_DRY=()
if [[ "$DRY_RUN" == "1" ]]; then
  REMOTE_DRY=(--dry-run)
fi

ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$REMOTE_HOST" bash -s -- \
  "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$GPUS" "$JOBS_PER_GPU" "$PRED_ROOT" "$ZIP_PATH" "$LOG_DIR" "$PYTHON_BIN" "${REMOTE_DRY[@]}" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; GPUS="$4"; JOBS_PER_GPU="$5"; PRED_ROOT="$6"; ZIP_PATH="$7"; LOG_DIR="$8"; PYTHON_BIN="$9"; shift 9
cd "$CODE_ROOT"
"$PYTHON_BIN" tools/launch_parallel_infer.py \
  --workspace "$WORKSPACE" \
  --script "$CODE_ROOT/tools/infer_mosev2_sam2.py" \
  --conda-env "$ENV_NAME" \
  --gpus "$GPUS" \
  --jobs-per-gpu "$JOBS_PER_GPU" \
  --pred-root "$PRED_ROOT" \
  --log-dir "$LOG_DIR" \
  --skip-existing \
  --make-submission-after \
  --zip-path "$ZIP_PATH" \
  "$@"
REMOTE

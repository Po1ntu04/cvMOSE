#!/usr/bin/env bash
set -euo pipefail
SSH_KEY="${B101_SSH_KEY:-$HOME/.ssh/id_ed25519_b101}"
REMOTE_HOST="${B101_HOST:-yuzhixiang@b101.guhk.cc}"
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
CONDA_ENV="${CONDA_ENV:-mose_sam2}"
GPU="${GPU:-4}"
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_b101}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_sam2.zip}"
ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$REMOTE_HOST" bash -s -- "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$GPU" "$PRED_ROOT" "$ZIP_PATH" <<'REMOTE'
set -euo pipefail
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; GPU="$4"; PRED_ROOT="$5"; ZIP_PATH="$6"
cd "$CODE_ROOT"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python tools/infer_mosev2_sam2.py \
  --workspace "$WORKSPACE" \
  --device cuda \
  --pred-root "$PRED_ROOT" \
  --skip-existing
conda run -n "$ENV_NAME" python tools/build_submission.py \
  --workspace "$WORKSPACE" \
  --pred-root "$PRED_ROOT" \
  --zip-path "$ZIP_PATH" \
  --overwrite
REMOTE

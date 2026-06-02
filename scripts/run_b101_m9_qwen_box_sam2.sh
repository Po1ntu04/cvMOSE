#!/usr/bin/env bash
# Build M9 Qwen-box -> SAM2 box-prompt candidate roots on b101.
set -euo pipefail
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
WORKSPACE="${WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
PROPOSALS_JSON="${PROPOSALS_JSON:-$REMOTE_CODE_ROOT/artifacts/m9_qwen_boxes/proposals_key_smoke.json}"
FALLBACK_ROOT="${FALLBACK_ROOT:-$WORKSPACE/homework/pred_sam2_m11_cycle}"
PRED_TEMPLATE="${PRED_TEMPLATE:-$WORKSPACE/homework/pred_m9_qwen_box_clip_rank{rank}}"
AUDIT_JSON="${AUDIT_JSON:-$WORKSPACE/homework/logs/m9_qwen_boxes/m9_box_sam2_clip_rank_audit.json}"
GPU="${GPU:-0}"
CONDA_ENV="${CONDA_ENV:-mose_sam2}"
cd "$REMOTE_CODE_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python tools/apply_m9_box_proposals_sam2.py \
  --workspace "$WORKSPACE" \
  --proposals-json "$PROPOSALS_JSON" \
  --fallback-root "$FALLBACK_ROOT" \
  --pred-root-template "$PRED_TEMPLATE" \
  --audit-json "$AUDIT_JSON" \
  --device cuda \
  --ranks ${RANKS:-1 2 3} \
  --min-confidence "${MIN_CONFIDENCE:-0.01}" \
  --clip-to-prompt-box \
  --clip-pad-frac "${CLIP_PAD_FRAC:-0.25}" \
  --overwrite

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY="${B101_SSH_KEY:-$HOME/.ssh/id_ed25519_b101}"
REMOTE_HOST="${B101_HOST:-yuzhixiang@b101.guhk.cc}"
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
CONDA_ENV="${CONDA_ENV:-mose_sam2}"
GPU="${GPU:-4}"
SYNC_CODE="${SYNC_CODE:-1}"
LOCAL_SPLIT_JSON="${LOCAL_SPLIT_JSON:-$REPO_ROOT/artifacts/m13_split_all15/all15_split_q36.json}"
if [[ -z "${REMOTE_SPLIT_JSON:-}" ]]; then
  REMOTE_SPLIT_JSON="$REMOTE_CODE_ROOT/artifacts/m13_split_inputs/$(basename "$LOCAL_SPLIT_JSON")"
fi
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_m13_teach_boxes_p0}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m13_teach_boxes_p0}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_m13_teach_boxes_p0.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m13_teach_boxes_p0.json}"
VIDEOS="${VIDEOS:-8jsm23a7 q0sizv6m amfdu83t z6dx46qr}"
TARGETS="${TARGETS:-8jsm23a7:1 q0sizv6m:2 amfdu83t:1 z6dx46qr:1}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
EXTRA_ARGS="${M13_EXTRA_ARGS:---merge-mode window --merge-radius 12 --propagate-direction both --max-actions-per-object 1}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-}"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)
if [[ "$SYNC_CODE" == "1" ]]; then "$SCRIPT_DIR/sync_code_b101.sh"; fi
[[ -s "$LOCAL_SPLIT_JSON" ]] || { echo "missing LOCAL_SPLIT_JSON: $LOCAL_SPLIT_JSON" >&2; exit 2; }
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$(dirname "$REMOTE_SPLIT_JSON")'"
rsync -a -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" "$LOCAL_SPLIT_JSON" "$REMOTE_HOST:$REMOTE_SPLIT_JSON"
MAKE_FLAG=()
if [[ "$MAKE_SUBMISSION" == "1" ]]; then MAKE_FLAG=(--make-submission --overwrite-submission); fi
REMOTE_PREFIX="CODE_ROOT=$(printf '%q' "$REMOTE_CODE_ROOT") WORKSPACE=$(printf '%q' "$MOSE_WORKSPACE") CONDA_ENV=$(printf '%q' "$CONDA_ENV") GPU=$(printf '%q' "$GPU") SPLIT_JSON=$(printf '%q' "$REMOTE_SPLIT_JSON") PRED_ROOT=$(printf '%q' "$PRED_ROOT") SUBMIT_ROOT=$(printf '%q' "$SUBMIT_ROOT") ZIP_PATH=$(printf '%q' "$ZIP_PATH") AUDIT_JSON=$(printf '%q' "$AUDIT_JSON") VIDEOS=$(printf '%q' "$VIDEOS") TARGETS=$(printf '%q' "$TARGETS") EXTRA_ARGS=$(printf '%q' "$EXTRA_ARGS") MAKE_FLAGS=$(printf '%q' "${MAKE_FLAG[*]}")"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$REMOTE_PREFIX bash -s" <<'REMOTE'
set -euo pipefail
cd "$CODE_ROOT"
read -r -a VIDEO_ARGS <<< "$VIDEOS"
read -r -a TARGET_ARGS <<< "$TARGETS"
read -r -a EXTRA <<< "$EXTRA_ARGS"
read -r -a MAKE <<< "$MAKE_FLAGS"
rm -rf "$PRED_ROOT"
mkdir -p "$(dirname "$AUDIT_JSON")"
echo "m13_teach_boxes_remote videos=$VIDEOS targets=$TARGETS pred=$PRED_ROOT extra=$EXTRA_ARGS"
CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python tools/infer_mosev2_sam2_teach_boxes.py \
  --workspace "$WORKSPACE" \
  --split-json "$SPLIT_JSON" \
  --pred-root "$PRED_ROOT" \
  --submit-root "$SUBMIT_ROOT" \
  --zip-path "$ZIP_PATH" \
  --audit-json "$AUDIT_JSON" \
  --device cuda \
  --videos "${VIDEO_ARGS[@]}" \
  --targets "${TARGET_ARGS[@]}" \
  "${EXTRA[@]}" \
  "${MAKE[@]}"
REMOTE
if [[ -n "$DOWNLOAD_ROOT" ]]; then
  mkdir -p "$DOWNLOAD_ROOT"
  rsync -a --delete -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" "$REMOTE_HOST:$PRED_ROOT/" "$DOWNLOAD_ROOT/$(basename "$PRED_ROOT")/"
  rsync -a -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" "$REMOTE_HOST:$AUDIT_JSON" "$DOWNLOAD_ROOT/$(basename "$AUDIT_JSON")"
  if [[ "$MAKE_SUBMISSION" == "1" ]]; then
    rsync -a -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" "$REMOTE_HOST:$ZIP_PATH" "$DOWNLOAD_ROOT/$(basename "$ZIP_PATH")"
  fi
fi

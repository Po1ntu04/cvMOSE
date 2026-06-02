#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY="${B101_SSH_KEY:-$HOME/.ssh/id_ed25519_b101}"
REMOTE_HOST="${B101_HOST:-yuzhixiang@b101.guhk.cc}"
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/data1/yuzhixiang/cv_mosev2/external}"
LOCAL_EXTERNAL_ROOT="${LOCAL_EXTERNAL_ROOT:-$REPO_ROOT/external}"
CONDA_ENV="${CONDA_ENV:-mose_sam2}"
GPU="${GPU:-4}"
VIDEOS="${VIDEOS:-r13u5z4y q0sizv6m msinig6m 1qlssuz2 8jsm23a7 lcgc29va}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
SYNC_CODE="${SYNC_CODE:-1}"
SYNC_EXTERNAL="${SYNC_EXTERNAL:-1}"
NUM_PATHWAY="${NUM_PATHWAY:-2}"
IOU_THRE="${IOU_THRE:-0.3}"
UNCERTAINTY="${UNCERTAINTY:-1}"
RUN_NAME="${RUN_NAME:-np${NUM_PATHWAY}_iou${IOU_THRE}_u${UNCERTAINTY}}"
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_m6_sam2long_${RUN_NAME}}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m6_sam2long_${RUN_NAME}}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_sam2long_${RUN_NAME}.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m6_sam2long/${RUN_NAME}.json}"
CHECKPOINT="${CHECKPOINT:-$MOSE_WORKSPACE/5_19/data/sam2/sam2.1_hiera_base_plus.pt}"
MODEL_CFG="${MODEL_CFG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

if [[ "$SYNC_CODE" == "1" ]]; then "$SCRIPT_DIR/sync_code_b101.sh"; fi
if [[ "$SYNC_EXTERNAL" == "1" ]]; then
  test -d "$LOCAL_EXTERNAL_ROOT/SAM2Long" || { echo "missing local external/SAM2Long; clone it first" >&2; exit 2; }
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$EXTERNAL_ROOT/SAM2Long'"
  rsync -a --delete \
    --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
    -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" \
    "$LOCAL_EXTERNAL_ROOT/SAM2Long/" "$REMOTE_HOST:$EXTERNAL_ROOT/SAM2Long/"
fi

LOCAL_GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LOCAL_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
REMOTE_ARGS=("$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$EXTERNAL_ROOT" "$CONDA_ENV" "$GPU" "${VIDEOS:-__EMPTY__}" "$MAKE_SUBMISSION" "$PRED_ROOT" "$SUBMIT_ROOT" "$ZIP_PATH" "$AUDIT_JSON" "$CHECKPOINT" "$MODEL_CFG" "$NUM_PATHWAY" "$IOU_THRE" "$UNCERTAINTY" "$PYTHON_BIN" "${LOCAL_GIT_SHA:-__EMPTY__}" "${LOCAL_GIT_BRANCH:-__EMPTY__}")
REMOTE_CMD="bash -s --"
for arg in "${REMOTE_ARGS[@]}"; do REMOTE_CMD+=" $(printf '%q' "$arg")"; done

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$REMOTE_CMD" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; EXTERNAL_ROOT="$3"; ENV_NAME="$4"; GPU="$5"; VIDEOS="$6"; MAKE_SUBMISSION="$7"; PRED_ROOT="$8"; SUBMIT_ROOT="$9"; ZIP_PATH="${10}"; AUDIT_JSON="${11}"; CHECKPOINT="${12}"; MODEL_CFG="${13}"; NUM_PATHWAY="${14}"; IOU_THRE="${15}"; UNCERTAINTY="${16}"; PYTHON_BIN="${17}"; GIT_SHA="${18}"; GIT_BRANCH="${19}"
if [[ "$VIDEOS" == "__EMPTY__" ]]; then VIDEOS=""; fi
if [[ "$GIT_SHA" != "__EMPTY__" ]]; then export CVMOSE_GIT_SHA="$GIT_SHA"; fi
if [[ "$GIT_BRANCH" != "__EMPTY__" ]]; then export CVMOSE_GIT_BRANCH="$GIT_BRANCH"; fi
cd "$CODE_ROOT"
mkdir -p "$(dirname "$AUDIT_JSON")"
rm -rf "$PRED_ROOT"
if [[ "$MAKE_SUBMISSION" == "1" ]]; then rm -rf "$SUBMIT_ROOT"; rm -f "$ZIP_PATH"; fi
read -r -a VIDEO_ARGS <<< "$VIDEOS"
RUN_ARGS=(--workspace "$WORKSPACE" --sam2long-root "$EXTERNAL_ROOT/SAM2Long" --model-cfg "$MODEL_CFG" --checkpoint "$CHECKPOINT" --pred-root "$PRED_ROOT" --submit-root "$SUBMIT_ROOT" --zip-path "$ZIP_PATH" --audit-json "$AUDIT_JSON" --num-pathway "$NUM_PATHWAY" --iou-thre "$IOU_THRE" --uncertainty "$UNCERTAINTY" --python "$PYTHON_BIN" --device cuda)
if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
if [[ "$MAKE_SUBMISSION" == "1" ]]; then RUN_ARGS+=(--make-submission --overwrite-submission); fi

echo "sam2long_remote videos=${VIDEOS:-<all>} params=np$NUM_PATHWAY iou=$IOU_THRE u=$UNCERTAINTY pred=$PRED_ROOT audit=$AUDIT_JSON"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python tools/infer_mosev2_sam2long_adapter.py "${RUN_ARGS[@]}"

"$PYTHON_BIN" - <<PY
import json, pathlib
pred=pathlib.Path(r"$PRED_ROOT"); audit=pathlib.Path(r"$AUDIT_JSON")
print(f"pred_check {pred} videos={len([p for p in pred.iterdir() if p.is_dir()]) if pred.is_dir() else 'missing'} pngs={sum(1 for _ in pred.rglob('*.png')) if pred.is_dir() else 0}")
data=json.loads(audit.read_text())
print('audit_summary=' + json.dumps(data.get('summary',{}), sort_keys=True))
PY
if [[ "$MAKE_SUBMISSION" == "1" ]]; then
  conda run -n "$ENV_NAME" python tools/validate_mose_submission.py --workspace "$WORKSPACE" --zip-path "$ZIP_PATH" --submit-root "$SUBMIT_ROOT"
fi
REMOTE

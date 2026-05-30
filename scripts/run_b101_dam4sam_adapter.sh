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
MODE="${MODE:-d4sm_multi}"
VIDEOS="${VIDEOS:-r13u5z4y q0sizv6m msinig6m 8jsm23a7 lcgc29va 1qlssuz2}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
SYNC_CODE="${SYNC_CODE:-1}"
SYNC_EXTERNAL="${SYNC_EXTERNAL:-1}"
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_m6_dam4sam_${MODE}_smoke}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m6_dam4sam_${MODE}_smoke}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_dam4sam_${MODE}_15only.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m6_dam4sam/${MODE}_smoke.json}"
D4SM_MODEL_SIZE="${D4SM_MODEL_SIZE:-large}"
DAM4SAM_TRACKER_NAME="${DAM4SAM_TRACKER_NAME:-sam21pp-B}"
# If a standard SAM2.1-L checkpoint is unavailable on b101, use the already-synced Fudan official L checkpoint as a diagnostic fallback.
D4SM_LARGE_SOURCE="${D4SM_LARGE_SOURCE:-$MOSE_WORKSPACE/homework/external_checkpoints/FudanCVL_MOSEv2_baseline/sam2.1_hiera_l_MOSEv2_mss_lvt16.pt}"
DAM_BPLUS_SOURCE="${DAM_BPLUS_SOURCE:-$MOSE_WORKSPACE/5_19/data/sam2/sam2.1_hiera_base_plus.pt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

if [[ "$SYNC_CODE" == "1" ]]; then "$SCRIPT_DIR/sync_code_b101.sh"; fi
if [[ "$SYNC_EXTERNAL" == "1" ]]; then
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$EXTERNAL_ROOT'"
  for repo in DAM4SAM d4sm; do
    test -d "$LOCAL_EXTERNAL_ROOT/$repo" || { echo "missing local external/$repo; clone it first" >&2; exit 2; }
    rsync -a --delete \
      --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
      -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" \
      "$LOCAL_EXTERNAL_ROOT/$repo/" "$REMOTE_HOST:$EXTERNAL_ROOT/$repo/"
  done
fi

LOCAL_GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LOCAL_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
REMOTE_ARGS=("$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$EXTERNAL_ROOT" "$CONDA_ENV" "$GPU" "$MODE" "${VIDEOS:-__EMPTY__}" "$MAKE_SUBMISSION" "$PRED_ROOT" "$SUBMIT_ROOT" "$ZIP_PATH" "$AUDIT_JSON" "$D4SM_MODEL_SIZE" "$DAM4SAM_TRACKER_NAME" "$D4SM_LARGE_SOURCE" "$DAM_BPLUS_SOURCE" "$PYTHON_BIN" "${LOCAL_GIT_SHA:-__EMPTY__}" "${LOCAL_GIT_BRANCH:-__EMPTY__}")
REMOTE_CMD="bash -s --"
for arg in "${REMOTE_ARGS[@]}"; do REMOTE_CMD+=" $(printf '%q' "$arg")"; done

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$REMOTE_CMD" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; EXTERNAL_ROOT="$3"; ENV_NAME="$4"; GPU="$5"; MODE="$6"; VIDEOS="$7"; MAKE_SUBMISSION="$8"; PRED_ROOT="$9"; SUBMIT_ROOT="${10}"; ZIP_PATH="${11}"; AUDIT_JSON="${12}"; D4SM_MODEL_SIZE="${13}"; DAM4SAM_TRACKER_NAME="${14}"; D4SM_LARGE_SOURCE="${15}"; DAM_BPLUS_SOURCE="${16}"; PYTHON_BIN="${17}"; GIT_SHA="${18}"; GIT_BRANCH="${19}"
if [[ "$VIDEOS" == "__EMPTY__" ]]; then VIDEOS=""; fi
if [[ "$GIT_SHA" != "__EMPTY__" ]]; then export CVMOSE_GIT_SHA="$GIT_SHA"; fi
if [[ "$GIT_BRANCH" != "__EMPTY__" ]]; then export CVMOSE_GIT_BRANCH="$GIT_BRANCH"; fi
cd "$CODE_ROOT"
mkdir -p "$EXTERNAL_ROOT/DAM4SAM/checkpoints" "$EXTERNAL_ROOT/d4sm/checkpoints" "$(dirname "$AUDIT_JSON")"
ln -sf "$DAM_BPLUS_SOURCE" "$EXTERNAL_ROOT/DAM4SAM/checkpoints/sam2.1_hiera_base_plus.pt"
ln -sf "$D4SM_LARGE_SOURCE" "$EXTERNAL_ROOT/d4sm/checkpoints/sam2.1_hiera_large.pt"
rm -rf "$PRED_ROOT"
if [[ "$MAKE_SUBMISSION" == "1" ]]; then rm -rf "$SUBMIT_ROOT"; rm -f "$ZIP_PATH"; fi
read -r -a VIDEO_ARGS <<< "$VIDEOS"
RUN_ARGS=(--workspace "$WORKSPACE" --mode "$MODE" --dam4sam-root "$EXTERNAL_ROOT/DAM4SAM" --d4sm-root "$EXTERNAL_ROOT/d4sm" --checkpoint-dir "$EXTERNAL_ROOT/d4sm/checkpoints" --d4sm-model-size "$D4SM_MODEL_SIZE" --tracker-name "$DAM4SAM_TRACKER_NAME" --pred-root "$PRED_ROOT" --submit-root "$SUBMIT_ROOT" --zip-path "$ZIP_PATH" --audit-json "$AUDIT_JSON" --device cuda)
if [[ "$MODE" == "dam4sam_single_per_obj" ]]; then RUN_ARGS=(--workspace "$WORKSPACE" --mode "$MODE" --dam4sam-root "$EXTERNAL_ROOT/DAM4SAM" --d4sm-root "$EXTERNAL_ROOT/d4sm" --checkpoint-dir "$EXTERNAL_ROOT/DAM4SAM/checkpoints" --d4sm-model-size "$D4SM_MODEL_SIZE" --tracker-name "$DAM4SAM_TRACKER_NAME" --pred-root "$PRED_ROOT" --submit-root "$SUBMIT_ROOT" --zip-path "$ZIP_PATH" --audit-json "$AUDIT_JSON" --device cuda); fi
if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
if [[ "$MAKE_SUBMISSION" == "1" ]]; then RUN_ARGS+=(--make-submission --overwrite-submission); fi

echo "dam4sam_remote mode=$MODE videos=${VIDEOS:-<all>} pred=$PRED_ROOT audit=$AUDIT_JSON"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python tools/infer_mosev2_dam4sam_adapter.py "${RUN_ARGS[@]}"

"$PYTHON_BIN" - <<PY
import json, pathlib
pred=pathlib.Path(r"$PRED_ROOT"); audit=pathlib.Path(r"$AUDIT_JSON")
print(f"pred_check {pred} videos={len([p for p in pred.iterdir() if p.is_dir()]) if pred.is_dir() else 'missing'} pngs={sum(1 for _ in pred.rglob('*.png')) if pred.is_dir() else 0}")
data=json.loads(audit.read_text())
print('audit_summary=' + json.dumps(data.get('summary',{}), sort_keys=True))
PY
REMOTE

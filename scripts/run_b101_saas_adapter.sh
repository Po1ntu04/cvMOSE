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
CONDA_ENV="${CONDA_ENV:-saas}"
GPU="${GPU:-4}"
VIDEOS="${VIDEOS:-1qlssuz2 2smf7uq9 3epdtmyr pe0d85lk r13u5z4y q0sizv6m 8jsm23a7}"
CHECKPOINT="${CHECKPOINT:-$EXTERNAL_ROOT/SAAS/checkpoints/SAAS_b+_ytvos_tma.pt}"
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_m6_saas_bplus_smoke}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m6_saas/saas_bplus_smoke.json}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
SYNC_CODE="${SYNC_CODE:-1}"
SYNC_EXTERNAL="${SYNC_EXTERNAL:-1}"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)
if [[ "$SYNC_CODE" == "1" ]]; then "$SCRIPT_DIR/sync_code_b101.sh"; fi
if [[ "$SYNC_EXTERNAL" == "1" ]]; then
  test -d "$LOCAL_EXTERNAL_ROOT/SAAS" || { echo "missing local external/SAAS; clone it first" >&2; exit 2; }
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$EXTERNAL_ROOT/SAAS'"
  rsync -a --delete --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" "$LOCAL_EXTERNAL_ROOT/SAAS/" "$REMOTE_HOST:$EXTERNAL_ROOT/SAAS/"
fi
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s -- "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$EXTERNAL_ROOT" "$CONDA_ENV" "$GPU" "$VIDEOS" "$CHECKPOINT" "$PRED_ROOT" "$AUDIT_JSON" "$MAKE_SUBMISSION" <<'REMOTE'
set -euo pipefail
CODE_ROOT="$1"; WORKSPACE="$2"; EXTERNAL_ROOT="$3"; ENV_NAME="$4"; GPU="$5"; VIDEOS="$6"; CHECKPOINT="$7"; PRED_ROOT="$8"; AUDIT_JSON="$9"; MAKE_SUBMISSION="${10}"
cd "$CODE_ROOT"
read -r -a VIDEO_ARGS <<< "$VIDEOS"
RUN_ARGS=(--workspace "$WORKSPACE" --saas-root "$EXTERNAL_ROOT/SAAS" --checkpoint "$CHECKPOINT" --pred-root "$PRED_ROOT" --audit-json "$AUDIT_JSON")
if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
if [[ "$MAKE_SUBMISSION" == "1" ]]; then RUN_ARGS+=(--make-submission --overwrite-submission); fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python tools/infer_mosev2_saas_adapter.py "${RUN_ARGS[@]}"
REMOTE

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
SYNC_CODE="${SYNC_CODE:-1}"
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m5r_reanchor}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m5r_reanchor}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_m5r_reanchor.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m5r_reanchor_latest.json}"
AUDIT_DIR="${AUDIT_DIR:-$MOSE_WORKSPACE/homework/logs/m5r_reanchor_by_video}"
VIDEOS="${VIDEOS:-}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-auto}"
EXTRA_ARGS="${M5R_EXTRA_ARGS:-}"
SYNC_DINO="${SYNC_DINO:-0}"
LOCAL_DINO_WEIGHTS="${LOCAL_DINO_WEIGHTS:-/home/yu/projects/cv/from fdu/MOSEv2/homework/external_checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth}"
REMOTE_DINO_WEIGHTS="${REMOTE_DINO_WEIGHTS:-$MOSE_WORKSPACE/homework/external_checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LOCAL_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

if [[ "$SYNC_CODE" == "1" ]]; then "$SCRIPT_DIR/sync_code_b101.sh"; fi

if [[ "$SYNC_DINO" == "1" ]]; then
  test -d "$LOCAL_EXTERNAL_ROOT/dinov2" || { echo "missing local external/dinov2; clone it first" >&2; exit 2; }
  test -s "$LOCAL_DINO_WEIGHTS" || { echo "missing local DINO weights: $LOCAL_DINO_WEIGHTS" >&2; exit 2; }
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$EXTERNAL_ROOT/dinov2' '$(dirname "$REMOTE_DINO_WEIGHTS")'"
  rsync -a --delete \
    --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
    -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" \
    "$LOCAL_EXTERNAL_ROOT/dinov2/" "$REMOTE_HOST:$EXTERNAL_ROOT/dinov2/"
  rsync -a --partial \
    -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10" \
    "$LOCAL_DINO_WEIGHTS" "$REMOTE_HOST:$REMOTE_DINO_WEIGHTS"
fi

REMOTE_ARGS=(
  "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$GPU" "$PRED_ROOT" "$SUBMIT_ROOT" "$ZIP_PATH"
  "$AUDIT_JSON" "$AUDIT_DIR" "${VIDEOS:-__EMPTY__}" "$MAKE_SUBMISSION" "${EXTRA_ARGS:-__EMPTY__}"
  "$PYTHON_BIN" "${LOCAL_GIT_SHA:-__EMPTY__}" "${LOCAL_GIT_BRANCH:-__EMPTY__}"
)
REMOTE_CMD="bash -s --"
for arg in "${REMOTE_ARGS[@]}"; do
  REMOTE_CMD+=" $(printf '%q' "$arg")"
done

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$REMOTE_CMD" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; GPU="$4"; PRED_ROOT="$5"; SUBMIT_ROOT="$6"; ZIP_PATH="$7"
AUDIT_JSON="$8"; AUDIT_DIR="$9"; VIDEOS="${10}"; MAKE_SUBMISSION="${11}"; EXTRA_ARGS="${12}"; PYTHON_BIN="${13}"; GIT_SHA="${14}"; GIT_BRANCH="${15}"
if [[ "$VIDEOS" == "__EMPTY__" ]]; then VIDEOS=""; fi
if [[ "$EXTRA_ARGS" == "__EMPTY__" ]]; then EXTRA_ARGS=""; fi
if [[ "$GIT_SHA" != "__EMPTY__" ]]; then export CVMOSE_GIT_SHA="$GIT_SHA"; fi
if [[ "$GIT_BRANCH" != "__EMPTY__" ]]; then export CVMOSE_GIT_BRANCH="$GIT_BRANCH"; fi
cd "$CODE_ROOT"

require_under() {
  local path="$1" prefix="$2" label="$3" canon_path canon_prefix
  canon_path="$(realpath -m -- "$path")"; canon_prefix="$(realpath -m -- "$prefix")"
  if [[ -z "$canon_path" || "$canon_path" == "/" || "$canon_path" == "$canon_prefix" || "$canon_path" == "$WORKSPACE" || "$canon_path" == "$WORKSPACE/homework" || "$canon_path" == "$WORKSPACE/homework/logs" ]]; then
    echo "Refusing unsafe $label path: $path (canonical: $canon_path)" >&2; exit 2
  fi
  case "$canon_path/" in "$canon_prefix"/*) ;; *) echo "Refusing unsafe $label path: $path (required under: $canon_prefix)" >&2; exit 2 ;; esac
}
require_prefix() {
  local path="$1" prefix="$2" label="$3" canon_path canon_prefix
  canon_path="$(realpath -m -- "$path")"; canon_prefix="$(realpath -m -- "$prefix")"
  [[ "$canon_path" == "$canon_prefix"* && "$canon_path" != "$canon_prefix" ]] || { echo "Refusing unsafe $label path: $path (required prefix: $canon_prefix)" >&2; exit 2; }
}
require_prefix "$PRED_ROOT" "$WORKSPACE/homework/pred_" PRED_ROOT
require_prefix "$SUBMIT_ROOT" "$WORKSPACE/homework/submission_" SUBMIT_ROOT
require_under "$ZIP_PATH" "$WORKSPACE/homework" ZIP_PATH
require_under "$AUDIT_JSON" "$WORKSPACE/homework/logs" AUDIT_JSON
require_under "$AUDIT_DIR" "$WORKSPACE/homework/logs" AUDIT_DIR

read -r -a VIDEO_ARGS <<< "$VIDEOS"
read -r -a EXTRA <<< "$EXTRA_ARGS"
ALLOWED_EXTRA_FLAGS=(
  --descriptor --merge-policy --merge-radius --identity-margin --positive-thr --negative-thr
  --max-area-ratio --min-area-ratio --max-anchors-per-object --min-anchor-separation
  --anchor-after-hard-event --no-anchor-after-hard-event --anchor-delay-after-hard-event --hard-event-area-drop-ratio
  --candidate-pad --candidate-frame-stride --component-min-area --component-max-count
  --positive-max-frames --negative-max-items --max-candidates-per-object --no-any-fg-components
  --sam2-auto-mask-candidates --auto-mask-max-frames-per-object --auto-mask-points-per-side
  --auto-mask-pred-iou-thr --auto-mask-stability-thr --auto-mask-min-area --auto-mask-max-count
  --dino-root --dino-weights --dino-variant --dino-max-side --dino-tiny-min-tokens --dino-tiny-crop-min-side --dino-tiny-crop-mult
  --sameclass-margin --anchor-confirm-mode --anchor-confirm-window --anchor-confirm-min-count --strong-margin
  --rollback-max-change-frac --rollback-high-conf-margin
  --baseline-root --fallback-root --m11-root --m2-light-root --tiny-crop-root --sam31-root --rar-rcms-root --rar-state-root --rar-audit-json
  --target-profiles-json --mllm-candidate-judgments-json --mllm-tracklet-judgments-json
  --mllm-policy --mllm-min-veto-confidence --mllm-min-support-confidence --mllm-require-tracklet-for-anchor --mllm-uncertain-action
)
is_allowed_extra_flag() { local flag="$1" allowed; for allowed in "${ALLOWED_EXTRA_FLAGS[@]}"; do [[ "$flag" == "$allowed" || "$flag" == "$allowed="* ]] && return 0; done; return 1; }
for token in "${EXTRA[@]}"; do
  if [[ "$token" == --* ]] && ! is_allowed_extra_flag "$token"; then echo "Refusing unsafe M5R_EXTRA_ARGS flag: $token" >&2; exit 2; fi
done
if [[ "$MAKE_SUBMISSION" == "auto" ]]; then if (( ${#VIDEO_ARGS[@]} > 0 )); then MAKE_SUBMISSION=0; else MAKE_SUBMISSION=1; fi; fi
mkdir -p "$(dirname "$AUDIT_JSON")" "$AUDIT_DIR"
rm -rf "$PRED_ROOT" "$AUDIT_DIR"; rm -f "$AUDIT_JSON"
if [[ "$MAKE_SUBMISSION" == "1" ]]; then rm -rf "$SUBMIT_ROOT"; rm -f "$ZIP_PATH"; fi
mkdir -p "$AUDIT_DIR"
RUN_ARGS=(--workspace "$WORKSPACE" --device cuda --pred-root "$PRED_ROOT" --submit-root "$SUBMIT_ROOT" --zip-path "$ZIP_PATH" --audit-json "$AUDIT_JSON" --audit-dir "$AUDIT_DIR")
if [[ "$MAKE_SUBMISSION" == "1" ]]; then RUN_ARGS+=(--make-submission --overwrite-submission); fi
if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
if (( ${#EXTRA[@]} > 0 )); then RUN_ARGS+=("${EXTRA[@]}"); fi

echo "m5r_remote_run videos=${VIDEOS:-<all>} make_submission=$MAKE_SUBMISSION pred_root=$PRED_ROOT extra=${EXTRA_ARGS:-<none>}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" XFORMERS_DISABLED=1 conda run -n "$ENV_NAME" python tools/infer_mosev2_sam2_reanchor.py "${RUN_ARGS[@]}"

"$PYTHON_BIN" - <<PY
import json, pathlib, zipfile
pred=pathlib.Path(r"$PRED_ROOT"); audit=pathlib.Path(r"$AUDIT_JSON"); submit=pathlib.Path(r"$SUBMIT_ROOT"); zip_path=pathlib.Path(r"$ZIP_PATH"); make_submission="$MAKE_SUBMISSION"=="1"
if not pred.is_dir(): raise SystemExit(f"missing pred root {pred}")
print(f"pred_check dir={pred} videos={len([p for p in pred.iterdir() if p.is_dir()])} pngs={sum(1 for _ in pred.rglob('*.png'))}")
if not audit.is_file(): raise SystemExit(f"missing audit json {audit}")
data=json.loads(audit.read_text(encoding='utf-8'))
print('audit_summary=' + json.dumps(data.get('summary',{}), ensure_ascii=False, sort_keys=True))
if make_submission:
    video_dirs=sorted(p for p in submit.iterdir() if p.is_dir()); png_count=sum(1 for _ in submit.rglob('*.png'))
    print(f"submission_check dir={submit} videos={len(video_dirs)} pngs={png_count}")
    with zipfile.ZipFile(zip_path) as zf: bad=zf.testzip(); names=zf.namelist()
    print(f"zip_check path={zip_path} entries={len(names)} bad={bad} size={zip_path.stat().st_size}")
else:
    print('subset_smoke_no_submission_validation=1')
PY
REMOTE

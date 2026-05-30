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
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_tiny_crop_candidate}"
WORK_ROOT="${WORK_ROOT:-$MOSE_WORKSPACE/homework/tiny_crop_work}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_tiny_crop_candidate}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_tiny_crop_candidate.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/tiny_crop_candidate_latest.json}"
AUDIT_DIR="${AUDIT_DIR:-$MOSE_WORKSPACE/homework/logs/tiny_crop_candidate_by_video}"
GUIDE_ROOT="${GUIDE_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m11_cycle}"
VIDEOS="${VIDEOS:-}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-0}"
EXTRA_ARGS="${TINY_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LOCAL_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
LOCAL_GIT_DIRTY="$(git -C "$REPO_ROOT" status --porcelain=v1 2>/dev/null | wc -l | tr -d ' ')"
LOCAL_GIT_DIFF_HASH="$( (git -C "$REPO_ROOT" diff --binary; git -C "$REPO_ROOT" ls-files --others --exclude-standard -z | xargs -0 -r -I{} sh -c 'printf "UNTRACKED:%s\n" "$1"; cat "$1"' sh "$REPO_ROOT/{}") 2>/dev/null | sha256sum | awk '{print $1}' || true)"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)
REMOTE_ARGS=("$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$GPU" "$PRED_ROOT" "$WORK_ROOT" "$SUBMIT_ROOT" "$ZIP_PATH" "$AUDIT_JSON" "$AUDIT_DIR" "$GUIDE_ROOT" "${VIDEOS:-__EMPTY__}" "$MAKE_SUBMISSION" "${EXTRA_ARGS:-__EMPTY__}" "$PYTHON_BIN" "${LOCAL_GIT_SHA:-__EMPTY__}" "${LOCAL_GIT_BRANCH:-__EMPTY__}" "${LOCAL_GIT_DIRTY:-__EMPTY__}" "${LOCAL_GIT_DIFF_HASH:-__EMPTY__}")
REMOTE_CMD="bash -s --"
for arg in "${REMOTE_ARGS[@]}"; do REMOTE_CMD+=" $(printf '%q' "$arg")"; done
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$REMOTE_CMD" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; GPU="$4"; PRED_ROOT="$5"; WORK_ROOT="$6"; SUBMIT_ROOT="$7"; ZIP_PATH="$8"; AUDIT_JSON="$9"; AUDIT_DIR="${10}"; GUIDE_ROOT="${11}"; VIDEOS="${12}"; MAKE_SUBMISSION="${13}"; EXTRA_ARGS="${14}"; PYTHON_BIN="${15}"; GIT_SHA="${16}"; GIT_BRANCH="${17}"; GIT_DIRTY="${18}"; GIT_DIFF_HASH="${19}"
[[ "$VIDEOS" == "__EMPTY__" ]] && VIDEOS=""
[[ "$EXTRA_ARGS" == "__EMPTY__" ]] && EXTRA_ARGS=""
[[ "$GIT_SHA" != "__EMPTY__" ]] && export CVMOSE_GIT_SHA="$GIT_SHA"
[[ "$GIT_BRANCH" != "__EMPTY__" ]] && export CVMOSE_GIT_BRANCH="$GIT_BRANCH"
[[ "$GIT_DIRTY" != "__EMPTY__" ]] && export CVMOSE_GIT_DIRTY="$GIT_DIRTY"
[[ "$GIT_DIFF_HASH" != "__EMPTY__" ]] && export CVMOSE_GIT_DIFF_HASH="$GIT_DIFF_HASH"
cd "$CODE_ROOT"
require_under() {
  local path="$1" prefix="$2" label="$3" canon_path canon_prefix
  canon_path="$(realpath -m -- "$path")"; canon_prefix="$(realpath -m -- "$prefix")"
  if [[ -z "$canon_path" || "$canon_path" == "/" || "$canon_path" == "$WORKSPACE" || "$canon_path" == "$WORKSPACE/homework" || "$canon_path" == "$WORKSPACE/homework/logs" ]]; then
    echo "Refusing unsafe $label path: $path (canonical: $canon_path)" >&2; exit 2
  fi
  if [[ "$prefix" == *_ ]]; then
    [[ "$canon_path" == "$canon_prefix"* ]] || { echo "Refusing unsafe $label path: $path (required prefix: $canon_prefix)" >&2; exit 2; }
  else
    case "$canon_path/" in "$canon_prefix/"|"$canon_prefix"/*) ;; *) echo "Refusing unsafe $label path: $path (required under: $canon_prefix)" >&2; exit 2;; esac
  fi
}
require_prefix_any() {
  local path="$1" label="$2" canon_path canon_prefix prefix ok=0
  shift 2
  canon_path="$(realpath -m -- "$path")"
  if [[ -z "$canon_path" || "$canon_path" == "/" || "$canon_path" == "$WORKSPACE" || "$canon_path" == "$WORKSPACE/homework" || "$canon_path" == "$WORKSPACE/homework/logs" ]]; then
    echo "Refusing unsafe $label path: $path (canonical: $canon_path)" >&2
    exit 2
  fi
  for prefix in "$@"; do
    canon_prefix="$(realpath -m -- "$prefix")"
    if [[ "$canon_path" == "$canon_prefix"* ]]; then ok=1; break; fi
  done
  if [[ "$ok" != "1" ]]; then
    echo "Refusing unsafe $label path: $path (not an allowed tiny-crop output/log prefix)" >&2
    exit 2
  fi
}
require_under "$PRED_ROOT" "$WORKSPACE/homework/pred_sam2_tiny_crop_candidate" PRED_ROOT
require_under "$WORK_ROOT" "$WORKSPACE/homework/tiny_crop_work" WORK_ROOT
require_under "$SUBMIT_ROOT" "$WORKSPACE/homework/submission_433_tiny_crop_candidate" SUBMIT_ROOT
require_under "$ZIP_PATH" "$WORKSPACE/homework/" ZIP_PATH
require_prefix_any "$AUDIT_JSON" AUDIT_JSON "$WORKSPACE/homework/logs/tiny_crop_candidate"
require_prefix_any "$AUDIT_DIR" AUDIT_DIR "$WORKSPACE/homework/logs/tiny_crop_candidate"
read -r -a VIDEO_ARGS <<< "$VIDEOS"
read -r -a EXTRA <<< "$EXTRA_ARGS"
if [[ "$MAKE_SUBMISSION" == "auto" ]]; then MAKE_SUBMISSION=0; fi
if [[ "$MAKE_SUBMISSION" == "1" ]]; then
  echo "tiny-crop is candidate-only; refusing MAKE_SUBMISSION=1. Use M3 selector for final submission." >&2
  exit 2
fi
mkdir -p "$(dirname "$AUDIT_JSON")" "$AUDIT_DIR"
rm -rf "$PRED_ROOT" "$WORK_ROOT" "$AUDIT_DIR"
rm -f "$AUDIT_JSON"
mkdir -p "$AUDIT_DIR"
RUN_ARGS=(--workspace "$WORKSPACE" --device cuda --guide-root "$GUIDE_ROOT" --out-pred-root "$PRED_ROOT" --work-root "$WORK_ROOT" --audit-json "$AUDIT_JSON" --audit-dir "$AUDIT_DIR")
if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
if [[ "$MAKE_SUBMISSION" == "1" ]]; then RUN_ARGS+=(--make-submission --submit-root "$SUBMIT_ROOT" --zip-path "$ZIP_PATH" --overwrite-submission); fi
if (( ${#EXTRA[@]} > 0 )); then RUN_ARGS+=("${EXTRA[@]}"); fi
echo "tiny_crop_remote_run videos=${VIDEOS:-<all>} make_submission=$MAKE_SUBMISSION pred_root=$PRED_ROOT guide=$GUIDE_ROOT"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python tools/infer_mosev2_sam2_tiny_crop.py "${RUN_ARGS[@]}"
"$PYTHON_BIN" - <<PY
import json, pathlib, zipfile
pred=pathlib.Path(r"$PRED_ROOT"); audit=pathlib.Path(r"$AUDIT_JSON"); submit=pathlib.Path(r"$SUBMIT_ROOT"); zip_path=pathlib.Path(r"$ZIP_PATH"); make_submission="$MAKE_SUBMISSION"=="1"
print(f"pred_check dir={pred} videos={len([p for p in pred.iterdir() if p.is_dir()])} pngs={sum(1 for _ in pred.rglob('*.png'))}")
data=json.loads(audit.read_text(encoding='utf-8'))
print('audit_summary=' + json.dumps(data.get('summary',{}), ensure_ascii=False, sort_keys=True))
print('runtime=' + json.dumps(data.get('runtime',{}), ensure_ascii=False, sort_keys=True))
if make_submission:
    print(f"submission_check dir={submit} videos={len([p for p in submit.iterdir() if p.is_dir()])} pngs={sum(1 for _ in submit.rglob('*.png'))}")
    with zipfile.ZipFile(zip_path) as zf: bad=zf.testzip(); names=zf.namelist()
    print(f"zip_check path={zip_path} entries={len(names)} bad={bad} size={zip_path.stat().st_size}")
PY
if [[ "$MAKE_SUBMISSION" == "1" ]]; then
  VALIDATION_JSON="$WORKSPACE/homework/logs/tiny_crop_candidate_validation.json"
  conda run -n "$ENV_NAME" python tools/validate_mose_submission.py --workspace "$WORKSPACE" --pred-root "$PRED_ROOT" --submit-root "$SUBMIT_ROOT" --zip-path "$ZIP_PATH" --output-json "$VALIDATION_JSON"
  echo "validation_json=$VALIDATION_JSON"
fi
REMOTE

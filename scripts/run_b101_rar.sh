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
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_rar}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_rar}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_rar.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/rar_latest.json}"
AUDIT_DIR="${AUDIT_DIR:-$MOSE_WORKSPACE/homework/logs/rar_by_video}"
VIDEOS="${VIDEOS:-}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-auto}"
EXTRA_ARGS="${RAR_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LOCAL_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

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
  local path="$1" prefix="$2" label="$3"
  local canon_path canon_prefix
  canon_path="$(realpath -m -- "$path")"
  canon_prefix="$(realpath -m -- "$prefix")"
  if [[ -z "$canon_path" || "$canon_path" == "/" || "$canon_path" == "$canon_prefix" || "$canon_path" == "$WORKSPACE" || "$canon_path" == "$WORKSPACE/homework" || "$canon_path" == "$WORKSPACE/homework/logs" ]]; then
    echo "Refusing unsafe $label path: $path (canonical: $canon_path)" >&2
    exit 2
  fi
  case "$canon_path/" in
    "$canon_prefix"/*) ;;
    *) echo "Refusing unsafe $label path: $path (required under: $canon_prefix)" >&2; exit 2 ;;
  esac
}
require_prefix() {
  local path="$1" prefix="$2" label="$3"
  local canon_path canon_prefix
  canon_path="$(realpath -m -- "$path")"
  canon_prefix="$(realpath -m -- "$prefix")"
  [[ "$canon_path" == "$canon_prefix"* && "$canon_path" != "$canon_prefix" ]] || {
    echo "Refusing unsafe $label path: $path (required prefix: $canon_prefix)" >&2
    exit 2
  }
}
require_prefix "$PRED_ROOT" "$WORKSPACE/homework/pred_" "PRED_ROOT"
require_prefix "$SUBMIT_ROOT" "$WORKSPACE/homework/submission_" "SUBMIT_ROOT"
require_under "$ZIP_PATH" "$WORKSPACE/homework" "ZIP_PATH"
require_under "$AUDIT_JSON" "$WORKSPACE/homework/logs" "AUDIT_JSON"
require_under "$AUDIT_DIR" "$WORKSPACE/homework/logs" "AUDIT_DIR"

read -r -a VIDEO_ARGS <<< "$VIDEOS"
read -r -a EXTRA <<< "$EXTRA_ARGS"
ALLOWED_EXTRA_FLAGS=(
  --rar-mode
  --rar-output-policy
  --rar-stable-quality
  --rar-ambiguous-quality
  --rar-rcms-quality-thr
  --rar-rcms-max-anchors
  --rar-reservoir-size
  --rar-confirm-frames
)
is_allowed_extra_flag() {
  local flag="$1" allowed
  for allowed in "${ALLOWED_EXTRA_FLAGS[@]}"; do
    [[ "$flag" == "$allowed" || "$flag" == "$allowed="* ]] && return 0
  done
  return 1
}
for token in "${EXTRA[@]}"; do
  if [[ "$token" == --* ]] && ! is_allowed_extra_flag "$token"; then
    echo "Refusing unsafe RAR_EXTRA_ARGS flag: $token" >&2
    exit 2
  fi
done
if [[ "$MAKE_SUBMISSION" == "auto" ]]; then
  if (( ${#VIDEO_ARGS[@]} > 0 )); then MAKE_SUBMISSION=0; else MAKE_SUBMISSION=1; fi
fi
mkdir -p "$(dirname "$AUDIT_JSON")" "$AUDIT_DIR"
rm -rf "$PRED_ROOT" "$AUDIT_DIR"
rm -f "$AUDIT_JSON"
if [[ "$MAKE_SUBMISSION" == "1" ]]; then
  rm -rf "$SUBMIT_ROOT"
  rm -f "$ZIP_PATH"
fi
mkdir -p "$AUDIT_DIR"

RUN_ARGS=(
  --workspace "$WORKSPACE"
  --device cuda
  --pred-root "$PRED_ROOT"
  --submit-root "$SUBMIT_ROOT"
  --zip-path "$ZIP_PATH"
  --rar-audit-json "$AUDIT_JSON"
  --rar-audit-dir "$AUDIT_DIR"
)
if [[ "$MAKE_SUBMISSION" == "1" ]]; then RUN_ARGS+=(--make-submission --overwrite-submission); fi
if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
if (( ${#EXTRA[@]} > 0 )); then RUN_ARGS+=("${EXTRA[@]}"); fi

echo "rar_remote_run videos=${VIDEOS:-<all>} make_submission=$MAKE_SUBMISSION pred_root=$PRED_ROOT extra=${EXTRA_ARGS:-<none>}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python tools/infer_mosev2_sam2_rar.py "${RUN_ARGS[@]}"

"$PYTHON_BIN" - <<PY
import json, pathlib, zipfile
pred = pathlib.Path(r"$PRED_ROOT")
audit = pathlib.Path(r"$AUDIT_JSON")
submit = pathlib.Path(r"$SUBMIT_ROOT")
zip_path = pathlib.Path(r"$ZIP_PATH")
make_submission = "$MAKE_SUBMISSION" == "1"
if not pred.is_dir():
    raise SystemExit(f"missing pred root {pred}")
print(f"pred_check dir={pred} videos={len([p for p in pred.iterdir() if p.is_dir()])} pngs={sum(1 for _ in pred.rglob('*.png'))}")
if not audit.is_file():
    raise SystemExit(f"missing audit json {audit}")
data = json.loads(audit.read_text(encoding='utf-8'))
print("audit_summary=" + json.dumps(data.get('summary', {}), ensure_ascii=False, sort_keys=True))
if make_submission:
    video_dirs = sorted(p for p in submit.iterdir() if p.is_dir())
    png_count = sum(1 for _ in submit.rglob('*.png'))
    print(f"submission_check dir={submit} videos={len(video_dirs)} pngs={png_count}")
    if len(video_dirs) != 433 or png_count != 66526:
        raise SystemExit(f"bad submission shape videos={len(video_dirs)} pngs={png_count}")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip(); names = zf.namelist()
    print(f"zip_check path={zip_path} entries={len(names)} bad={bad} size={zip_path.stat().st_size}")
    if bad is not None or len([n for n in names if n.endswith('.png')]) != 66526:
        raise SystemExit("bad zip validation")
else:
    print("subset_smoke_no_submission_validation=1")
PY
if [[ "$MAKE_SUBMISSION" == "1" ]]; then
  VALIDATION_JSON="$WORKSPACE/homework/logs/rar_validation.json"
  conda run -n "$ENV_NAME" python tools/validate_mose_submission.py \
    --workspace "$WORKSPACE" \
    --pred-root "$PRED_ROOT" \
    --submit-root "$SUBMIT_ROOT" \
    --zip-path "$ZIP_PATH" \
    --output-json "$VALIDATION_JSON"
  echo "validation_json=$VALIDATION_JSON"
fi
REMOTE

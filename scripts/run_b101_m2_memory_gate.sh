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
GPUS="${GPUS:-$GPU}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
M2_PARALLEL="${M2_PARALLEL:-0}"
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m2_memory_gate}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m2_memory_gate}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_m2_memory_gate.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m2_memory_gate_latest.json}"
AUDIT_DIR="${AUDIT_DIR:-$MOSE_WORKSPACE/homework/logs/m2_memory_gate_by_video}"
LOG_DIR="${LOG_DIR:-$MOSE_WORKSPACE/homework/logs/m2_memory_gate_parallel}"
VIDEOS="${VIDEOS:-}"
EXTRA_ARGS="${M2_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# By default, full runs build/validate a 433-video submission. Subset VIDEO smoke tests only validate predictions/audit.
MAKE_SUBMISSION="${MAKE_SUBMISSION:-auto}"
VIDEOS_ARG="${VIDEOS:-__EMPTY__}"
EXTRA_ARGS_ARG="${EXTRA_ARGS:-__EMPTY__}"
LOCAL_GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LOCAL_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
LOCAL_GIT_SHA_ARG="${LOCAL_GIT_SHA:-__EMPTY__}"
LOCAL_GIT_BRANCH_ARG="${LOCAL_GIT_BRANCH:-__EMPTY__}"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s -- \
  "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$GPU" "$GPUS" "$JOBS_PER_GPU" "$M2_PARALLEL" \
  "$PRED_ROOT" "$SUBMIT_ROOT" "$ZIP_PATH" "$AUDIT_JSON" "$AUDIT_DIR" "$LOG_DIR" "$VIDEOS_ARG" "$EXTRA_ARGS_ARG" "$PYTHON_BIN" "$MAKE_SUBMISSION" "$LOCAL_GIT_SHA_ARG" "$LOCAL_GIT_BRANCH_ARG" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; GPU="$4"; GPUS="$5"; JOBS_PER_GPU="$6"; M2_PARALLEL="$7"
PRED_ROOT="$8"; SUBMIT_ROOT="$9"; ZIP_PATH="${10}"; AUDIT_JSON="${11}"; AUDIT_DIR="${12}"; LOG_DIR="${13}"; VIDEOS="${14}"; EXTRA_ARGS="${15}"; PYTHON_BIN="${16}"; MAKE_SUBMISSION="${17}"; GIT_SHA="${18}"; GIT_BRANCH="${19}"
if [[ "$VIDEOS" == "__EMPTY__" ]]; then VIDEOS=""; fi
if [[ "$EXTRA_ARGS" == "__EMPTY__" ]]; then EXTRA_ARGS=""; fi
if [[ "$GIT_SHA" != "__EMPTY__" ]]; then export CVMOSE_GIT_SHA="$GIT_SHA"; fi
if [[ "$GIT_BRANCH" != "__EMPTY__" ]]; then export CVMOSE_GIT_BRANCH="$GIT_BRANCH"; fi
cd "$CODE_ROOT"

require_under() {
  local path="$1"
  local prefix="$2"
  local label="$3"
  if [[ -z "$path" || "$path" == "/" || "$path" != "$prefix"* ]]; then
    echo "Refusing unsafe $label path: $path (required prefix: $prefix)" >&2
    exit 2
  fi
}
require_under "$PRED_ROOT" "$WORKSPACE/homework/pred_" "PRED_ROOT"
require_under "$SUBMIT_ROOT" "$WORKSPACE/homework/submission_" "SUBMIT_ROOT"
require_under "$ZIP_PATH" "$WORKSPACE/homework/" "ZIP_PATH"
require_under "$AUDIT_JSON" "$WORKSPACE/homework/logs/" "AUDIT_JSON"
require_under "$AUDIT_DIR" "$WORKSPACE/homework/logs/" "AUDIT_DIR"
require_under "$LOG_DIR" "$WORKSPACE/homework/logs/" "LOG_DIR"
mkdir -p "$(dirname "$AUDIT_JSON")" "$AUDIT_DIR" "$LOG_DIR"
rm -rf "$PRED_ROOT" "$SUBMIT_ROOT"
rm -f "$ZIP_PATH" "$AUDIT_JSON"
rm -f "$AUDIT_DIR"/*.json 2>/dev/null || true

read -r -a VIDEO_ARGS <<< "$VIDEOS"
read -r -a EXTRA <<< "$EXTRA_ARGS"
if [[ "$MAKE_SUBMISSION" == "auto" ]]; then
  if (( ${#VIDEO_ARGS[@]} > 0 )); then
    MAKE_SUBMISSION=0
  else
    MAKE_SUBMISSION=1
  fi
fi
echo "m2_remote_run videos=${VIDEOS:-<all>} make_submission=$MAKE_SUBMISSION parallel=$M2_PARALLEL pred_root=$PRED_ROOT"

if [[ "$M2_PARALLEL" == "1" ]]; then
  PARALLEL_ARGS=(
    --workspace "$WORKSPACE"
    --script "$CODE_ROOT/tools/infer_mosev2_sam2_m2_memory_gate.py"
    --conda-env "$ENV_NAME"
    --gpus "$GPUS"
    --jobs-per-gpu "$JOBS_PER_GPU"
    --pred-root "$PRED_ROOT"
    --log-dir "$LOG_DIR"
    --extra-arg "--m2-audit-dir $AUDIT_DIR --m2-audit-json /dev/null --no-zip"
  )
  if [[ "$MAKE_SUBMISSION" == "1" ]]; then
    PARALLEL_ARGS+=(--make-submission-after --submit-root "$SUBMIT_ROOT" --zip-path "$ZIP_PATH")
  fi
  if (( ${#VIDEO_ARGS[@]} > 0 )); then
    PARALLEL_ARGS+=(--videos "${VIDEO_ARGS[@]}")
  fi
  if (( ${#EXTRA[@]} > 0 )); then
    # Preserve caller-provided arguments as one shell-like string consumed by launch_parallel_infer.
    PARALLEL_ARGS+=(--extra-arg "$EXTRA_ARGS")
  fi
  "$PYTHON_BIN" tools/launch_parallel_infer.py "${PARALLEL_ARGS[@]}"
  "$PYTHON_BIN" - <<PY
import json, pathlib
root = pathlib.Path(r"$AUDIT_DIR")
out = pathlib.Path(r"$AUDIT_JSON")
videos = {}
summary = {"videos": 0, "noncond_total": 0, "memory_written": 0, "memory_skipped": 0, "skip_reason_counts": {}}
for p in sorted(root.glob('*.json')):
    data = json.loads(p.read_text(encoding='utf-8'))
    videos[data['video']] = data
    s = data.get('summary', {})
    summary['videos'] += 1
    for k in ['noncond_total', 'memory_written', 'memory_skipped']:
        summary[k] += int(s.get(k, 0))
    for reason, count in s.get('skip_reason_counts', {}).items():
        summary['skip_reason_counts'][reason] = summary['skip_reason_counts'].get(reason, 0) + int(count)
summary['skip_ratio'] = summary['memory_skipped'] / summary['noncond_total'] if summary['noncond_total'] else 0.0
payload = {"method": "m2_reliable_memory_gate", "summary": summary, "videos": videos}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print('m2_audit_json=' + str(out))
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY
else
  RUN_ARGS=(
    --workspace "$WORKSPACE"
    --device cuda
    --pred-root "$PRED_ROOT"
    --submit-root "$SUBMIT_ROOT"
    --zip-path "$ZIP_PATH"
    --m2-audit-json "$AUDIT_JSON"
    --m2-audit-dir "$AUDIT_DIR"
  )
  if [[ "$MAKE_SUBMISSION" == "1" ]]; then
    RUN_ARGS+=(--make-submission --overwrite-submission)
  fi
  if (( ${#VIDEO_ARGS[@]} > 0 )); then
    RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}")
  fi
  if (( ${#EXTRA[@]} > 0 )); then
    RUN_ARGS+=("${EXTRA[@]}")
  fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python tools/infer_mosev2_sam2_m2_memory_gate.py "${RUN_ARGS[@]}"
fi

"$PYTHON_BIN" - <<PY
import json, pathlib, zipfile
pred = pathlib.Path(r"$PRED_ROOT")
submit = pathlib.Path(r"$SUBMIT_ROOT")
zip_path = pathlib.Path(r"$ZIP_PATH")
audit = pathlib.Path(r"$AUDIT_JSON")
make_submission = "$MAKE_SUBMISSION" == "1"
if not pred.is_dir():
    raise SystemExit(f"missing pred root {pred}")
pred_videos = sorted(p for p in pred.iterdir() if p.is_dir())
pred_png_count = sum(1 for _ in pred.rglob('*.png'))
print(f"pred_check dir={pred} videos={len(pred_videos)} pngs={pred_png_count}")
if not pred_videos or pred_png_count == 0:
    raise SystemExit("empty M2 predictions")
if not audit.is_file():
    raise SystemExit(f"missing audit json {audit}")
summary = json.loads(audit.read_text(encoding='utf-8')).get('summary', {})
print("audit_summary=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
if make_submission:
    video_dirs = sorted(p for p in submit.iterdir() if p.is_dir())
    png_count = sum(1 for _ in submit.rglob('*.png'))
    print(f"submission_check dir={submit} videos={len(video_dirs)} pngs={png_count}")
    if len(video_dirs) != 433:
        raise SystemExit(f"expected 433 video dirs, got {len(video_dirs)}")
    if not zip_path.is_file():
        raise SystemExit(f"missing zip {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    print(f"zip_check path={zip_path} entries={len(names)} bad={bad} size={zip_path.stat().st_size}")
    if bad is not None:
        raise SystemExit(f"bad zip member {bad}")
else:
    print("subset_smoke_no_submission_validation=1")
PY
REMOTE

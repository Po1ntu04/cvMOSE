#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY="${B101_SSH_KEY:-$HOME/.ssh/id_ed25519_b101}"
REMOTE_HOST="${B101_HOST:-yuzhixiang@b101.guhk.cc}"
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
CONDA_ENV="${CONDA_ENV:-mose_sam2}"
VARIANT="${M3_VARIANT:-conservative}"
PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m3_state_${VARIANT}}"
SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m3_state_${VARIANT}}"
ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_m3_state_${VARIANT}.zip}"
AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m3_state_${VARIANT}.json}"
AUDIT_DIR="${AUDIT_DIR:-$MOSE_WORKSPACE/homework/logs/m3_state_${VARIANT}_by_video}"
BASELINE_ROOT="${BASELINE_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_b101}"
M2_ROOT="${M2_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m2_memory_gate}"
M2_LIGHT_ROOT="${M2_LIGHT_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m2_light}"
M11_ROOT="${M11_ROOT:-$MOSE_WORKSPACE/homework/pred_sam2_m11_cycle}"
SAM31_ROOT="${SAM31_ROOT:-$MOSE_WORKSPACE/homework/pred_sam31_b101}"
M2_AUDIT_JSON="${M2_AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m2_memory_gate_latest.json}"
M2_LIGHT_AUDIT_JSON="${M2_LIGHT_AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m2_light_latest.json}"
M11_AUDIT_JSON="${M11_AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m11_cycle_gate_latest.json}"
VIDEOS="${VIDEOS:-}"
MAKE_SUBMISSION="${MAKE_SUBMISSION:-auto}"
EXTRA_ARGS="${M3_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LOCAL_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

REMOTE_ARGS=(
  "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$VARIANT" "$PRED_ROOT" "$SUBMIT_ROOT" "$ZIP_PATH" "$AUDIT_JSON" "$AUDIT_DIR"
  "$BASELINE_ROOT" "$M2_ROOT" "$M2_LIGHT_ROOT" "$M11_ROOT" "$SAM31_ROOT"
  "$M2_AUDIT_JSON" "$M2_LIGHT_AUDIT_JSON" "$M11_AUDIT_JSON" "${VIDEOS:-__EMPTY__}" "$MAKE_SUBMISSION" "${EXTRA_ARGS:-__EMPTY__}" "$PYTHON_BIN" "${LOCAL_GIT_SHA:-__EMPTY__}" "${LOCAL_GIT_BRANCH:-__EMPTY__}"
)
REMOTE_CMD="bash -s --"
for arg in "${REMOTE_ARGS[@]}"; do
  REMOTE_CMD+=" $(printf '%q' "$arg")"
done

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$REMOTE_CMD" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; VARIANT="$4"; PRED_ROOT="$5"; SUBMIT_ROOT="$6"; ZIP_PATH="$7"; AUDIT_JSON="$8"; AUDIT_DIR="$9"
BASELINE_ROOT="${10}"; M2_ROOT="${11}"; M2_LIGHT_ROOT="${12}"; M11_ROOT="${13}"; SAM31_ROOT="${14}"; M2_AUDIT_JSON="${15}"; M2_LIGHT_AUDIT_JSON="${16}"; M11_AUDIT_JSON="${17}"; VIDEOS="${18}"; MAKE_SUBMISSION="${19}"; EXTRA_ARGS="${20}"; PYTHON_BIN="${21}"; GIT_SHA="${22}"; GIT_BRANCH="${23}"
if [[ "$VIDEOS" == "__EMPTY__" ]]; then VIDEOS=""; fi
if [[ "$EXTRA_ARGS" == "__EMPTY__" ]]; then EXTRA_ARGS=""; fi
if [[ "$GIT_SHA" != "__EMPTY__" ]]; then export CVMOSE_GIT_SHA="$GIT_SHA"; fi
if [[ "$GIT_BRANCH" != "__EMPTY__" ]]; then export CVMOSE_GIT_BRANCH="$GIT_BRANCH"; fi
cd "$CODE_ROOT"
require_under() {
  local path="$1"; local prefix="$2"; local label="$3"
  local canon_path canon_prefix
  canon_path="$(realpath -m -- "$path")"
  canon_prefix="$(realpath -m -- "$prefix")"
  if [[ -z "$canon_path" || "$canon_path" == "/" || "$canon_path" == "$canon_prefix" || "$canon_path" == "$WORKSPACE" || "$canon_path" == "$WORKSPACE/homework" || "$canon_path" == "$WORKSPACE/homework/logs" ]]; then
    echo "Refusing unsafe $label path: $path (canonical: $canon_path)" >&2
    exit 2
  fi
  if [[ "$prefix" == *_ ]]; then
    [[ "$canon_path" == "$canon_prefix"* ]] || {
      echo "Refusing unsafe $label path: $path (required prefix: $canon_prefix)" >&2
      exit 2
    }
  else
    case "$canon_path/" in
      "$canon_prefix"/*) ;;
      *)
        echo "Refusing unsafe $label path: $path (required under: $canon_prefix)" >&2
        exit 2
        ;;
    esac
  fi
}
require_under "$PRED_ROOT" "$WORKSPACE/homework/pred_" "PRED_ROOT"
require_under "$SUBMIT_ROOT" "$WORKSPACE/homework/submission_" "SUBMIT_ROOT"
require_under "$ZIP_PATH" "$WORKSPACE/homework/" "ZIP_PATH"
require_under "$AUDIT_JSON" "$WORKSPACE/homework/logs/" "AUDIT_JSON"
require_under "$AUDIT_DIR" "$WORKSPACE/homework/logs/" "AUDIT_DIR"
mkdir -p "$(dirname "$AUDIT_JSON")" "$AUDIT_DIR"
rm -rf "$PRED_ROOT" "$SUBMIT_ROOT" "$AUDIT_DIR"
rm -f "$ZIP_PATH" "$AUDIT_JSON"
mkdir -p "$AUDIT_DIR"
read -r -a VIDEO_ARGS <<< "$VIDEOS"
read -r -a EXTRA <<< "$EXTRA_ARGS"
if [[ "$MAKE_SUBMISSION" == "auto" ]]; then
  if (( ${#VIDEO_ARGS[@]} > 0 )); then MAKE_SUBMISSION=0; else MAKE_SUBMISSION=1; fi
fi
RUN_ARGS=(
  --workspace "$WORKSPACE"
  --variant "$VARIANT"
  --baseline-root "$BASELINE_ROOT"
  --out-pred-root "$PRED_ROOT"
  --audit-json "$AUDIT_JSON"
  --audit-dir "$AUDIT_DIR"
)
[[ -d "$M2_ROOT" ]] && RUN_ARGS+=(--m2-root "$M2_ROOT") || echo "warn: missing M2_ROOT=$M2_ROOT"
[[ -d "$M2_LIGHT_ROOT" ]] && RUN_ARGS+=(--m2-light-root "$M2_LIGHT_ROOT") || echo "warn: missing M2_LIGHT_ROOT=$M2_LIGHT_ROOT"
[[ -d "$M11_ROOT" ]] && RUN_ARGS+=(--m11-root "$M11_ROOT") || echo "warn: missing M11_ROOT=$M11_ROOT"
[[ -d "$SAM31_ROOT" ]] && RUN_ARGS+=(--sam31-adapter-root "$SAM31_ROOT") || echo "warn: missing SAM31_ROOT=$SAM31_ROOT"
[[ -f "$M2_AUDIT_JSON" ]] && RUN_ARGS+=(--m2-audit-json "$M2_AUDIT_JSON") || echo "warn: missing M2_AUDIT_JSON=$M2_AUDIT_JSON"
[[ -f "$M2_LIGHT_AUDIT_JSON" ]] && RUN_ARGS+=(--m2-light-audit-json "$M2_LIGHT_AUDIT_JSON") || echo "warn: missing M2_LIGHT_AUDIT_JSON=$M2_LIGHT_AUDIT_JSON"
[[ -f "$M11_AUDIT_JSON" ]] && RUN_ARGS+=(--m11-audit-json "$M11_AUDIT_JSON") || echo "warn: missing M11_AUDIT_JSON=$M11_AUDIT_JSON"
if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
if [[ "$MAKE_SUBMISSION" == "1" ]]; then RUN_ARGS+=(--make-submission --submit-root "$SUBMIT_ROOT" --zip-path "$ZIP_PATH" --overwrite-submission); fi
if (( ${#EXTRA[@]} > 0 )); then RUN_ARGS+=("${EXTRA[@]}"); fi

echo "m3_remote_run variant=$VARIANT videos=${VIDEOS:-<all>} make_submission=$MAKE_SUBMISSION pred_root=$PRED_ROOT"
conda run -n "$ENV_NAME" python tools/apply_m3_state_select.py "${RUN_ARGS[@]}"

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
if not audit.is_file():
    raise SystemExit(f"missing audit {audit}")
data = json.loads(audit.read_text(encoding='utf-8'))
print("audit_summary=" + json.dumps(data.get('summary', {}), ensure_ascii=False, sort_keys=True))
if make_submission:
    video_dirs = sorted(p for p in submit.iterdir() if p.is_dir())
    png_count = sum(1 for _ in submit.rglob('*.png'))
    print(f"submission_check dir={submit} videos={len(video_dirs)} pngs={png_count}")
    if len(video_dirs) != 433 or png_count != 66526:
        raise SystemExit(f"bad submission shape videos={len(video_dirs)} pngs={png_count}")
    if not zip_path.is_file():
        raise SystemExit(f"missing zip {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip(); names = zf.namelist()
    print(f"zip_check path={zip_path} entries={len(names)} bad={bad} size={zip_path.stat().st_size}")
    if bad is not None or len([n for n in names if n.endswith('.png')]) != 66526:
        raise SystemExit("bad zip validation")
else:
    print("subset_smoke_no_submission_validation=1")
PY
if [[ "$MAKE_SUBMISSION" == "1" ]]; then
  VALIDATION_JSON="$WORKSPACE/homework/logs/m3_state_${VARIANT}_validation.json"
  conda run -n "$ENV_NAME" python tools/validate_mose_submission.py \
    --workspace "$WORKSPACE" \
    --pred-root "$PRED_ROOT" \
    --submit-root "$SUBMIT_ROOT" \
    --zip-path "$ZIP_PATH" \
    --output-json "$VALIDATION_JSON"
  echo "validation_json=$VALIDATION_JSON"
fi
REMOTE

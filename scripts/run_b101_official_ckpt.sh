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
VARIANTS="${VARIANTS:-bplus large}"
VIDEOS="${VIDEOS:-}"
RUN_INFER="${RUN_INFER:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
SYNC_CODE="${SYNC_CODE:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$MOSE_WORKSPACE/homework/external_checkpoints/FudanCVL_MOSEv2_baseline}"
LOG_DIR="${LOG_DIR:-$MOSE_WORKSPACE/homework/logs/m6_official}"
INSTALL_HF_HUB="${INSTALL_HF_HUB:-1}"
DOWNLOAD="${DOWNLOAD:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

if [[ "$SYNC_CODE" == "1" ]]; then
  "$SCRIPT_DIR/sync_code_b101.sh"
fi

LOCAL_GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LOCAL_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
REMOTE_ARGS=(
  "$REMOTE_CODE_ROOT" "$MOSE_WORKSPACE" "$CONDA_ENV" "$GPU" "$VARIANTS" "${VIDEOS:-__EMPTY__}"
  "$RUN_INFER" "$RUN_EXTRACT" "$CHECKPOINT_DIR" "$LOG_DIR" "$INSTALL_HF_HUB" "$DOWNLOAD" "$PYTHON_BIN"
  "${LOCAL_GIT_SHA:-__EMPTY__}" "${LOCAL_GIT_BRANCH:-__EMPTY__}"
)
REMOTE_CMD="bash -s --"
for arg in "${REMOTE_ARGS[@]}"; do
  REMOTE_CMD+=" $(printf '%q' "$arg")"
done

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$REMOTE_CMD" <<'REMOTE'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="$1"; WORKSPACE="$2"; ENV_NAME="$3"; GPU="$4"; VARIANTS="$5"; VIDEOS="$6"
RUN_INFER="$7"; RUN_EXTRACT="$8"; CHECKPOINT_DIR="$9"; LOG_DIR="${10}"; INSTALL_HF_HUB="${11}"; DOWNLOAD="${12}"; PYTHON_BIN="${13}"
GIT_SHA="${14}"; GIT_BRANCH="${15}"
if [[ "$VIDEOS" == "__EMPTY__" ]]; then VIDEOS=""; fi
if [[ "$GIT_SHA" != "__EMPTY__" ]]; then export CVMOSE_GIT_SHA="$GIT_SHA"; fi
if [[ "$GIT_BRANCH" != "__EMPTY__" ]]; then export CVMOSE_GIT_BRANCH="$GIT_BRANCH"; fi
cd "$CODE_ROOT"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

if [[ "$DOWNLOAD" == "1" && "$INSTALL_HF_HUB" == "1" ]]; then
  conda run -n "$ENV_NAME" python - <<'PY' >/tmp/m6_hfhub_check.log 2>&1 || conda run -n "$ENV_NAME" python -m pip install -q huggingface_hub
import huggingface_hub  # noqa: F401
PY
fi

if [[ "$DOWNLOAD" == "1" ]]; then
cat > "$LOG_DIR/download_fudan_mosev2_baseline.py" <<'PY'
from __future__ import annotations
import hashlib, json, os, pathlib, sys
repo = "FudanCVL/MOSEv2_baseline"
out_dir = pathlib.Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
wanted = [
    "sam2.1_hiera_b+_MOSEv2_mss_lvt16.pt",
    "sam2.1_hiera_l_MOSEv2_mss_lvt16.pt",
    "sam2_b+_MOSEv2_rcms_mqf_mss_lvt_submission.zip",
    "sam2_l_MOSEv2_rcms_mqf_mss_lvt_submission.zip",
]
result = {"repo": repo, "wanted": wanted, "downloaded": {}, "errors": []}
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)
    result["repo_files"] = files
    for name in wanted:
        if name not in files:
            result["errors"].append(f"missing_on_hub:{name}")
            continue
        try:
            path = pathlib.Path(hf_hub_download(repo_id=repo, filename=name, local_dir=out_dir, token=os.environ.get("HF_TOKEN") or None))
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            result["downloaded"][name] = {"path": str(path), "size": path.stat().st_size, "sha256": h}
        except Exception as exc:  # keep global phase running
            result["errors"].append(f"download_failed:{name}:{type(exc).__name__}:{exc}")
except Exception as exc:
    result["errors"].append(f"hf_hub_failed:{type(exc).__name__}:{exc}")
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
(out_dir / "download_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
PY
conda run -n "$ENV_NAME" python "$LOG_DIR/download_fudan_mosev2_baseline.py" "$CHECKPOINT_DIR" | tee "$LOG_DIR/download_manifest.stdout.json"
else
  echo "download_skipped CHECKPOINT_DIR=$CHECKPOINT_DIR"
fi

read -r -a VARIANT_ARR <<< "$VARIANTS"
read -r -a VIDEO_ARGS <<< "$VIDEOS"
for variant in "${VARIANT_ARR[@]}"; do
  case "$variant" in
    bplus)
      ckpt="$CHECKPOINT_DIR/sam2.1_hiera_b+_MOSEv2_mss_lvt16.pt"
      pred="$WORKSPACE/homework/pred_sam2_official_bplus_mosev2"
      submit="$WORKSPACE/homework/submission_433_official_bplus_mosev2"
      zip="$WORKSPACE/homework/submission_mosev2_official_bplus_15only.zip"
      official_zip="$CHECKPOINT_DIR/sam2_b+_MOSEv2_rcms_mqf_mss_lvt_submission.zip"
      ;;
    large)
      ckpt="$CHECKPOINT_DIR/sam2.1_hiera_l_MOSEv2_mss_lvt16.pt"
      pred="$WORKSPACE/homework/pred_sam2_official_large_mosev2"
      submit="$WORKSPACE/homework/submission_433_official_large_mosev2"
      zip="$WORKSPACE/homework/submission_mosev2_official_large_15only.zip"
      official_zip="$CHECKPOINT_DIR/sam2_l_MOSEv2_rcms_mqf_mss_lvt_submission.zip"
      ;;
    *) echo "unknown variant $variant" >&2; exit 2;;
  esac
  if [[ "$RUN_INFER" == "1" && -f "$ckpt" ]]; then
    echo "official_ckpt_infer variant=$variant pred=$pred zip=$zip"
    rm -rf "$pred" "$submit"; rm -f "$zip" "$LOG_DIR/validate_${variant}.json"
    RUN_ARGS=(--mode infer --variant "$variant" --workspace "$WORKSPACE" --checkpoint-dir "$CHECKPOINT_DIR" --checkpoint "$ckpt" --device cuda --pred-root "$pred" --submit-root "$submit" --zip-path "$zip" --make-submission --overwrite-submission)
    if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_NAME" python tools/infer_mosev2_sam2_official_ckpt.py "${RUN_ARGS[@]}"
    conda run -n "$ENV_NAME" python tools/validate_mose_submission.py --workspace "$WORKSPACE" --pred-root "$pred" --submit-root "$submit" --zip-path "$zip" --output-json "$LOG_DIR/validate_${variant}.json"
  else
    echo "official_ckpt_infer_skipped variant=$variant ckpt_exists=$(test -f "$ckpt" && echo 1 || echo 0) run=$RUN_INFER"
  fi
  if [[ "$RUN_EXTRACT" == "1" && -f "$official_zip" ]]; then
    epred="$WORKSPACE/homework/pred_official_submission_${variant}_15only"
    esubmit="$WORKSPACE/homework/submission_433_official_submission_${variant}_15only"
    ezip="$WORKSPACE/homework/submission_mosev2_official_submission_${variant}_15only.zip"
    echo "official_submission_extract variant=$variant pred=$epred zip=$ezip"
    rm -rf "$epred" "$esubmit"; rm -f "$ezip" "$LOG_DIR/validate_submission_${variant}.json" "$LOG_DIR/extract_submission_${variant}.json"
    RUN_ARGS=(--mode extract-submission --variant "$variant" --workspace "$WORKSPACE" --checkpoint-dir "$CHECKPOINT_DIR" --official-submission-zip "$official_zip" --pred-root "$epred" --submit-root "$esubmit" --zip-path "$ezip" --make-submission --overwrite-submission --audit-json "$LOG_DIR/extract_submission_${variant}.json")
    if (( ${#VIDEO_ARGS[@]} > 0 )); then RUN_ARGS+=(--videos "${VIDEO_ARGS[@]}"); fi
    conda run -n "$ENV_NAME" python tools/infer_mosev2_sam2_official_ckpt.py "${RUN_ARGS[@]}"
    conda run -n "$ENV_NAME" python tools/validate_mose_submission.py --workspace "$WORKSPACE" --pred-root "$epred" --submit-root "$esubmit" --zip-path "$ezip" --output-json "$LOG_DIR/validate_submission_${variant}.json"
  else
    echo "official_submission_extract_skipped variant=$variant zip_exists=$(test -f "$official_zip" && echo 1 || echo 0) run=$RUN_EXTRACT"
  fi
done

"$PYTHON_BIN" - <<PY
import json, pathlib, zipfile
log=pathlib.Path(r"$LOG_DIR")
print('m6_official_outputs')
for p in sorted(log.glob('validate*.json')):
    data=json.loads(p.read_text())
    print(p.name, 'ok=', data.get('ok'))
for z in sorted((pathlib.Path(r"$WORKSPACE")/'homework').glob('submission_mosev2_official*.zip')):
    with zipfile.ZipFile(z) as zf: entries=len([n for n in zf.namelist() if n.endswith('.png')])
    print('zip', z, 'size', z.stat().st_size, 'entries', entries)
PY
REMOTE

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOSE_WORKSPACE="${MOSE_WORKSPACE:-/data1/yuzhixiang/cv_mosev2/MOSEv2}"
VARIANT="${M3_VARIANT:-balanced}"
export M3_VARIANT="$VARIANT"
export REQUIRE_TINY_CROP="${REQUIRE_TINY_CROP:-1}"
export PRED_ROOT="${PRED_ROOT:-$MOSE_WORKSPACE/homework/pred_m4_tiny_${VARIANT}}"
export SUBMIT_ROOT="${SUBMIT_ROOT:-$MOSE_WORKSPACE/homework/submission_433_m4_tiny_${VARIANT}}"
export ZIP_PATH="${ZIP_PATH:-$MOSE_WORKSPACE/homework/submission_mosev2_m4_tiny_${VARIANT}.zip}"
export AUDIT_JSON="${AUDIT_JSON:-$MOSE_WORKSPACE/homework/logs/m4_tiny_${VARIANT}.json}"
export AUDIT_DIR="${AUDIT_DIR:-$MOSE_WORKSPACE/homework/logs/m4_tiny_${VARIANT}_by_video}"
exec "$SCRIPT_DIR/run_b101_m3_state.sh"

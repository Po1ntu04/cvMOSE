#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY="${B101_SSH_KEY:-$HOME/.ssh/id_ed25519_b101}"
REMOTE_HOST="${B101_HOST:-yuzhixiang@b101.guhk.cc}"
REMOTE_CODE_ROOT="${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes)
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_CODE_ROOT'"
rsync -av --delete --delete-excluded \
  -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes" \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --include='/README.md' \
  --include='/pyproject.toml' \
  --include='/configs/***' \
  --include='/scripts/***' \
  --include='/src/***' \
  --include='/tools/***' \
  --exclude='*' \
  "$REPO_ROOT/" "$REMOTE_HOST:$REMOTE_CODE_ROOT/"
echo "Code-only sync complete: $REPO_ROOT -> $REMOTE_HOST:$REMOTE_CODE_ROOT"

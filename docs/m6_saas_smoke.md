# M6 Phase 4 — SAAS adapter status

## Goal
Evaluate SAAS as an external multi-shot robustness candidate for strong view/shot changes, while keeping it outside the cvMOSE tree and under no-finetune constraints.

## External source

- Repository: `https://github.com/FudanCVL/SAAS`
- Local clone: `external/SAAS`
- Commit inspected: `b54360bc643929bb4ce55df7a84b8193efe9f3af`
- README status: code is released, but the README still marks pretrained-weight release as incomplete/coming; it lists `SAAS_b+_ytvos_tma.pt` with Google Drive only and no direct HuggingFace file URL in the cloned README.

## Implemented harness

- `tools/infer_mosev2_saas_adapter.py`
- `scripts/run_b101_saas_adapter.sh`

The adapter is non-vendored and mirrors the SAM2Long wrapper: call external `tools/vos_inference.py`, restore exact frame-0 GT, and optionally build a 433-video zip.

## Why no smoke result is promoted

No SAAS checkpoint was obtained in a reproducible scriptable way during this M6 run:

1. b101 has no direct internet access.
2. README does not expose a direct HuggingFace checkpoint file URL for `SAAS_b+_ytvos_tma.pt`.
3. Google Drive weight download was not automated to avoid brittle/manual credential-dependent behavior.
4. SAAS also requires a separate environment plus custom TreeFilter build, which is not worth doing without a verified checkpoint.

## Decision

- Safe/balanced/aggressive fusion: no SAAS source used.
- Status: adapter ready, smoke blocked by reproducible checkpoint availability.
- If a direct checkpoint URL is provided later, run the documented smoke list: `1qlssuz2 2smf7uq9 3epdtmyr pe0d85lk r13u5z4y q0sizv6m 8jsm23a7`.

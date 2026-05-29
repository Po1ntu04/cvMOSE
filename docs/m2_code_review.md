# M2 Code Review — Reliable Memory Gate

Date: 2026-05-30
Branch: `method/m2-reliable-memory`
Commit: `ba8069f`

## Verdict

Initial independent review result: **REQUEST CHANGES**.

After fixes and b101 validation: **COMMENT / experiment-ready**, not yet merge-to-main as a claimed improvement because score/visual comparison is still pending.

## Findings and resolution

| Severity | Finding | Resolution |
|---|---|---|
| HIGH | One-video smoke command still built a full 433-video submission and would fail. | `scripts/run_b101_m2_memory_gate.sh` now uses `MAKE_SUBMISSION=auto`: subset `VIDEOS` runs validate predictions/audit only; full runs build and validate the 433-video zip. |
| HIGH | Parallel mode could fail when `launch_parallel_infer.py --make-submission-after` used `args.python=None`. | `tools/launch_parallel_infer.py` now resolves build Python to `args.python or ("python" if conda_env else sys.executable)`. |
| MEDIUM | `conda run ... python - <<PY` aggregation/validation could silently not execute on this setup. | Remote script now uses plain `$PYTHON_BIN - <<PY` for stdlib-only aggregation/validation. |
| MEDIUM | `object_score_logits` parse errors failed open. | `_obj_score_value` now returns `(score, status)`; present-but-unparseable scores fail `obj_ok` and are audited by status. Missing score remains compatibility-allowed. |
| MEDIUM | Remote cleanup used env-overridable `rm -rf` paths without guards. | Added prefix guards requiring output paths under `$WORKSPACE/homework/pred_*`, `submission_*`, or `logs/`. |
| WATCH | Monkey patch copies SAM2 private propagation internals. | Accepted as experiment-branch risk; documented in `docs/m2_reliable_memory_gate.md`. Future productionization should subclass/fork with compatibility tests. |
| WATCH | Audit provenance lacked git metadata on code-only remote sync. | Script now passes local git SHA/branch through env; audit collection reads `CVMOSE_GIT_SHA/CVMOSE_GIT_BRANCH`. |

## Validation evidence

Static/local:

```bash
python3 -m py_compile tools/infer_mosev2_sam2_m2_memory_gate.py tools/launch_parallel_infer.py tools/build_submission.py tools/infer_mosev2_sam2.py
bash -n scripts/run_b101_m2_memory_gate.sh scripts/sync_code_b101.sh
git diff --check main...HEAD
git diff --check
```

Remote smoke after fixes (`ba8069f` provenance verified in review-smoke audit):

```bash
VIDEOS="r13u5z4y" ./scripts/run_b101_m2_memory_gate.sh
M2_PARALLEL=1 GPUS=4 JOBS_PER_GPU=1 VIDEOS="r13u5z4y" ./scripts/run_b101_m2_memory_gate.sh
```

Both produced:

```text
pred_check ... videos=1 pngs=45
audit_summary={"memory_skipped": 17, "memory_written": 27, "noncond_total": 44, "skip_ratio": 0.38636363636363635, ...}
subset_smoke_no_submission_validation=1
```

Remote full run after fixes:

```bash
./scripts/run_b101_m2_memory_gate.sh
```

Produced:

```text
pred_check ... videos=15 pngs=1004
audit_summary={"memory_skipped": 710, "memory_written": 825, "noncond_total": 1535, "skip_ratio": 0.46254071661237783, "skip_reason_counts": {"area_abs_ok": 596, "area_ratio_ok": 186, "motion_ok": 72, "obj_ok": 505}, "videos": 15}
submission_check ... videos=433 pngs=66526
zip_check ... entries=66526 bad=None size=71983174
```

Runtime/GPU from aggregate audit:

```text
seconds=230.69
frames=1004
fps=4.35
cuda_max_memory_allocated_mib=970.86
cuda_max_memory_reserved_mib=1190.0
```

## Remaining risks

- No Codabench score yet.
- No visual comparison yet for whether skipping memory improves or harms identity recovery.
- Gate is aggressive on the full split (`46.25%` non-cond memories skipped); this is scientifically useful but should be analyzed before claiming improvement.

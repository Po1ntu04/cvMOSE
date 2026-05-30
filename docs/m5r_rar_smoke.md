# M5R/RAR Smoke Evidence

Date: 2026-05-30
Status: updated execution evidence; Phase A/B smoke only, not a final candidate

## Command

Code-only sync, then b101 subset smoke without submission packaging:

```bash
scripts/sync_code_b101.sh
VIDEOS='r13u5z4y 8jsm23a7' \
MAKE_SUBMISSION=0 \
GPU=${GPU:-4} \
RAR_EXTRA_ARGS='--rar-mode rcms' \
scripts/run_b101_rar.sh
```

Remote output locations:

```text
/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_rar
/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/rar_latest.json
/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/rar_by_video/{r13u5z4y,8jsm23a7}.json
```

Local ignored copies for inspection only:

```text
artifacts/m5r_rar/smoke_rcms/rar_latest_rcms_2video.json
artifacts/m5r_rar/smoke_rcms/r13u5z4y.json
artifacts/m5r_rar/smoke_rcms/8jsm23a7.json
```

These local copies are under ignored `artifacts/` and are not versioned.

## Runtime / shape evidence

The run completed successfully on two videos, producing 94 PNGs:

```json
{
  "videos": 2,
  "state_counts": {
    "ambiguous": 4,
    "recovery": 15,
    "stable": 75
  },
  "commit_counts": {
    "promote_rcms_anchors": 15,
    "provisional_only": 4,
    "write_main_memory": 75
  },
  "rcms_promotions": 4
}
```

Per-video summary printed by the remote run:

| video | frames | states | commits | RCMS promoted anchors | seconds |
|---|---:|---|---|---:|---:|
| `r13u5z4y` | 45 | ambiguous 4, recovery 14, stable 27 | promote 14, provisional 4, write 27 | 4 | 10.05 |
| `8jsm23a7` | 49 | recovery 1, stable 48 | promote 1, write 48 | 0 | 9.23 |

## Interpretation

This validates only that Phase A plumbing works:

- SAM2 can run through the new RAR entrypoint.
- The audit records stable/ambiguous/recovery states.
- RCMS-lite can promote pre-disappearance conditioned anchors in `r13u5z4y`.
- Subset prediction shape is correct: 2 videos, 94 PNGs.

It does **not** prove visual improvement. The next required step is to generate summary and recovery zoom sheets and compare against baseline/M11; if the figures show only conservative emptying or no true re-acquisition, RCMS-lite alone must be treated as insufficient and Ablation C retrieval anchors become mandatory.

## Submission invariant status

Not checked in this smoke because `MAKE_SUBMISSION=0`. A full-run candidate must still pass:

```bash
python tools/validate_mose_submission.py \
  --workspace /data1/yuzhixiang/cv_mosev2/MOSEv2 \
  --pred-root /data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_rar \
  --submit-root /data1/yuzhixiang/cv_mosev2/MOSEv2/homework/submission_433_rar \
  --zip-path /data1/yuzhixiang/cv_mosev2/MOSEv2/homework/submission_mosev2_rar.zip
```


## 15-video update after review fixes

After code-review fixes, both `--rar-mode rcms` and default `--rar-mode state` were run on all 15 target videos with `MAKE_SUBMISSION=0`.

| mode | videos | RGB frames | object-frames | actual memory writes | blocked non-cond writes | RCMS promotions | seconds | peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rcms` | 15 | 1004 | 1555 | 1555 | 0 | 157 | 246.05 | 1240.67 |
| `state` | 15 | 1004 | 1555 | 642 | 913 | 172 | 262.86 | 1158.95 |

The audit semantics are now explicit:

- `policy_decision`: intended state/commit policy;
- `commit_decision`: actual memory effect (`write_main_memory`, `noncond_memory_blocked`, or `conditioned_memory_existing`);
- `actual_memory_write`, `blocked_noncond_write`, and `memory_storage_key`: direct memory-write evidence.

Full review and visual analysis are in `docs/m5r_rar_review_and_comparison.md`.

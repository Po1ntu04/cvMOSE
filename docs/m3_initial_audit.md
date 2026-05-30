# M3 Initial Audit
## Branch / base
- Current branch: `method/m3-state-reanchor`
- Current SHA: `7763bac673d42a707d0197b4e3d4af3bbb46eab7`
- Base branch: `method/m2-reliable-memory`
- Base merge point: `7763bac673d42a707d0197b4e3d4af3bbb46eab7`
- V2 plan source: `/home/yu/projects/cv/from fdu/MOSEv2/preplan.md`
- V2 plan sha256: `53b28201ed89800bc9564a808e2a601ad9ee08c886f3a67b3af9dda64ddf3983`

## Artifact validation
- `baseline`: `ok=True`
  - pred_root: path `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_b101`, videos=15, pngs=1004, provided_changed=n/a, pred_errors=n/a, testzip=n/a
  - zip: path `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_sam2.zip`, videos=433, pngs=66526, provided_changed=0, pred_errors=0, testzip=None
- `m2`: `ok=True`
  - pred_root: path `/home/yu/projects/cv/cvMOSE/artifacts/m2_memory_gate/pred`, videos=15, pngs=1004, provided_changed=n/a, pred_errors=n/a, testzip=n/a
  - zip: path `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m2_memory_gate.zip`, videos=433, pngs=66526, provided_changed=0, pred_errors=0, testzip=None
- `m11`: `ok=True`
  - zip: path `/home/yu/projects/cv/cvMOSE/artifacts/m11_cycle/submission_mosev2_m11_cycle.zip`, videos=433, pngs=66526, provided_changed=0, pred_errors=0, testzip=None

## M2 current summary
```json
{
  "videos": 15,
  "noncond_total": 1535,
  "memory_written": 825,
  "memory_skipped": 710,
  "skip_ratio": 0.46254071661237783,
  "skip_reason_counts": {
    "area_abs_ok": 596,
    "area_ratio_ok": 186,
    "motion_ok": 72,
    "obj_ok": 505
  }
}
```

## M11 cycle summary
- Method: `M1.1_training_free_tracklet_identity_gate`
- Frames: `1004`
- Suppressed total: `60`
- Seconds: `289.475`
- M11 source files are not present on this branch; V2 plan requires minimal import from `method/m1-1-tracklet-gate` or fallback to artifact/audit source.

## 418 provided-output invariant
- Baseline, M2-current, and M11 cycle zips all validate with `provided_changed_count=0` against local `homework/output`.
- This validator is now a hard gate for all later candidate zips: `tools/validate_mose_submission.py`.

## Repro commands used locally
```bash
conda run -n cv-hw2 python tools/validate_mose_submission.py --workspace '/home/yu/projects/cv/from fdu/MOSEv2' --pred-root '/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_b101' --zip-path '/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_sam2.zip' --output-json artifacts/m3_initial/baseline_validation.json
conda run -n cv-hw2 python tools/validate_mose_submission.py --workspace '/home/yu/projects/cv/from fdu/MOSEv2' --pred-root artifacts/m2_memory_gate/pred --zip-path '/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m2_memory_gate.zip' --output-json artifacts/m3_initial/m2_validation.json
conda run -n cv-hw2 python tools/validate_mose_submission.py --workspace '/home/yu/projects/cv/from fdu/MOSEv2' --zip-path artifacts/m11_cycle/submission_mosev2_m11_cycle.zip --output-json artifacts/m3_initial/m11_validation.json
```

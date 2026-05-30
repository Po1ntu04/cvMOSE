# M2-light Standalone Ablation

Date: 2026-05-30  
Branch: `method/m3-state-reanchor`

## Purpose

M2-light tests whether the original M2 failure mode is caused by an over-strict `previous` reference and by updating the reference even on unreliable frames. It keeps the output masks unchanged from the SAM2 propagation path and only changes memory-write decisions plus audit state labels.

This is intentionally **not** the final method: it remains an identity-blind memory hygiene ablation and is used as a candidate/audit source for M3-state.

## Configuration

```text
--m2-reference last_reliable_on_empty
--m2-no-update-prev-on-unreliable
--m2-obj-thr 0.25
--m2-stability-thr 0.50
--m2-min-area-pixels 1
--m2-min-ratio 0.08
--m2-max-ratio 8.0
--m2-max-motion-px-floor 60
--m2-motion-area-scale 5.0
```

M2-light audit state labels:

- `memory_write`: candidate is reliable enough to write into non-conditioning memory.
- `output_only_uncertain`: output can still be emitted but must not become trusted identity/memory evidence.
- `likely_absent`: evidence supports absence/occlusion rather than reliable visible target.

## Remote validation evidence

Smoke on `r13u5z4y`:

```text
pred_check videos=1 pngs=45
memory_skipped=39, memory_written=5, noncond_total=44, skip_ratio=0.88636
state_counts={"likely_absent":37,"memory_write":6,"output_only_uncertain":2}
CUDA peak allocated/reserved ~= 939.7 MiB / 1144 MiB
```

Full 15-video run on b101:

```text
pred_root=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m2_light
zip=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/submission_mosev2_m2_light.zip
zip entries=66526, zipfile.testzip=None, size=71999072
runtime=191.92s, frames=1004
CUDA peak allocated/reserved ~= 970.9 MiB / 1190.0 MiB
```

Aggregate audit:

```json
{
  "videos": 15,
  "noncond_total": 1535,
  "memory_written": 740,
  "memory_skipped": 795,
  "skip_ratio": 0.5179153094462541,
  "skip_reason_counts": {
    "area_abs_ok": 611,
    "area_ratio_ok": 737,
    "motion_ok": 130,
    "obj_ok": 406
  },
  "state_counts": {
    "likely_absent": 611,
    "memory_write": 760,
    "output_only_uncertain": 184
  }
}
```

Representative per-video effects:

| video | memory_written | memory_skipped | skip_ratio | state notes |
|---|---:|---:|---:|---|
| `r13u5z4y` | 5 | 39 | 0.886 | 37 likely-absent frames: useful occlusion evidence, but too sparse to reacquire identity. |
| `q0sizv6m` | 69 | 13 | 0.159 | Mostly writes memory; identity switch risk remains because same-class wrong masks can be stable. |
| `msinig6m` | 118 | 281 | 0.704 | Heavy skipping in crowded multi-koala/person scene; supports high uncertainty but may suppress visible true targets. |
| `4f98052b` | 102 | 254 | 0.713 | Repeated red/black distractor scene receives many uncertain/absent states. |
| `3epdtmyr` | 183 | 6 | 0.032 | Almost no skip: stable foreground animal/distractor cases can still pass. |
| `lcgc29va` | 23 | 11 | 0.324 | Tiny pedestrian remains fragile. |
| `4vznweiu` | 12 | 22 | 0.647 | Small bead/letter object is strongly affected by area/objectness gates. |
| `8jsm23a7` | 46 | 2 | 0.042 | Mostly writes memory despite small same-class tiles/cards. |

## Interpretation

M2-light is better as a **state/evidence provider** than as a final output method. It exposes occlusion/absence hypotheses (`likely_absent`) and safer memory-write decisions, but it still has no explicit identity comparison against distractors. Therefore:

- do not keep tuning M2 thresholds as the main line;
- feed `likely_absent` and `memory_write` into M3-state candidate selection;
- treat stable non-empty M2/M2-light output as candidate evidence, not as identity proof.

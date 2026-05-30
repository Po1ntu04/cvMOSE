# M6 Phase 6 — Conservative per-video fusion report

## Fusion principle
Fusion is not pixel voting.  It is a conservative per-video policy:

1. default to the current safest M11 cycle-gate output;
2. replace only when visual evidence suggests a candidate improves identity without introducing same-class drift;
3. keep q0sizv6m/msinig6m/r13u5z4y protected from large wrong same-class blobs;
4. preserve all 418 provided outputs unchanged;
5. validate every zip with `tools/validate_mose_submission.py`.

Code:
- `tools/apply_m6_safe_fusion.py`
- visual sheets: `docs/assets/m6_fusion/summary/`
- compact committed validation/audit summary: `docs/assets/m6_fusion/summary/audit_summary.json`
- audit JSONs: `/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m6_fusion/`

## Candidate zips

All three passed validation: 433 video dirs, 66526 PNGs, 418 provided outputs unchanged.

| candidate | zip path | size | policy | recommendation |
| --- | --- | ---: | --- | --- |
| safest | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m6_safe.zip` | 71,340,697 | pure M11 default | submit first if external checkpoints are disallowed or if risk tolerance is low |
| balanced | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m6_balanced.zip` | 71,347,410 | M11 + DINO on `1qlssuz2`, `2smf7uq9` | best training-free/rule-conservative probe; submit second |
| aggressive | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m6_aggressive.zip` | 71,492,179 | M11 + DINO selected + official-large on non-rejected videos | score probe only; rule-dependent due MOSEv2-finetuned official checkpoint source |

Validation artifacts:
- `artifacts/m6_phase6/m6_safe_validation.json`
- `artifacts/m6_phase6/m6_balanced_validation.json`
- `artifacts/m6_phase6/m6_aggressive_validation.json`
- `artifacts/m6_phase6/compare_fusion.json`
- `artifacts/m6_phase6/compare_fusion.csv`

## Per-video source table

| video | safe | balanced | aggressive | reason / risk |
| --- | --- | --- | --- | --- |
| `1qlssuz2` | M11 | M5R-DINO | M5R-DINO | DINO visual is plausible and close to baseline; modest balanced risk. |
| `2smf7uq9` | M11 | M5R-DINO | M5R-DINO | DINO best case: plausible flamingo recovery when baseline/M11 empty. |
| `3epdtmyr` | M11 | M11 | official-large | no DINO smoke; official-large only as rule-dependent probe. |
| `4f98052b` | M11 | M11 | official-large | no DINO smoke; official-large only as rule-dependent probe. |
| `4vznweiu` | M11 | M11 | official-large | DINO rejected; official-large visually keeps small bead-like target more often but may over-mask. |
| `8jsm23a7` | M11 | M11 | M11 | stable guard; DINO/DAM/SAM2Long changes are unnecessary. |
| `amfdu83t` | M11 | M11 | official-large | no local visual proof; aggressive only. |
| `c8lutf29` | M11 | M11 | official-large | no local visual proof; aggressive only. |
| `jadgtmfl` | M11 | M11 | M11 | stable guard; no replacement. |
| `lcgc29va` | M11 | M11 | official-large | official-large has partial tiny recall; still uncertain, aggressive only. |
| `msinig6m` | M11 | M11 | M11 | all external routes risk composites or no gain. |
| `pe0d85lk` | M11 | M11 | official-large | no DINO smoke; official-large only as rule-dependent probe. |
| `q0sizv6m` | M11 | M11 | M11 | protect from wrong same-class animal blobs. |
| `r13u5z4y` | M11 | M11 | M11 | only safe empty is trusted; no method recovered rear strawberry. |
| `z6dx46qr` | M11 | M11 | official-large | no DINO smoke; official-large only as rule-dependent probe. |

## Qualitative verdict

- M6 did **not** find a clearly superior safe replacement for M11 on the hardest reappearance cases.
- The only training-free candidate worth a balanced probe is DINO-reanchor on `2smf7uq9` (and weakly `1qlssuz2`).
- Official MOSEv2 checkpoint/submission remains the highest likely ceiling, but it is rule-dependent and visually not safe on every key distractor case.

## Recommended submission order

1. `submission_mosev2_m6_safe.zip` if public external checkpoints are disallowed or if only one conservative final is allowed.
2. `submission_mosev2_m6_balanced.zip` as the best training-free probe.
3. `submission_mosev2_official_submission_large_15only.zip` or `submission_mosev2_m6_aggressive.zip` only if official MOSEv2-finetuned public checkpoints/submissions are allowed and you want a score ceiling probe.

## Still-unsolved cases

- `r13u5z4y`: true rear strawberry slice is never reacquired after full hand occlusion.
- `q0sizv6m`: similar animals make wrong non-empty masks dangerous.
- `msinig6m`: person/koala composites remain the dominant failure of high-recall external trackers.
- `lcgc29va`: tiny target still lacks a reliable candidate source.

## Next action

The next real improvement should target candidate recall, not thresholds: global later-frame object proposals + DINO/SAM feature matching + hard-negative margin + delayed SAM2 prompt injection.

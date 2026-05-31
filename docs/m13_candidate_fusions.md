# M13 candidate fusion artifacts

## Generated candidates

| candidate | zip | changed videos | validator | intent |
| --- | --- | ---: | --- | --- |
| `m13_key3_safe_fusion` | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_key3_safe_fusion.zip` | 2 | pass, provided unchanged | Minimal event-endpoint probe: `8jsm23a7:1@48`, `q0sizv6m:2@41`. |
| `m13_key3_balanced_zofficial` | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_key3_balanced_zofficial.zip` | 3 | pass, provided unchanged | Safe fusion plus `z6dx46qr:1` official-large interval frames 8-24. |

## Local proxy interpretation

- `safe_fusion` is intentionally tiny: it tests whether the MLLM event endpoint is useful without disturbing long stable regions.
- `balanced_zofficial` is the first low-contrast interval probe. It uses official-large for `z6dx46qr` because direct MLLM box-to-SAM created large blobs even after clipping.
- Neither candidate is expected to be a 44-level breakthrough alone; they are high-information probes for hidden-log feedback.

## Important implementation lessons

1. Direct SAM2 box prompting can oversegment low-contrast objects (`z6dx46qr`) and must be routed away from safe fusion.
2. MLLM event reasoning is most useful for selecting *when/where to probe*, not for trusting its box geometry blindly.
3. Object/frame-range fusion is now policy-controlled by `tools/apply_m13_object_fusion.py`, preventing accidental whole-root replacement.

## Round 2B reverse-anchor candidates

| candidate | zip | changed videos | validator | intent |
| --- | --- | ---: | --- | --- |
| `m13_8rev_safe` | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_8rev_safe.zip` | 1 | pass, provided unchanged | Replace only `8jsm23a7:1` frames 36-48 with the reverse-propagated Qwen event anchor. |
| `m13_8rev_zofficial_balanced` | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_8rev_zofficial_balanced.zip` | 2 | pass, provided unchanged | `m13_8rev_safe` plus the previously useful official-large interval for `z6dx46qr:1` frames 8-24. |

Additional artifacts:

- Fusion policies: `artifacts/m13_fusion_revwin/policy_8rev_safe.json`, `artifacts/m13_fusion_revwin/policy_8rev_zofficial_balanced.json`
- Validation logs: `artifacts/m13_fusion_revwin/validate_8rev_safe.txt`, `artifacts/m13_fusion_revwin/validate_8rev_zofficial_balanced.txt`
- Visual sheets: `docs/assets/m13_revwin_fusions/`

Current recommendation: if testing one new M13 zip, submit `submission_mosev2_m13_8rev_zofficial_balanced.zip` first because it combines the clearest semantic re-anchor improvement (`8jsm23a7`) with the already isolated low-contrast official-large interval (`z6dx46qr`). If budget is risk-averse, submit `submission_mosev2_m13_8rev_safe.zip` to isolate the MLLM reverse-anchor effect.

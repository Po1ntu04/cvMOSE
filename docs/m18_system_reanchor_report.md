# M18 system re-anchor report

This report is the Round-1 durable memory layer for the object-centric temporal re-anchor system. It does not modify prediction outputs; it creates reviewable JSON ledgers that later candidate retrieval and fusion tools can consume.

## Gate A — ledger/schema review

- schema version: `m18.temporal-ledger.v1`
- ledger dir: `artifacts/m18_temporal_ledger`
- sample schema JSON: `artifacts/m18_temporal_ledger/schema_sample.json`
- two required sample ledgers: `q0sizv6m_obj2.json`, `8jsm23a7_obj1.json`

Pass condition coverage:

- target profile and first-frame stats are explicit, including tiny/edge/same-class flags.
- event story records visible/occluded/recovery/absent windows with allowed output modes.
- positive and distractor memory are separated; rejected policies remain available as negative memory.
- candidate anchors store positive evidence, negative evidence, risk tags, review status, and promotion decision.

## Round-1 object ledger index

| object | current best | review | state | score memory | next review window |
| --- | --- | --- | --- | --- | --- |
| `1qlssuz2:obj1` | `m15_safe_current_track` | needs_more_evidence | AMBIGUOUS | m15_safe: row 36.88 / hidden 43.44<br>m15_aggressive / hidden 43.38 | [13, 14, 37, 39]: detector-level small vehicle proof; no broad SAM2 window replacement |
| `4vznweiu:obj1` | `m15_safe_current_track` | needs_more_evidence | AMBIGUOUS | m15_safe: row 33.50 / hidden 43.44<br>m15_aggressive / hidden 43.38 | [0, 1, 10, 20, 34]: build high-resolution bead/letter atlas before any replacement |
| `8jsm23a7:obj1` | `m15_safe_unchanged_for_8js` | needs_more_evidence | AMBIGUOUS | m15_safe: row 2.13 / hidden 43.44<br>m15_balanced: row 2.13 / hidden 43.44 | [0, 2, 20, 26, 48]: build M18 Mahjong entity atlas with target hypothesis plus >=2 hard distractors |
| `amfdu83t:obj1` | `m15_safe_m14_from8_noclip` | approved | RECONFIRMED | m13_zofficial_family: row 28.44 / hidden 43.34<br>m15_safe: row 84.84 / hidden 43.44 | [12, 23]: monitor continuation; if regression appears, trim but keep positive memory |
| `q0sizv6m:obj2` | `m17_q0_full_box` | approved | RECOVERY_BRANCH | m17_q0_early_box / hidden 43.47<br>m17_q0_full_box: row 68.29 / hidden 43.54 | [27, 34]: choose trim/rump/empty schedule using hidden feedback and zoom sheet |
| `r13u5z4y:obj1` | `m15_safe_empty_after_occlusion` | needs_more_evidence | OCCLUDED | m15_safe: row 38.84 / hidden 43.44 | [0, 7, 19, 20, 44]: strawberry atlas; only accept anchor if target slice can be distinguished from hard negatives |
| `z6dx46qr:obj1` | `m13_zofficial_balanced_official_large_interval` | approved | RECONFIRMED | m7_family: row 40.47 / hidden 43.30<br>m13_zofficial_balanced: row 65.14 / hidden 43.34 | [25, 37]: look for source-independent low-contrast continuation; do not direct-box prompt |

## Gate B/C/D hooks now represented in JSON

- Gate B atlas results should append `distractor_memory` plus `candidate_anchors[*].negative_evidence` before any propagation.
- Gate C promotion cards map directly to `candidate_anchors[*]`: source, frame, positive/negative evidence, story compatibility, descriptor margin, risk tags, and decision.
- Gate D fusion must write a source table and validation JSON; `tools/apply_m18_reanchor_fusion.py` consumes these ledgers plus explicit policies.

## Current conservative interpretation

- Approved positive memory: `amfdu83t:1` M14/M15 from8 no-clip; `z6dx46qr:1` official-large interval; `q0sizv6m:2` M17 full-box bottom path with late schedule still under review.
- Needs atlas before safe/balanced: `8jsm23a7:1`, `r13u5z4y:1`, `1qlssuz2:1`, `4vznweiu:1`.
- Rejected/negative patterns: old-position Mahjong distractor, q0 mid-left herd animal, r13 more-salient strawberry slice, 1ql broad expansion, 4v bead/flower semantic swap, z6 direct-box blob.

## No-output-change invariant

Round 1 generated only docs and JSON ledgers. No prediction root or submission zip was modified.

## Review/optimization pass — identity-bank and fusion safety fixes

Review finding: the first Round-1 retrieval scaffold was structurally correct but underused the new temporal ledger. `q0_fullbox_hidden_good`, `amf_from8_hidden_good`, `z6_official_interval`, and known late/distractor ledger memories were recorded in JSON but were not entering the descriptor identity bank unless they also appeared as approved candidate anchors with an exact source-root name. This meant the first retrieval audit still behaved too much like an old first-frame-only verifier.

Fixes applied:

- `tools/build_m18_candidate_retrieval.py` now resolves fuzzy ledger source names to available prediction roots and samples ledger `positive_memory` / `distractor_memory` frame ranges into the identity bank.
- `src/cvmose/retrieval_memory.py` now treats non-empty candidates in `OCCLUDED` state as `needs_more_evidence` rather than `output_only`, so occluded same-class proposals cannot silently become balanced fusion edits.
- `tools/apply_m18_reanchor_fusion.py` now blocks non-empty same-class-dense edits in safe/balanced unless the edit is approved or explicitly `balanced_allowed` after review.

Fresh retrieval audit after the fix:

- q0 identity bank increased from first-frame-only to 4 positives / 6 distractors; top q0 cards now mainly audit the current full-box branch instead of pretending new anchors are automatically promotable.
- r13 occluded strawberry candidates are now `needs_more_evidence` review cards, not output-only cards, preserving the safe-empty policy until a strawberry atlas distinguishes the true slice.
- 8js remains output-only/current-reference material; it still needs a Mahjong entity atlas before any promotion.

Verification:

```bash
conda run -n cv-hw2 python -m py_compile src/cvmose/temporal_ledger.py src/cvmose/retrieval_memory.py tools/build_m18_temporal_ledger.py tools/build_m18_candidate_retrieval.py tools/apply_m18_reanchor_fusion.py tools/mllm_sameclass_atlas.py tools/test_m18_retrieval_memory.py
conda run -n cv-hw2 python tools/test_m18_retrieval_memory.py
conda run -n cv-hw2 python tools/build_m18_candidate_retrieval.py --ledger-dir artifacts/m18_temporal_ledger --targets q0sizv6m:2 8jsm23a7:1 r13u5z4y:1 --top-k 8 --max-frames-per-object 35 --out-json artifacts/m18_candidate_retrieval/candidates.json --out-csv artifacts/m18_candidate_retrieval/candidates.csv
```

## Gate B result interpretation

MLLM routing used fast `qwen-vl-plus` for per-frame visual captions and `qwen3.6-plus` for text-only aggregation. This avoided the repeated qwen3.6 vision timeout observed in the first Gate-B smoke while keeping qwen3.6 for event-chain reasoning.

| object | MLLM action | conf | ledger additions | retrieval effect | verdict |
| --- | --- | ---: | --- | --- | --- |
| `8jsm23a7:obj1` | manual_review | 0.65 | +8 candidates / +28 distractors | official_large:object_id output_only margin=0.12145; bank +1/-29 | Atlas useful as hard-negative memory; no safe promotion yet because official/current candidates remain below same-class margin. |
| `r13u5z4y:obj1` | manual_review | 0.00 | +0 candidates / +46 distractors | m16mask:anyfg:1 needs_more_evidence margin=0.19784; bank +1/-50 | Confirms no reliable strawberry reanchor; adds many hard negatives and preserves empty-safe policy. |
| `4vznweiu:obj1` | manual_review | 0.65 | +7 candidates / +33 distractors | official_large:object_id output_only margin=0.19241; bank +1/-34 | Finds letter/cube hypotheses but retrieval rejects/keeps output-only due strong hard-negative similarity; needs higher-res detector proof. |
| `1qlssuz2:obj1` | manual_review | 0.75 | +16 candidates / +27 distractors | ledger_anchor:m18_atlas_target_f39_E1 reject margin=-0.02425; bank +1/-28 | MLLM proposed car anchors, but retrieval rejects late atlas boxes as closer to negatives; no broad replacement. |

### Gate B safety conclusions

- `8jsm23a7`: do **not** promote the MLLM leftmost-tile boxes yet. The atlas is valuable mainly because it names old-position/static tile hard negatives and flags frame-36 spatial jump risk.
- `r13u5z4y`: MLLM explicitly failed to identify a positive strawberry reanchor; this is a useful negative result. Keep M11/M15 empty-safe unless a stronger detector provides source-independent evidence.
- `4vznweiu`: Qwen sees a small T/letter/cube track, but retrieval says many boxes are close to hard negatives. Treat as review material, not final.
- `1qlssuz2`: atlas boxes are rejected by the current positive-vs-negative descriptor gate; do not revive previous aggressive broad replacement.


### Gate B maintenance note

If `tools/build_m18_temporal_ledger.py` is rerun, immediately rerun `tools/apply_m18_atlas_to_ledger.py` before candidate retrieval; the atlas integration is intentionally stored as review-time ledger evidence and is not generated by the base ledger builder.

# M5R/RAR Code Review and Real Visual Comparison

Date: 2026-05-30
Status: reviewed, smoke-tested on all 15 target videos, **not promoted to final**

## Conclusion first

RAR Phase A/B is useful instrumentation, but it does **not** yet solve the task bottleneck: it can delay or block questionable SAM2 memory writes, but it still lacks an independent later-frame identity evidence path, so it cannot reliably re-acquire the target after occlusion or same-class confusion.

## Review result

Two independent review lanes were used before the final comparison.

### Initial review findings

| severity | finding | fix applied |
|---|---|---|
| HIGH | `RAR_EXTRA_ARGS` could append protected CLI flags after launcher path checks and override safe `--workspace`, `--submit-root`, `--zip-path`, etc. | `scripts/run_b101_rar.sh` now allowlists only RAR tuning flags and rejects protected flags before any cleanup/inference. |
| MEDIUM | `commit_decision` in `--rar-mode rcms` looked like memory control even though current non-conditioning memory was still written. | `FrameAudit` now separates `policy_decision`, `commit_decision`, `actual_memory_write`, `blocked_noncond_write`, and `memory_storage_key`. |
| MEDIUM | Provisional ambiguous masks could update the geometry reference and later make a wrong branch look stable. | `tools/infer_mosev2_sam2_rar.py` now keeps `reference_stats` tied to committed/conditioning outputs; provisional blocked frames do not become the next reference. |
| MEDIUM | Direct `--make-submission` path did not force the full 418-provided-output and 66,526-PNG invariant check. | Direct submission now guards path prefixes and calls `tools/validate_mose_submission.py` after packaging. |

### Post-fix validation

```bash
python3 -m py_compile src/cvmose/reanchor.py tools/infer_mosev2_sam2_rar.py
bash -n scripts/run_b101_rar.sh
git diff --check
VIDEOS='r13u5z4y' MAKE_SUBMISSION=0 RAR_EXTRA_ARGS='--zip-path /tmp/unsafe.zip' scripts/run_b101_rar.sh
# expected: Refusing unsafe RAR_EXTRA_ARGS flag: --zip-path
```

Remote path-guard import smoke in the b101 SAM2 environment also passed with `remote_path_guard_ok`.

### Remaining architectural watch

The implementation is still **Ablation A/B**, not full RAR. It has no retrieval-anchor mining, no DINO/SAM3/object-proposal identity matching, and no true delayed promotion of a newly recovered external candidate. Therefore it is not a final recovery system.

## Full 15-video smoke evidence

Commands run on b101 after code-only sync:

```bash
ALL_VIDEOS='1qlssuz2 q0sizv6m lcgc29va z6dx46qr 4vznweiu amfdu83t 4f98052b 3epdtmyr msinig6m c8lutf29 2smf7uq9 8jsm23a7 jadgtmfl r13u5z4y pe0d85lk'

VIDEOS="$ALL_VIDEOS" MAKE_SUBMISSION=0 GPU=${GPU:-4} \
  RAR_EXTRA_ARGS='--rar-mode rcms' \
  scripts/run_b101_rar.sh

VIDEOS="$ALL_VIDEOS" MAKE_SUBMISSION=0 GPU=${GPU:-4} \
  PRED_ROOT='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_rar_state' \
  AUDIT_JSON='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/rar_state_latest.json' \
  AUDIT_DIR='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/rar_state_by_video' \
  RAR_EXTRA_ARGS='--rar-mode state' \
  scripts/run_b101_rar.sh
```

Remote prediction roots:

```text
/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_rar
/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_rar_state
```

Local ignored audit copies:

```text
artifacts/m5r_rar/full15_rcms/rar_latest.json
artifacts/m5r_rar/full15_state/rar_state_latest.json
artifacts/m5r_rar/key_compare/metrics.json
```

### Aggregate audit

| mode | videos | RGB frames | object-frames | stable | ambiguous | recovery | actual memory writes | blocked non-cond writes | RCMS promotions | seconds | peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rcms` | 15 | 1004 | 1555 | 689 | 85 | 781 | 1555 | 0 | 157 | 246.05 | 1240.67 |
| `state` | 15 | 1004 | 1555 | 642 | 139 | 774 | 642 | 913 | 172 | 262.86 | 1158.95 |

Interpretation:

- `rcms` is mostly a reservoir/conditioned-anchor ablation; it **does not** block current non-conditioning memory.
- `state` blocks 913/1555 object-frame non-conditioning writes, so the delayed-memory path is active.
- The large number of recovery states confirms the detector is sensitive, but sensitivity alone does not imply correct recovery.

### Per-video state summary

| video | frames | objects | rcms states | rcms promotions | state blocked / object-frames | state states |
|---|---:|---:|---|---:|---:|---|
| `1qlssuz2` | 40 | 1 | recovery 5, stable 35 | 8 | 5/40 | recovery 5, stable 35 |
| `q0sizv6m` | 42 | 2 | ambiguous 3, recovery 24, stable 57 | 14 | 21/84 | ambiguous 7, recovery 14, stable 63 |
| `lcgc29va` | 35 | 1 | ambiguous 5, recovery 9, stable 21 | 12 | 10/35 | ambiguous 3, recovery 7, stable 25 |
| `z6dx46qr` | 38 | 1 | ambiguous 19, recovery 12, stable 7 | 8 | 26/38 | ambiguous 13, recovery 13, stable 12 |
| `4vznweiu` | 35 | 1 | ambiguous 3, recovery 2, stable 30 | 4 | 6/35 | ambiguous 1, recovery 5, stable 29 |
| `amfdu83t` | 24 | 1 | recovery 22, stable 2 | 1 | 18/24 | recovery 18, stable 6 |
| `4f98052b` | 179 | 2 | ambiguous 32, recovery 229, stable 97 | 46 | 303/358 | ambiguous 43, recovery 260, stable 55 |
| `3epdtmyr` | 190 | 1 | recovery 58, stable 132 | 4 | 38/190 | ambiguous 2, recovery 36, stable 152 |
| `msinig6m` | 134 | 3 | ambiguous 4, recovery 296, stable 102 | 20 | 317/402 | ambiguous 46, recovery 271, stable 85 |
| `c8lutf29` | 47 | 1 | ambiguous 2, recovery 34, stable 11 | 4 | 30/47 | ambiguous 3, recovery 27, stable 17 |
| `2smf7uq9` | 43 | 1 | ambiguous 3, recovery 11, stable 29 | 4 | 13/43 | ambiguous 2, recovery 11, stable 30 |
| `8jsm23a7` | 49 | 1 | recovery 1, stable 48 | 0 | 2/49 | recovery 2, stable 47 |
| `jadgtmfl` | 41 | 1 | ambiguous 3, recovery 5, stable 33 | 4 | 8/41 | ambiguous 2, recovery 6, stable 33 |
| `r13u5z4y` | 45 | 1 | ambiguous 4, recovery 14, stable 27 | 4 | 18/45 | ambiguous 4, recovery 14, stable 27 |
| `pe0d85lk` | 62 | 2 | ambiguous 7, recovery 59, stable 58 | 24 | 98/124 | ambiguous 13, recovery 85, stable 26 |

## Visual evidence index

Key comparison sheets compare RGB, SAM2, M11, RAR-rcms, and RAR-state.

```text
docs/assets/m5r_rar/key_compare/full/{video}_rar_key_compare.jpg
docs/assets/m5r_rar/key_compare/zoom/{video}_rar_key_compare_zoom.jpg
```

Versioned key videos:

- `r13u5z4y`
- `q0sizv6m`
- `msinig6m`
- `1qlssuz2`
- `2smf7uq9`
- `8jsm23a7`
- `lcgc29va`
- `4vznweiu`

Legacy two-video smoke sheets remain in:

```text
docs/assets/m5r_rar/smoke_compare/full/
docs/assets/m5r_rar/smoke_compare/zoom/
```

## Real visual comparison and diagnosis

### `r13u5z4y` — sliced strawberry, heavy occlusion, same-class distractors

Visual sheets:

- `docs/assets/m5r_rar/key_compare/zoom/r13u5z4y_rar_key_compare_zoom.jpg`
- `docs/assets/m5r_rar/key_compare/full/r13u5z4y_rar_key_compare.jpg`

Observed chain:

- Before occlusion, all methods track the visible strawberry piece or its visible edge.
- Around frames 5--18 the target is hand-occluded; RAR correctly enters ambiguous/recovery and promotes pre-disappearance anchors.
- At frames 20--22, the visible evidence is now a cluster of highly similar strawberry slices. SAM2 locks onto a plausible but wrong/uncertain slice region; RAR-rcms and RAR-state largely follow the same wrong path.
- M11 is safer because it empties many post-gap frames, but it also does not recover the true target.

Diagnosis: RCMS-lite can recall old memory, but old memory is not enough when the pre-disappearance observation was already a tiny partial edge and the reappearance set contains near-identical slices. This is the clearest proof that Ablation C needs later-frame candidate retrieval + identity verification.

### `q0sizv6m` — multiple similar pigs, object identity ambiguity

Visual sheet:

- `docs/assets/m5r_rar/key_compare/zoom/q0sizv6m_rar_key_compare_zoom.jpg`

Metrics:

| method | changed vs SAM2 | mean binary IoU vs SAM2 | empty frames |
|---|---:|---:|---:|
| M11 | 0/42 | 1.0000 | 0 |
| RAR-rcms | 41/42 | 0.8333 | 0 |
| RAR-state | 41/42 | 0.2412 | 0 |

Observed chain:

- The scene contains several near-identical pigs; the target instance must be preserved, not merely any pig-like region.
- RAR-state often emits large masks or shifted composite regions while marking some object tracks as recovery/ambiguous.
- The state machine blocks memory writes, but the visible output can still be a broad/wrong same-class mask.

Diagnosis: geometry/stability gating cannot prove identity in a herd. Without negative examples and object-level retrieval, delayed commit only changes memory bookkeeping; it does not create discriminative evidence.

### `msinig6m` — multi-object koala/person contact and heavy occlusion

Visual sheet:

- `docs/assets/m5r_rar/key_compare/zoom/msinig6m_rar_key_compare_zoom.jpg`

Metrics:

| method | changed vs SAM2 | mean binary IoU vs SAM2 | empty frames |
|---|---:|---:|---:|
| M11 | 25/134 | 0.8284 | 74 |
| RAR-rcms | 79/134 | 0.7060 | 86 |
| RAR-state | 113/134 | 0.4176 | 37 |

Observed chain:

- Early frames already involve several object masks and contact with people/background.
- RAR-state blocks 317/402 object-frame non-conditioning writes, but visual masks do not become more semantically correct; they become different and sometimes less conservative.
- The method lacks a way to say “this recovered region is the same koala/person/object as the first-frame object,” especially when multiple object ids interact.

Diagnosis: the state machine is detecting uncertainty but cannot resolve it. Multi-object cases require candidate-level matching and negative-bank suppression before writing or replacing masks.

### `1qlssuz2` — tiny moving car, scale/view change, overpass disappearance

Visual sheet:

- `docs/assets/m5r_rar/key_compare/zoom/1qlssuz2_rar_key_compare_zoom.jpg`

Observed chain:

- Early frames are almost identical across SAM2, M11, and RAR.
- When the car becomes tiny or disappears under/near an overpass, all methods output empty or weak masks.
- RCMS promotions occur, but they do not re-detect the tiny car later in the wide aerial view.

Diagnosis: this is not mainly memory pollution; it is missed re-detection at tiny scale. RAR needs a proposal source or high-resolution candidate search after state enters recovery.

### `2smf7uq9` — flamingo, target exits/occludes in similar flock

Visual sheet:

- `docs/assets/m5r_rar/key_compare/zoom/2smf7uq9_rar_key_compare_zoom.jpg`

Observed chain:

- Initial tracking is similar across all methods.
- Once the target becomes absent or tiny in the group, SAM2/M11/RAR all become empty for long stretches.
- RAR does not re-acquire a plausible flamingo instance after the empty gap.

Diagnosis: suppressing memory writes is not harmful here, but it is insufficient. Recovery requires a later candidate proposal and a check that it is the original flamingo, not any flamingo.

### `8jsm23a7` — mahjong tile, early hand occlusion then stable tile

Visual sheet:

- `docs/assets/m5r_rar/key_compare/zoom/8jsm23a7_rar_key_compare_zoom.jpg`

Observed chain:

- SAM2 and M11 are already stable after the first transient empty/occluded frame.
- RAR-state introduces more changed masks than needed and briefly empties a frame that SAM2 can segment.
- RAR-rcms is visually almost identical to SAM2, so it brings no meaningful gain.

Diagnosis: when SAM2 is already stable, the RAR-state conservative reference logic can be net negative. This supports routing: only enter recovery-heavy logic when there is strong evidence of disappearance/confusion.

### `lcgc29va` and `4vznweiu` — tiny/edge targets and local refinement failures

Visual sheets:

- `docs/assets/m5r_rar/key_compare/zoom/lcgc29va_rar_key_compare_zoom.jpg`
- `docs/assets/m5r_rar/key_compare/zoom/4vznweiu_rar_key_compare_zoom.jpg`

Observed chain:

- `lcgc29va`: the target is small and often at a cluttered station edge. RAR sometimes keeps or creates vertical/edge-like mask fragments rather than a clearly recovered target.
- `4vznweiu`: the bead/letter target is small and sometimes occluded by cloth/feather; RAR can shift to nearby beads or partial regions without proving identity.

Diagnosis: tiny-target refinement should be applied **after** identity/location is selected. Running memory governance alone cannot make the high-resolution local decision.

## Quantitative proxy comparison on key videos

These numbers compare binary masks against SAM2, not ground truth. They measure how much the method changed the baseline and help flag catastrophic divergence.

| video | RAR-rcms mean IoU vs SAM2 | RAR-state mean IoU vs SAM2 | RAR-state empty frames | visual verdict |
|---|---:|---:|---:|---|
| `r13u5z4y` | 0.4663 | 0.4671 | 14 | Not recovered; follows wrong strawberry region after reappearance. |
| `q0sizv6m` | 0.8333 | 0.2412 | 0 | State mode diverges strongly in same-class herd; unsafe. |
| `msinig6m` | 0.7060 | 0.4176 | 37 | Heavy divergence without clearer identity; unsafe. |
| `1qlssuz2` | 0.9685 | 0.9657 | 5 | Mostly baseline-like; no re-detection improvement. |
| `2smf7uq9` | 0.7788 | 0.7788 | 11 | Does not recover after empty gap. |
| `8jsm23a7` | 0.9992 | 0.4424 | 2 | RAR-state changes too much when baseline is already good. |
| `lcgc29va` | 0.7871 | 0.6631 | 7 | Tiny/edge instability remains. |
| `4vznweiu` | 0.9131 | 0.8680 | 5 | Small local shifts; no robust identity gain. |

## Decision

Do **not** package RAR Phase A/B as a final submission candidate.

Keep the code because it provides valuable audit fields and a clean place to insert later retrieval anchors. The next meaningful experiment should be Ablation C:

1. generate later-frame candidate masks/boxes in recovery windows;
2. match candidates against first-frame and pre-disappearance anchors with object-level features plus negatives;
3. only promote a recovered candidate after delayed temporal confirmation;
4. use tiny-crop refinement only after a candidate identity/location is selected.

Until that exists, M11 remains the safer final candidate because it avoids some catastrophic wrong tracklets, whereas RAR A/B can either mimic SAM2 or diverge without enough identity evidence.

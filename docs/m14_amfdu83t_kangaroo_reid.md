# M14 amfdu83t same-class kangaroo re-ID probe

## One-line conclusion

The next genuinely useful improvement is **same-class entity memory**: enumerate the target and each similar animal as separate physical hypotheses, keep strong distractors as a negative bank, then inject only tight, locally clipped re-anchor boxes into SAM2.

## User observation normalized

`amfdu83t:obj1` is not a normal short-term propagation problem.

- Frame `00000`: the target is the **far-left edge kangaroo**, only head/neck/front foot visible.
- Frame `00001`: only the far-left head remains.
- Frame `00002`: target likely leaves the view due camera motion.
- Frames after disappearance contain a very strong left-side kangaroo distractor, but it is not necessarily the original target.
- Likely reappearance: around `00008` as the **bottom-left / near-camera kangaroo**.
- Later motion hypothesis: fast near-camera motion, overtaking the strong distractor around `00010`, then jumping around `00011`.

This is a same-class physical-instance problem: category/appearance similarity is actively misleading.

## What existing sources do

Visual sheets:

- `docs/assets/m14_amfdu83t_sources_a/amfdu83t_m6_compare.jpg`
- `docs/assets/m14_amfdu83t_sources_b/amfdu83t_m6_compare.jpg`

Observed result: SAM2/M7, M11, official B+/L, M6 balanced/aggressive, SAM2Long and DAM4SAM are mostly empty after the early frames. They do not recover the frame-8/11 reappearance path.

Hidden-score memory from latest logs:

- `zofficial_balanced` full score: `J&F_new=43.34`.
- `amfdu83t` row under that zip: `J&F_new=28.44`, so there is still large local headroom.

## Implementation change: safer SAM2 teach-box clipping

File changed:

- `tools/infer_mosev2_sam2_teach_boxes.py`

Added:

- `--clip-mode union|nearest`
- default remains `union` for backward compatibility.
- `nearest` clips each merged frame to the temporally nearest anchor box instead of the union of all anchor boxes.

Why this matters:

- In crowded same-class scenes, union clipping lets a later box admit an adjacent distractor in an earlier merge frame.
- `nearest` is still simple, but it localizes each frame to the closest anchor and reduced the obvious multi-kangaroo merges seen in the first M14 probes.

Verification:

```bash
python3 -m py_compile tools/infer_mosev2_sam2_teach_boxes.py tools/mllm_sameclass_atlas.py
```

## Box-probe experiments

All probes used first-frame GT as the conditioning mask, then added non-GT box prompts to SAM2 only as a diagnostic re-anchor test.

### Early broad probes

- `nearfront`: plausible but frame `00010` merged two kangaroos.
- `jump`: frame `00010` looked better, but frame `00011` merged the intended jump candidate with an adjacent right kangaroo.

Visuals:

- `docs/assets/m14_amfdu83t_box_probes/amfdu83t_m6_compare.jpg`
- `docs/assets/m14_amfdu83t_box_probes/large_frames/`

### Tight probes with `--clip-mode nearest`

Compared roots:

- `tight_jump`
- `tight_near`
- `single_f8`
- `f8_f10_only`
- `temporal_8_11`

Visuals:

- `docs/assets/m14_amfdu83t_tight_probes_a/amfdu83t_m6_compare.jpg`
- `docs/assets/m14_amfdu83t_tight_probes_b/amfdu83t_m6_compare.jpg`
- `docs/assets/m14_amfdu83t_tight_probes_large/`
- `docs/assets/m14_amfdu83t_temporal_compare/amfdu83t_m6_compare.jpg`
- `docs/assets/m14_amfdu83t_temporal_8_11_large/`

Best diagnostic candidate: `temporal_8_11`.

- Changes only frames `8,9,10,11` when fused.
- Frame `00008`: left-edge near-camera reappearance candidate.
- Frame `00009`: large far-left near-camera kangaroo, separated from mid-left distractor.
- Frame `00010`: lower-left/foreground overtaking candidate.
- Frame `00011`: central-left jump candidate, no obvious adjacent-right merge after tight clipping.

Invariant audit:

- `compare_temporal_8_11.csv`: `changed_vs_baseline=4`, `first_frame_preserved=True`, `label_ids_valid=True`.

## Qwen3.6 split-harness result

Qwen3.6 split harness was run on frames `0,1,2,8,9,10,11` with the user-provided observation as an explicit diagnostic hint.

Output:

- JSON: `artifacts/m14_amfdu_candidates/amfdu_q36_split.json` (not committed; artifact cache)
- Doc: `docs/m14_amfdu83t_qwen36_split.md`

Qwen did capture the high-level event:

- `event_type=exits_reappears`
- `recommended_action=reanchor_at_frame`
- diagnosis: same-class / empty-after-event failure.

But Qwen's direct boxes were too broad, especially lower-frame boxes, so they were **not** used directly for final mask construction. This supports the architectural lesson: MLLM is better as event-chain / negative-bank / candidate-routing evidence than as an unconstrained coordinate generator.

A richer same-class atlas tool was added:

- `tools/mllm_sameclass_atlas.py`

It renders reference + coordinate-grid panels, asks Qwen to enumerate per-frame same-class candidates and hard negatives, then aggregates a re-anchor/negative-memory plan. A first rich-panel run timed out on qwen3.6-plus for early frames, which confirms the earlier timeout lesson: use compact visual calls and text-only aggregation for production.

## Candidate zip

A validated diagnostic zip was produced by starting from the current best `zofficial_balanced` root and replacing only `amfdu83t:obj1` frames `8-11` with the visually reviewed `temporal_8_11` candidate.

- Repo hardlink/local path: `submission_mosev2_m14_zofficial_amfdu_temporal811.zip`
- MOSE workspace path: `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m14_zofficial_amfdu_temporal811.zip`
- Pred root: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m14_zofficial_amfdu_temporal811`
- Validation: `ok=true`, 433 videos, 66526 PNGs, 418 provided outputs changed count `0`.

This should be treated as a **balanced/probe candidate**, not a safe final replacement, until hidden score confirms that the temporal interpretation is correct.

## Why this is not yet a general breakthrough

The good part:

- It targets a real local failure where all existing model roots are empty.
- It uses physical motion reasoning rather than category similarity.
- It shows why nearest-anchor clipping is necessary in same-class crowds.

The limitation:

- The best boxes are still partly human/MLLM-guided and video-specific.
- Qwen's direct coordinate boxes remain too coarse.
- There is no automatic detector/tracker that reliably proposes all kangaroo instances and maintains their separate identities.

## Next engineering direction

1. Use a real external tracker/detector candidate source for same-class entities, not just SAM2 box prompts.
   - DAM4SAM/d4sm remains the first installed route, but existing amfdu smoke was empty after early frames.
   - The next better fit is a detector/tracker that can propose every same-class instance as boxes, after which Qwen/DINO/SAM descriptors can build positive/negative banks.
2. Convert `mllm_sameclass_atlas.py` to compact split mode:
   - short per-frame candidate enumeration/caption calls;
   - text-only qwen3.6 aggregation;
   - no dense multi-frame panels.
3. Teach SAM2 with **tight positive boxes plus explicit negative/distractor memory**, then use delayed commit and nearest-anchor clipping.
4. Only fuse frame intervals whose masks are visually single-instance and do not merge adjacent same-class objects.

# M2 Visual Analysis — SAM2 Reliable Memory Gate

Date: 2026-05-30  
Branch: `method/m2-reliable-memory`  
Method under inspection: original SAM2 inference path + training-free non-conditioning memory write gate.

## 1. Evidence used

Local downloaded outputs:

- M2 predictions: `artifacts/m2_memory_gate/pred/`
- M2 submission zip: `artifacts/m2_memory_gate/submission/submission_mosev2_m2_memory_gate.zip`
- Local homework copy: `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m2_memory_gate.zip`
- Audit: `artifacts/m2_memory_gate/m2_memory_gate_latest.json`
- Per-video comparison summary: `artifacts/m2_memory_gate/visual_analysis/compare_summary.json`
- Full-frame sheets: `artifacts/m2_memory_gate/visual_analysis/*_sam2_vs_m2_sheet.jpg`
- Local zoom sheets: `artifacts/m2_memory_gate/visual_analysis/zooms/*_m2_zoom_sheet.jpg`

Submission zip verification:

- Size: `71,983,174` bytes
- Zip entries: `66,526`
- `zipfile.testzip()`: `None`
- Expected submission coverage: `433` video directories, `66,526` PNG masks.

Run/audit summary:

- Frames inferred on the 15 homework videos: `1004`
- Runtime on b101: `230.69s`, about `4.35 fps`
- CUDA peak allocated/reserved: `970.86 MiB` / `1190.0 MiB`
- Memory writes: `825`; memory skips: `710`; non-conditioning decisions: `1535`; skip ratio: `0.4625`
- Skip reasons: `area_abs_ok=596`, `obj_ok=505`, `area_ratio_ok=186`, `motion_ok=72`

Important limitation: only frame `00000` has GT. Later-frame judgments below combine mask-to-mask comparison with raw-image semantic inspection; they are not a substitute for hidden test GT.

## 2. Quantitative mask-difference summary versus SAM2 baseline

This table compares foreground masks between the SAM2 baseline and M2. Low IoU means the method changed behavior strongly, not necessarily that it improved accuracy.

| video | frames | changed | mean IoU | base empty | M2 empty | skips | skip ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q0sizv6m` | 42 | 40 | 0.206 | 0 | 0 | 22 | 0.27 |
| `lcgc29va` | 35 | 28 | 0.368 | 8 | 18 | 20 | 0.59 |
| `msinig6m` | 134 | 123 | 0.371 | 53 | 12 | 199 | 0.50 |
| `1qlssuz2` | 40 | 32 | 0.397 | 5 | 11 | 13 | 0.33 |
| `4vznweiu` | 35 | 31 | 0.449 | 3 | 21 | 23 | 0.68 |
| `r13u5z4y` | 45 | 29 | 0.489 | 16 | 15 | 17 | 0.39 |
| `8jsm23a7` | 49 | 47 | 0.704 | 1 | 4 | 13 | 0.27 |
| `4f98052b` | 179 | 133 | 0.709 | 84 | 52 | 220 | 0.62 |
| `3epdtmyr` | 190 | 181 | 0.725 | 58 | 6 | 7 | 0.04 |
| `z6dx46qr` | 38 | 16 | 0.759 | 30 | 21 | 23 | 0.62 |
| `2smf7uq9` | 43 | 31 | 0.776 | 20 | 11 | 14 | 0.33 |
| `amfdu83t` | 24 | 5 | 0.872 | 21 | 18 | 19 | 0.83 |
| `jadgtmfl` | 41 | 33 | 0.927 | 5 | 5 | 7 | 0.17 |
| `pe0d85lk` | 62 | 41 | 0.978 | 19 | 19 | 77 | 0.63 |
| `c8lutf29` | 47 | 12 | 0.997 | 34 | 34 | 36 | 0.78 |

## 3. Image-based findings

### 3.1 `r13u5z4y` — strawberry cutting, long occlusion, same-class slices

Observation from zoom sheet:

- M2 correctly skips memory around the hand-occluded/empty interval, e.g. frames around `00019` and `00020` have no output and are not written as reliable memory.
- Around reappearance (`00021`/`00022`), both SAM2 and M2 output strawberry-like regions, but they select different portions of the strawberry slices.
- Later (`00026`, `00027`, `00044`), SAM2 tends to cover a larger left/front strawberry piece while M2 often keeps a smaller region near another slice/edge.

Interpretation:

- The memory gate does address the easy part of the problem: do not write obvious hand/empty frames into memory.
- It does **not** solve the hard part: after the slice is fully occluded and placed among similar strawberry pieces, the model still lacks an identity re-identification cue to decide which slice is the original target.
- Therefore M2 is a memory-hygiene improvement but not a robust solution for this canonical failure case.

### 3.2 `q0sizv6m` — many visually similar guinea pigs

Observation from zoom sheet:

- Frame `00000` starts from a specific masked animal instance, but the scene contains many same-class animals with close appearance and frequent overlap.
- By `00005` onward, M2 masks expand to multiple green regions or larger foreground/side animals, while SAM2 often marks a different smaller animal/region.
- Frames `00020`, `00035`, `00038` show severe disagreement: M2 follows large foreground/right animals, not a confidently same target instance.

Interpretation:

- M2 can convert a stable but wrong segmentation into accepted memory because its gate mostly tests geometry/stability/objectness, not identity.
- This is one of the clearest negative cases: the mask is non-empty and stable, but semantically likely wrong.

### 3.3 `lcgc29va` — tiny pedestrian in station/crowd scene

Observation from zoom sheet:

- The target is extremely small in a crowded station scene.
- M2 becomes empty in more frames than SAM2 (`18` vs `8` empty frames).
- Later M2 non-empty regions around frames `00023`-`00026` appear on a different small person/object cluster, while SAM2 marks tiny red regions elsewhere.

Interpretation:

- The current gate is too harsh for tiny, low-information targets and can create stale-memory behavior.
- When it later produces a mask, the selected small object is not clearly the original instance.

### 3.4 `msinig6m` — crowded koala/person scene with multiple targets/instances

Observation from zoom sheet:

- The initial mask includes animal/person-associated regions in a dense scene with many similar koalas and handlers.
- Later SAM2 often becomes empty, while M2 remains non-empty on large green koala/person clusters.
- The non-empty M2 regions frequently attach to a different left/foreground cluster rather than providing clear evidence of preserved original identity.

Interpretation:

- Lower empty count is misleading here: M2 may be recovering masks on salient distractors.
- Multi-object/same-class identity constraints are missing from the gate.

### 3.5 `3epdtmyr` — lions/large animals, foreground distractor

Observation from zoom sheet:

- At frame `00000`, the mask is on a foreground/bottom animal region.
- Later, when SAM2 is empty, M2 often outputs a foreground-bottom green blob (`000138`-`000160`) rather than the standing/background animal visible in the scene.
- Some M2 outputs occur at the frame boundary/foreground, which may reflect attraction to salient foreground texture rather than correct target continuity.

Interpretation:

- M2 reduces empty outputs (`6` vs SAM2 `58`) but the extra masks are not reliably correct. This again shows that “non-empty” is not equivalent to “right instance”.

### 3.6 `1qlssuz2` — small vehicle under bridges / similar vehicles

Observation from zoom sheet:

- The first-frame target is a very small vehicle in an aerial road scene.
- When the road/bridge geometry changes, M2 later outputs on a different visible vehicle/vehicle-like blob in some frames where SAM2 is empty or elsewhere.

Interpretation:

- Current geometry gate cannot determine whether a later vehicle is the original vehicle after occlusion/bridge crossing.
- This is another instance-level re-identification problem, not just memory poisoning.

### 3.7 `4vznweiu` — small letter bead on white background

Observation from zoom sheet:

- M2 is much more conservative than SAM2 (`21` empty frames vs `3`).
- SAM2 continues to mark a small bead-like region through many frames, while M2 often suppresses all output after the object cluster moves/occludes.

Interpretation:

- For small high-contrast but low-area objects, the area/objectness gate can over-suppress useful memory and output.
- M2 is likely worse unless the SAM2 continuation is actually an identity switch, which is not evident from the sheet alone.

### 3.8 `4f98052b` — AC Milan room, repeated red/black chairs/logos

Observation from zoom sheet:

- M2 alternates between empty frames and masks on repeated chair/seat-like regions.
- The repeated red/black visual pattern makes objectness/stability insufficient: wrong chairs can pass the gate if they are stable and mask-shaped.

Interpretation:

- Memory gating does not distinguish an original target from repeated layout elements.

### 3.9 `z6dx46qr` — low-contrast underwater/sand scene

Observation from zoom sheet:

- Early frames overlap well, then both methods become empty while the target is hard to see.
- Later M2 emits green blobs on sand/texture-like regions (`00025`, `00030`), while SAM2 remains empty.

Interpretation:

- M2 can introduce false positives after the target becomes low contrast or absent.
- In low-texture/low-contrast scenes, stability and area checks can be satisfied by background blobs.

## 4. Overall judgment

M2 is a useful diagnostic and partially correct mechanism, but the full visual evidence does **not** support treating it as a better submission than the original SAM2 baseline yet.

What M2 helps:

- It prevents many obviously unreliable frames from being written into SAM2's non-conditioning memory.
- It makes the autoregressive memory-poisoning hypothesis measurable through per-frame audit logs.
- It is conceptually aligned with SAM2's known failure mode in long occlusion/crowded scenes.

What M2 fails to solve:

- Stable wrong-object masks still pass the gate.
- It has no appearance/identity memory beyond geometry, area, objectness, and local stability.
- Tiny targets are often over-suppressed.
- Long occlusion followed by same-class reappearance remains unresolved.
- More non-empty masks can be worse than empty masks when they attach to distractors.

Recommendation:

- Keep M2 code as an experiment and diagnostic tool.
- Do **not** prefer M2 as the final submission unless hidden-score validation unexpectedly shows a gain.
- The next scientifically meaningful direction should add an identity-aware recovery condition, not merely tune thresholds. Candidate M3 directions:
  1. maintain a high-confidence identity anchor from frame 0 / last reliable frame;
  2. when reappearing after empty/skipped frames, require local crop appearance consistency or explicit re-detection around motion/anchor candidates before writing memory;
  3. treat empty during likely occlusion as acceptable, but make reactivation require stronger instance evidence;
  4. for tiny targets, use a separate small-object policy rather than one global area/motion gate.


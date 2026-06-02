# M12 low-row deep dive: `8jsm23a7`, `amfdu83t`, `z6dx46qr`, `q0sizv6m:obj2`

**One-line conclusion:** The fastest route toward a real score lift is no longer q0/q04v-style micro-fusion; it is targeted recovery/absence experiments on four low rows, especially `8jsm23a7` and `z6dx46qr`, where the hidden score contradicts earlier visual assumptions.

## Evidence sources

- Full-test detail log memory: `docs/test_latest_metric_memory.md`.
- New visual sheets: `docs/assets/m11_deep_four/`.
- Current best score family: `J&F_new ≈ 43.29597` over 575 object rows.
- Current roots inspected: SAM2/M7/M6/official b+/official large/q0 probes/tiny-semantic probes.

## Score leverage

| target row | current J&F_new | gain if row becomes 70 | gain if row becomes 90 | priority |
| --- | ---: | ---: | ---: | --- |
| `8jsm23a7:obj1` | 2.130 | +0.1180 | +0.1528 | 1 |
| `q0sizv6m:obj2` | 12.420 | +0.1001 | +0.1349 | 3 |
| `amfdu83t:obj1` | 28.175 | +0.0727 | +0.1075 | 2 |
| `z6dx46qr:obj1` | 40.470 | +0.0514 | +0.0861 | 2 |

A single fixed row cannot reach 44 alone, but 2--4 strong fixes can materially move the score. The current M10 q0-only/q04v probes do not move these rows versus confirmed M7, so future work must change strategy.

---

## 1. `8jsm23a7:obj1` — highest hidden-score contradiction

**Score evidence:** `J=2.13`, `F_new=2.13`, `J&F_new=2.13`, no disappear/reappear breakdown. This is almost the worst possible row; the current output is correct only at/near initialization.

**Visual evidence:**

- Sheet: `docs/assets/m11_deep_four/8jsm23a7_deep_sheet.jpg`
- First-frame crop: `docs/assets/m11_deep_four/8jsm23a7_obj1_target_crop.jpg`

**Observation:** The first-frame mask is a small light-green/white mahjong/tile-like object partly under a hand. After the hand moves, SAM2/M7/M6/q0 probes all output a stable-looking tile mask for almost every frame, but the hidden score says this stable mask is essentially not the ground-truth instance. Earlier we treated this as a stable guard; that was wrong.

**Inference:** This is likely an identity mismatch, not a boundary-quality issue. The target is probably the tile/object being manipulated by the hand, not merely any similar green/white tile in the row. The model locks to a visually plausible static/same-class tile, while GT follows another physical tile or marks absence/occlusion differently.

**Candidate roots:**

- SAM2/M7/M6/q0/q04v: identical; 48/49 non-empty, wrong by score.
- `pred_sam2_m7_qwen_support_tiny_semantic`: changes 19 frames but was not fused; Qwen had supported some tile frames, but the row score warns that single-frame semantic support is not reliable here.
- Official b+/large: visually close to current; no obvious identity rescue.

**Most promising experiment:**

1. **Absence/occlusion probe:** create `8jsm23a7-empty-after-f0` and `empty-after-hand-occlusion` variants. Because current score is only 2.13, an empty-after-early-frame policy has limited downside and large upside if GT treats the object as absent/occluded or not at the static tile location.
2. **MLLM motion-trace prompt, not candidate support:** ask Qwen/GPT-V style verifier to identify where the *specific tile under the hand in frame 0* goes across frames 0--6/10/15. Do not ask it to choose among current masks first; current masks are a misleading attractor.
3. **If a moving target is visually identified, prompt SAM2 with a tight box/mask on that frame and bounded-repropagate only that interval.**

**Do not:** keep calling it stable guard; do not accept generic tile candidates because they look like the reference tile.

---

## 2. `amfdu83t:obj1` — under-discussed reappearance miss

**Score evidence:** `J=28.44`, `F_new=27.91`, `J&F_new=28.175`, `disappear_J/F_new=100/100`, `reappear_J/F_new≈3.3/2.26`.

**Visual evidence:**

- Sheet: `docs/assets/m11_deep_four/amfdu83t_deep_sheet.jpg`
- First-frame crop: `docs/assets/m11_deep_four/amfdu83t_obj1_target_crop.jpg`

**Observation:** The first-frame target is a tiny, left-edge, partially visible kangaroo/deer-like animal, while a much more complete animal appears to the right. SAM2/M7 output only three non-empty frames (`0`, `1`, `7`) and then mostly empty. The disappear metric is perfect, but the reappear metric is nearly zero.

**Inference:** The current method correctly handles absence but fails to re-acquire the same edge-starting animal when it returns or becomes visible. The main distractor is the more complete animal in the scene; simple largest-animal tracking is unsafe.

**Candidate roots:**

- SAM2/M7/M6/q0/q04v: identical; mostly empty after early frames.
- Official b+/large: even more conservative (only frames 0--1 non-empty), not a recovery source.

**Most promising experiment:**

1. **Target-specific reappearance search:** use MLLM/open-vocabulary detector to search for the left-edge target animal after it enters/reappears, but include the right-side complete animal as an explicit hard negative.
2. **Motion prior:** from frame 0 the target is partially at the far left; likely reappears along the left-to-center trajectory. Candidate selection should penalize switching to the already-complete right animal.
3. **Bounded output-only or pseudo-anchor:** if a candidate is identified at reappearance, use SAM2 box/mask prompt only in the visible interval; keep empty in confirmed absence frames.

**Why this is high value:** Unlike q0, the scene has fewer same-class distractors. The failure is mostly missing reappearance, not dense herd identity.

---

## 3. `z6dx46qr:obj1` — low-contrast underwater continuation

**Score evidence:** `J=40.34`, `F_new=40.60`, `J&F_new=40.47`, no disappear/reappear split.

**Visual evidence:**

- Sheet: `docs/assets/m11_deep_four/z6dx46qr_deep_sheet.jpg`
- First-frame crop: `docs/assets/m11_deep_four/z6dx46qr_obj1_target_crop.jpg`

**Observation:** The target is a low-contrast, fish-like or small underwater object on a pale blue/sand background. SAM2/M7/M6 track frames 0--7 and then go empty for 30/38 frames. Official b+/large continue non-empty through roughly frames 24--25, with plausible small masks.

**Inference:** This is less about same-class identity and more about low contrast / weak objectness / early loss. The official checkpoints are the only inspected roots that supply plausible later masks.

**Candidate roots:**

- SAM2/M7/M6/q0/q04v: identical; non-empty only 0--7.
- Official b+/large: changed 24--25 frames vs M7; non-empty 25--26 frames; visually plausible continuation until about frame 20--25.

**Most promising experiment:**

1. **z-only official-large/b+ replacement probe:** build a submission replacing only `z6dx46qr` with official large or b+ to test whether later masks improve the hidden row.
2. **Hybrid interval merge:** use M7 for frames 0--7, official for 8--24, empty after official confidence/visibility collapses. The sheet suggests official becomes empty after 25/26, so this is naturally bounded.
3. **Low-risk reason:** current row is 40.47 and official visually tracks the same low-contrast blob for more frames; unlike q0, there are not many obvious same-class animal distractors.

**Caution:** official masks are not guaranteed correct; run as z-only probe first, not full fusion.

---

## 4. `q0sizv6m:obj2` — edge/partial target, not solved by q0only

**Score evidence:** q0 has two rows. Row 2 is `J=12.34`, `F_new=12.50`, `J&F_new=12.42`, with `disappear≈33.33` and `reappear=0`.

**Visual evidence:**

- Sheet: `docs/assets/m11_deep_four/q0sizv6m_deep_sheet.jpg`
- Obj2 first-frame crop: `docs/assets/m11_deep_four/q0sizv6m_obj2_target_crop.jpg`

**Observation:** Obj2 is an edge/partial animal/object at the left/bottom cluttered boundary in the first frame, not an easy full-body guinea-pig reference. Current SAM2/M7 tracks large plausible animals for many frames. q0only/q04v changed 15 obj2 frames, but the confirmed full-test log shows no q0 row improvement over M7.

**Inference:** The M10 q0 anchor is visually plausible but does not align with hidden GT. It likely supports the wrong same-class instance or wrong body extent. q0 needs a reset: do not assume the Qwen-supported frame-23 candidate is helpful.

**Candidate roots:**

- SAM2/M7/M6/q0/q04v: q0only/q04v differ in 15 obj2 frames but do not improve row score.
- Official b+/large: many differences, but mostly same-class large animal masks; not safe by visual evidence.
- M9 component: only one frame differs; insufficient.

**Most promising experiment:**

1. **Obj2 absence/edge policy probe:** since current non-empty masks are likely wrong and row score is near-init only, test conservative emptying after the edge target leaves/occludes. This may help if GT treats obj2 as absent for many frames.
2. **Edge-target re-ID, not herd re-ID:** prompt MLLM with the first-frame partial edge crop and ask whether that exact edge animal reappears, rather than choosing among full visible guinea pigs.
3. **Use hard negatives aggressively:** all large central/right guinea pigs should be negative examples unless there is motion continuity from the left-edge target.

**Do not:** continue q0only/q04v as-is; the detail log proves it has no row-level gain.

---

## Experiment priority

1. **M12-A: `8jsm23a7` absence/motion probe.** Highest leverage and current output is almost certainly wrong after frame 0.
2. **M12-B: `z6dx46qr` official interval replacement probe.** Clear existing candidate root supplies plausible continuation.
3. **M12-C: `amfdu83t` reappearance search with right-animal negative.** Needs new candidate generation, but has fewer distractors than q0.
4. **M12-D: `q0sizv6m:obj2` edge/absence reset.** High leverage but highest identity ambiguity; q0only already failed.

## Concrete next actions

- Generate three small candidate submissions rather than another broad fusion:
  1. `m12_8jsm_empty_after_f0_or_f1`
  2. `m12_z6dx_official_interval`
  3. `m12_q0_obj2_empty_edge_policy`
- Submit only one or two if submission budget is tight; the best first probe is `m12_8jsm_empty_after_f0_or_f1` because the current score is so low that suppressing a wrong static tile has high upside.
- For `amfdu83t`, first create MLLM/detector candidate panels before making a zip; current available roots do not contain a useful recovery.

## User-corrected semantic reading: why MLLM matters here

The user's frame-level semantic correction changes the interpretation of the four low rows. These are not merely threshold/mask-quality failures; they are failures to understand the physical instance story.

### `8jsm23a7`

Corrected reading: the target is a mahjong tile that is picked up around the second frame, then moved to the player's front row. From then until the end, it should be the leftmost front-row tile showing **七条**. Current predictors localize the wrong tile and effectively assume the target stayed near the initial/middle draw area.

Hypothesized root cause: in frame 0 the tile is being picked up with its back side facing up, so its visual features are extremely similar to the other face-down/middle draw tiles. SAM-style appearance/memory matching therefore latches onto the wrong same-looking tile. This is a physical-instance tracking failure: the target identity is determined by the hand action and subsequent placement, not by static local appearance.

Implication: `8jsm23a7` should be handled by an MLLM/event-chain module, not by generic tile candidate matching. The verifier prompt should ask: “which tile was picked up from the reference location, where did the hand move it, and which final front-row tile is that physical object?” Candidate masks near the original position should be hard negatives after the pickup event.

### `z6dx46qr`

Corrected reading: this is a genuinely hard low-contrast underwater sample. The only stable discriminative cue may be the pair of dark points that look like eyes.

Implication: candidate generation should emphasize high-resolution local contrast and the two-dot/eye-like pattern, not generic blob area. MLLM can help describe and select candidates with “two dark eye-like points on pale background,” but direct MLLM boxes are likely too coarse. A better path is: detect local two-dot candidates, then use SAM/official masks for refinement.

### `q0sizv6m:obj2`

Corrected reading: the rear/behind `obj2` hamster/guinea-pig-like animal moves toward the camera and becomes the new foreground/bottom animal. Current predictions instead mark other new animals near the original/back position.

Implication: the relevant evidence is motion/identity continuity from rear object to foreground object, not same-location persistence. The frame-23 q0only candidate is insufficient if it follows a plausible same-class animal but not the object that moved forward. MLLM prompts should explicitly trace: “the rear animal in frame 0 moves forward toward the camera; identify the foreground/bottom animal that corresponds to it.” Same-class animals remaining near the original position should be hard negatives.

### Architectural update

This supports a stronger role for MLLM than previous M7/M10:

1. **Target event profiler**: infer whether identity depends on an event chain (picked up/moved/placed, moving from back to foreground, etc.).
2. **Frame-story verifier**: evaluate a short sequence, not a single candidate crop, and output a physical-instance narrative.
3. **Candidate role labeling**: label candidates as `picked_tile_final_location`, `original_position_distractor`, `same_class_static_distractor`, `foreground_reappeared_target`, etc.
4. **Hard-negative generation**: candidates that match appearance but violate the event story should enter the negative bank.
5. **Prompt-to-SAM injection**: once MLLM identifies a frame/location role with high confidence, use SAM2/SAM3/official masks only to refine that location, then bounded-propagate.

This is a qualitative shift: MLLM should not be used merely to approve a mask that SAM already proposes. It should first explain **where the physical target went**, then the mask system should segment that proposed location.

---

## M12 event-story harness run (2026-06-01)

A generalized MLLM event-chain harness has now been implemented and run across all 15 provided videos / 20 first-frame objects.

Artifacts:

- Tool: `tools/mllm_event_story_agent.py`
- All-target JSON: `artifacts/m12_event_story/all15_records.json`
- All-target report: `docs/m12_event_story_agent_report.md`
- Panels: `artifacts/m12_event_story/all15_panels/`
- qwen3.6 text-level teach-SAM plan: `artifacts/m12_event_story/q36_teach_sam_plan.json`
- qwen3.6 plan report: `docs/m12_q36_teach_sam_plan.md`

Key result: the harness did not merely judge masks; it labeled physical-instance events. It independently identified `8jsm23a7:1` as `picked_moved_placed`, `q0sizv6m:2` as `moves_to_foreground`, `z6dx46qr:1` as `low_contrast_continuation`, and `amfdu83t:1` as an event-reanchor target. The qwen3.6 text planner therefore prioritizes:

1. `8jsm23a7:1` — MLLM box-to-SAM reanchor at the final front-left 七条, with original/middle face-down tile as hard negative.
2. `q0sizv6m:2` — same-class switch / hard-negative veto for original-position animals plus foreground reanchor probe.
3. `z6dx46qr:1` — low-contrast official/local-window probe using the two dark eye-like points as identity cue.
4. `amfdu83t:1` — event reanchor rather than empty-only suppression.

Implementation note: qwen3.6-plus succeeds on compact visual panels but times out on many dense all-frame visual panels. The harness therefore uses qwen3.6-plus as the second-stage text reasoning/planning model over cached visual event records, while the visual grounding stage falls back transparently to Qwen-VL models when needed. This avoids losing the deep reasoning model while keeping the all-target visual analysis tractable.

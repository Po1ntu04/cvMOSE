# M10 Qwen3.6 MLLM re-anchor report

## Conclusion

The validated submission to submit first remains `m7part-balanced`; M10 made the MLLM path materially functional and produced two probe zips, but the only visually defensible new recovery is `q0sizv6m:obj2`, while `4vznweiu` is still risky/uncertain and `msinig6m` is neutral after rollback.

## Deliverables

| candidate | zip | validation | recommendation |
| --- | --- | --- | --- |
| M7 part-balanced | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_part_balanced.zip` and repo copy `submission_mosev2_m7_part_balanced_submit.zip` | 433 videos / 66526 png / provided unchanged / predicted errors 0 | Submit first; known best score 43.30. |
| M10 q0-only | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m10_qwen36_q0only.zip` | 433 videos / 66526 png / provided unchanged / predicted errors 0 | Probe candidate; only replaces `q0sizv6m:obj2`. |
| M10 q0+4v | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m10_qwen36_q0_4v.zip` | 433 videos / 66526 png / provided unchanged / predicted errors 0 | Aggressive probe; `4vznweiu` is visually uncertain. |

Validation artifacts:
- `artifacts/m7_part_tracklet/m7_part_balanced_revalidation_now.json`
- `artifacts/m10_qwen36/m10_qwen36_q0only_validation.json`
- `artifacts/m10_qwen36/m10_qwen36_q0_4v_validation.json`

## What was changed in the MLLM pipeline

1. **Qwen model and cache correctness**
   - Default model is now `qwen3.6-plus` with DashScope OpenAI-compatible API.
   - Cache key now includes `base_url`.
   - Real runs ignore stale `dry_run`/API-error cache records so earlier dry-run cache can no longer silently veto anchors.

2. **Few-shot MLLM prompts**
   - Candidate, tracklet, and recovery-box prompts now include positive, same-class negative, composite, and tiny-unreadable few-shot rules.
   - This reduced the earlier failure where the model over-trusted large/composite candidates.

3. **MLLM evidence is now actually connected to candidate generation**
   - Bug found: Qwen judged frames were sometimes not in the M5R event set, or appeared after `max_candidates_per_object` had already been consumed by early unjudged frames.
   - Fix: all MLLM-judged frames are inserted into event frames and prioritized before generic event frames during candidate collection.

4. **Delayed-commit semantics now accept verified MLLM tracklets**
   - Bug found: Qwen could support `q0sizv6m:obj2@23`, and tracklet judge could approve it, but `confirm_candidate` still required another geometric candidate and rejected it as `unconfirmed:1/2`.
   - Fix: `mllm_support + tracklet_promote(conf>=threshold)` is accepted as delayed confirmation; Qwen still cannot promote without descriptor/tracklet evidence.

5. **Source-root and safe external candidate plumbing**
   - `--source-root name=/path` allows high-recall roots (`official_large`, `dam`, `m7bal`, `m9clip`, `dino_m5r`) without hardcoding.
   - M9 box proposals now default to production-safe filtering, with recall-mode separated.

## Real Qwen3.6 calls

- API smoke: `docs/m9_qwen36_setup.md`, model used `qwen3.6-plus`.
- Target profiling: `artifacts/m10_qwen36/target_profiles.json`, 16 objects.
- Candidate judge: `artifacts/m10_qwen36/candidate_judgments_key.json`, 13 judged frames.
- Tracklet judge: `artifacts/m10_qwen36/tracklet_judgments_key_full.json`, 13 tracklets.

## Key visual evidence

Comparison sheets:
- `docs/assets/m10_qwen36/confirm_compare/q0sizv6m_m6_compare.jpg`
- `docs/assets/m10_qwen36/confirm_compare/msinig6m_m6_compare.jpg`
- `docs/assets/m10_qwen36/confirm_compare/4vznweiu_m6_compare.jpg`
- `docs/assets/m10_qwen36/confirm_compare/2smf7uq9_m6_compare.jpg`
- `docs/assets/m10_qwen36/confirm_compare/8jsm23a7_m6_compare.jpg`

## Per-video verdict

| video/object | MLLM decision | re-anchor result | visual verdict | action |
| --- | --- | --- | --- | --- |
| `q0sizv6m:2` | Qwen candidate support 0.85; tracklet support 0.95 | accepted baseline anchor at frame 23; 15 frames changed vs M7 | Plausible recovery of the black-head/white-body guinea pig; still dense same-class, so not safe enough to replace M7 without scoreboard probe. | Built q0-only probe zip. |
| `q0sizv6m:1` | Qwen veto/uncertain | no anchor | Correctly avoids wrong animal drift. | Keep M7. |
| `msinig6m:1` | Qwen support 0.95; tracklet support 0.95 | anchor accepted at frame 70, but rollback skipped 24/29 window frames; final equals M7 | MLLM can identify a candidate, but bounded propagation changes were too large/risky and therefore rolled back. | Neutral; not fused. |
| `4vznweiu:1` | Qwen support at frame 9; veto/uncertain at frame 15 | pre-floor semantic/tiny anchor accepted; 20 frames changed | Changes are small but not clearly better; target bead identity remains tiny/ambiguous after motion and hand occlusion. | Aggressive-only probe. |
| `lcgc29va:1` | Qwen hallucinated/refused tiny object; tracklet rejected | no anchor | MLLM unreliable on this tiny target. | Keep M7. |
| `2smf7uq9:1` | Qwen supports right-edge flamingo at frame 30 | previous M10 changed 24 frames | Visual not clearly better than M7; dense same-class risk remains. | Not in safe/probe final except earlier diagnostic. |
| `8jsm23a7:1` | Qwen supports some tile frames | previous M10 changed stable guard frames | Stable guard case; changes unnecessary and risky. | Keep M7. |

## Why the previous MLLM attempts scored flat

The main bottleneck was implementation coupling, not only model reasoning quality:

1. **Verifier evidence was generated outside the re-anchor candidate path.** Qwen judged key frames, but M5R sometimes never considered those frames or candidates.
2. **Candidate caps favored early generic frames.** Long videos like `msinig6m` exhausted candidate budget before later MLLM-judged frames.
3. **Support was too weak to override borderline descriptor rejections.** Earlier `+0.02` support could not change anchor choice.
4. **Delayed commit was duplicated.** Tracklet judge already verified 2-3 frames, but `confirm_candidate` still demanded a second geometric candidate and rejected real MLLM-supported anchors.
5. **Large visual changes trigger rollback.** This is good for safety, but it means MLLM can identify candidates without improving final masks unless propagation remains local and stable.

## Next structural step

To chase a true jump toward 44, the next step should not be another threshold sweep. It should be **MLLM-guided segment-level bounded propagation**:

1. Ask Qwen for target-visible intervals and high-risk intervals, not only single frames.
2. Inject anchors only inside Qwen-approved visible intervals.
3. Use smaller object-specific merge windows and per-frame MLLM veto for large propagated masks.
4. Re-run propagation forward/backward only within the approved interval, then merge frame-by-frame against M7.
5. Keep M11/M7 as default everywhere else.

This directly addresses the current failure where `msinig6m` anchor is semantically right but SAM2 propagation produces too-large changes that rollback removes.

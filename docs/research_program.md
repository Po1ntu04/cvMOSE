# Research Program: principled MOSEv2 optimization

## 1. Task

Given a video sequence and first-frame instance mask(s), predict the complete video masklet for each object id.  The target is **instance identity preservation**, not category discovery.

Formal view:

- input video: `V = (I_0, ..., I_{T-1})`
- first-frame labels: `Y_0 in {0..K}^{H x W}`
- output: `Y_hat_{0:T-1}` with stable ids

The scoring target is submission quality (`J&F'`), but the research target is to understand and reduce failures caused by disappearance/reappearance, same-class distractors, tiny objects, and edge truncation.

## 2. Challenge attribution

| Failure symptom | Deeper cause | Seen in |
|---|---|---|
| mask becomes empty while object is visible | visibility gate too conservative, small target below feature resolution, or memory loses target | `1qlssuz2`, `lcgc29va`, `z6dx46qr` |
| mask stays non-empty but moves to wrong object | identity ambiguity after occlusion; same-class distractor more salient | `r13u5z4y`, `q0sizv6m`, `msinig6m` |
| target never recovers after full occlusion | single first-frame anchor cannot re-identify target after appearance/position reset | `r13u5z4y`, `2smf7uq9` |
| two ids overlap or swap | independent per-object tracking lacks cross-object competition | `msinig6m`, `q0sizv6m`, `pe0d85lk` |
| tiny/edge target is lost immediately | object occupies too few encoded tokens; partial contour is not enough identity evidence | `amfdu83t`, `lcgc29va`, `1qlssuz2` |

## 3. Insights

### Insight A — The hard part is identity, not mask boundary

Once the target disappears behind a hand, edge, animal, or same-class cluster, the next visible mask must answer “is this the same instance?”  Boundary refinement alone cannot solve this.

### Insight B — Visibility is a latent state that should gate masks

For this task, a wrong same-class mask can be worse than an empty mask.  Therefore mask prediction should be conditioned on an explicit or inferred visible/confirmed state.

### Insight C — Memory quality matters more than memory quantity

Adding memories after drift poisons future predictions.  Pseudo prompts must be accepted only after semantic/geometry confirmation.

### Insight D — Tiny targets need relative-resolution repair

A target with <1% area can vanish after downsampling.  Crop-based reruns are not just engineering tricks; they change the target-to-token ratio and can alter the feasible decision boundary.

### Insight E — Multi-object VOS requires competition

When several ids coexist, independent trackers can assign the same visual evidence to multiple objects.  Cross-id mutual exclusion and trajectory consistency are principled approximations of shared object-level context.

### Insight F — SAM3 ideas can be borrowed without making SAM3 the primary model

Presence, confirmation delay, detector-guided re-prompting, and positive/negative exemplar logic are system-level ideas.  They can guide SAM2 re-anchoring and postprocessing even if SAM3 public prompting is discarded.

### Insight G — The missing layer is re-acquisition, not another suppressor

M11, M2, M3, and M4 each constrain existing SAM2 evidence, but they do not create a sufficiently independent path to recover the original instance after occlusion.  The next method must separate proposal generation from identity verification and delay memory/final-output commitment until reappearance is confirmed.

## 4. Method roadmap

Do **not** apply everything at once. Each method must correspond to a failure attribution and leave an experiment record.

### M1. Visibility-gated SAM2 baseline

- Input: SAM2 masks/logits and per-frame area/bbox continuity.
- Method: suppress outputs when target confidence/geometry indicates likely distractor or true invisibility.
- Hypothesis: improves cases where wrong same-class masks hurt more than empties.
- Main risk: over-suppression during true reappearance.

### M2. Reliable tracker-memory gate

- Input: original SAM2 per-frame mask logits, object score, previous/last-reliable mask geometry.
- Method: after each non-conditioning prediction, decide whether the current mask is reliable enough to enter SAM2's memory bank; unreliable masks are still output for the current frame but are not written as future non-conditioning memory.
- Hypothesis: reduces autoregressive error accumulation from drift frames in occlusion/same-class distractor cases.
- Main risk: over-strict gating makes memory too sparse, while stable wrong-instance tracks may still pass geometry-only reliability.

### M3. Confirmed multi-anchor re-prompt

- Input: first-frame GT + selected high-confidence pseudo masks, preferably only after M2-style reliability checks.
- Method: rerun SAM2 with additional prompts on verified frames, then merge bidirectional predictions.
- Hypothesis: reduces memory staleness after long occlusion.
- Main risk: pseudo prompt identity switch poisons memory.

### M4. Tiny-target crop tracking

- Input: full-frame SAM2 trajectory or search window.
- Method: crop around expected target area, rerun at higher relative resolution, map mask back.
- Hypothesis: recovers targets lost by feature downsampling.
- Main risk: crop misses fast motion or locks onto distractor.

### M5. Cross-id competition and duplicate suppression

- Input: multi-id predictions in the same video.
- Method: resolve overlap, reject implausible id crossings, prefer history-consistent assignment.
- Hypothesis: reduces id swap/merge in `msinig6m`, `q0sizv6m`, `pe0d85lk`.
- Main risk: suppresses legitimate close contact.

### M6. Selective SAM3.1-assisted candidate replacement

- Input: SAM2 and SAM3.1 adapter outputs.
- Method: use SAM3.1 only on video segments where it is semantically verified better; never use public-simple.
- Hypothesis: helps specific videos such as `3epdtmyr` or `z6dx46qr` where adapter has useful non-empty continuity.
- Main risk: SAM3.1 emptiness or identity loss is mistaken for correctness.

### M5R. Training-free re-anchor tracker

- Input: baseline/M11/M2-light/M3 audits plus optional crop/SAM3 proposals in uncertain windows.
- Method: detect post-gap uncertainty, retrieve candidate masks, verify them against first-frame identity and hard negatives, then commit only after confirmation delay.
- Hypothesis: a proposal + verifier + delayed-commit loop can recover targets that M11 only empties and M4 only sees more clearly.
- Main risk: verifier remains correlated with the wrong proposal source, causing stable distractors to be accepted as anchors.
- Design note: see `docs/reanchor_tracker_direction.md`.

## 5. Experiment gates

A method can move from idea to submission candidate only if:

1. it states the target failure mode;
2. it names affected videos/frames;
3. it produces contact-sheet or frame audit evidence;
4. it logs changed files and exact command;
5. it compares against the SAM2 baseline before merging into a submission zip.

## 6. Current recommendation

Primary branch: **SAM2 + training-free re-anchor tracker: propagation + uncertainty management + proposal retrieval + identity verification + delayed commit**.

Secondary branch: **SAM3.1 adapter as diagnostic candidate source only**.

Discarded branch: **SAM3.1 public-simple prompting** for this homework interface, unless a future experiment changes the prompt construction fundamentally.

# External Method Study for M5R Re-anchor Tracker

Date: 2026-05-30
Status: source-backed design input, not yet an implementation result

## Direct recommendation

The next meaningful improvement should be a **training-free re-anchor tracker**: keep SAM2/M11 as the stable propagation backbone, but add an explicit loop for uncertainty state, candidate retrieval, identity verification, and delayed memory/output commit.  External evidence strongly suggests that continued M11/M2 threshold tuning is the wrong layer: stronger methods add a new identity evidence path and govern memory promotion, rather than only suppressing bad masks.

## Evidence used

### Upstream / official sources

| source | status | what it establishes for us |
|---|---:|---|
| [MOSEv2 paper, arXiv:2508.05630](https://arxiv.org/abs/2508.05630) and [MOSE-api](https://github.com/henghuiding/MOSE-api) | official paper/code page | MOSEv2 is dominated by disappearance/reappearance, occlusion, small objects, distractors, low-light/weather/multi-shot/knowledge cases.  The paper reports SAM2 degradation on MOSEv2 and proposes practical SAM2 improvements including RCMS, MQF, MSS, and LVT. |
| [FudanCVL/MOSEv2_baseline](https://huggingface.co/FudanCVL/MOSEv2_baseline/tree/main) | official artifact page | Public baseline artifacts include `sam2_b+_MOSEv2_rcms_mqf_mss_lvt_submission.zip`, `sam2_l_MOSEv2_rcms_mqf_mss_lvt_submission.zip`, and MOSEv2-tuned SAM2.1 checkpoints. |
| [SAM2Long paper, arXiv:2410.16268](https://arxiv.org/abs/2410.16268) and [Mark12Ding/SAM2Long](https://github.com/Mark12Ding/SAM2Long) | paper + open source | SAM2Long frames SAM2 failure as greedy-memory error accumulation and implements a training-free multi-path memory tree with branch pruning. |
| [SeC paper, arXiv:2507.15852](https://arxiv.org/abs/2507.15852) and [OpenIXCLab/SeC](https://github.com/OpenIXCLab/SeC) | paper + open source | SeC injects concept-level object memory only at scene/transition-like moments, combining semantic reasoning with feature matching for complex VOS. |
| [TEP 1st PVUW MOSE report, arXiv:2604.00395](https://arxiv.org/abs/2604.00395) | technical report, no code found during this pass | Training-free target-type routing: SAM3 is weak on tiny and semantic-dominated targets, so the method adds external tracker / MLLM prompts. |
| [OAMVOS 2nd PVUW MOSE report, arXiv:2604.22837](https://arxiv.org/abs/2604.22837) | technical report, no code found during this pass | Reliability-aware state machine, branch recovery, delayed memory promotion, and selective memory selection are explicitly proposed for occlusion/reappearance. |
| [Re-Prompting SAM3 via Object Retrieval, arXiv:2603.23788](https://arxiv.org/abs/2603.23788) | technical report, no code found during this pass | Detector + DINOv3 object-level matching retrieves reliable later anchors, then re-prompts the tracker with multiple anchors. |
| [SAM3-DMS, arXiv:2601.09699](https://arxiv.org/abs/2601.09699) and [FudanCVL/SAM3-DMS](https://github.com/FudanCVL/SAM3-DMS) | paper + open source | Decoupled per-object memory selection avoids group-average memory decisions and reduces multi-object identity drift. |

### Local source snapshots inspected

For code reading only, not vendored into this repository:

```text
/tmp/cvmose_research/
├── SAM2Long/     # cloned from Mark12Ding/SAM2Long
├── SeC/          # cloned from OpenIXCLab/SeC
├── SAM3-DMS/     # cloned from FudanCVL/SAM3-DMS
└── mosev2_2508.05630.txt  # local pdftotext extraction for RCMS/MQF/MSS/LVT passages
```

## What to learn from each method

### 1. MOSEv2 / SAM2RCMS: remember the right pre-disappearance evidence

**Useful principle.**  SAM2 is usually strong while the target is continuously visible before the first true disappearance.  RCMS exploits that by moving selected high-quality pre-disappearance memories into conditioned memory when disappearance happens.  This is not the same as blindly trusting recent frames: the timing of selection matters.

**Primary evidence.**  The paper's local text extraction states that RCMS selects the `N` nearest high-quality memories from the pre-disappearance sequence, uses MQF with a quality term `Q = score_iou * score_occ * maskness`, and reports RCMS +3.3 J&F, MQF +0.9, MSS +0.4, LVT +0.9 on SAM2-B+; all improvements together raise SAM2-B+ from 46.0 to 51.5 J&F.  It also says performance peaks around `N=4` and quality threshold `theta=0.6`.

**M5R mapping.**

- Maintain a `ReferenceBank` separate from SAM2's raw non-conditioning memory.
- Store only frames from stable visible intervals: first GT frame plus high-quality pre-gap frames.
- On disappearance, freeze/mark these as recovery references; do not let post-gap uncertain masks overwrite them.
- Use MQF-like fields where available (`iou`, object/presence score, maskness/stability), but add identity checks before promotion.

**Do not copy blindly.**  RCMS recovers memory diversity but still depends on SAM2's own quality signals.  In same-class distractor cases like `r13u5z4y`, a high-quality mask can still be the wrong strawberry piece, so M5R needs an independent verifier before adding pseudo anchors.

### 2. SAM2Long: beam search is useful; cumulative IoU-only scoring is insufficient for us

**Useful principle.**  Do not collapse to one greedy path when the tracker is uncertain.  Keep a small number of segmentation hypotheses and prune later.

**Code observations.**

- `sam2/sam2_video_predictor.py` adds `num_pathway`, `uncertainty`, branch expansion from multimask outputs, cumulative branch scores, and a `mem_pick_indexs` mapping that selects which branch memory each future path reads.
- `sam2/modeling/sam2_base.py` filters valid memory frames using `iou > iou_thre` and positive object score, and always includes the immediately previous frame.
- CLI exposes `--num_pathway`, `--iou_thre`, and `--uncertainty` in `tools/vos_inference.py`.

**M5R mapping.**

- Borrow the **branch object model**: `BranchState(frame, mask, score, memory_source, parent, evidence)`.
- Keep branch count small (`2-3`) and only in uncertain windows, not full-video by default.
- Use existing M3 candidate table as the branch audit table; extend it with lineage and verifier fields.

**Do not copy blindly.**  Cumulative predicted IoU can prefer stable wrong same-class distractors.  For MOSE15, branch survival must include identity evidence, not only mask quality.

### 3. SeC: trigger high-level evidence sparsely, not every frame

**Useful principle.**  High-level object concepts help when feature matching is ambiguous, but they should be triggered only around scene/transition/uncertainty events to control cost and avoid semantic overreach.

**Code observations.**

- `inference/modeling_sec.py` keeps an `mllm_memory` list seeded by the initial mask and appends reliable later masks.
- It checks scene change with `is_scene_change_hsv`; only when triggered does it build a labeled image sequence and call `predict_forward` for a language/concept embedding.
- `inference/sam2_video_predictor.py` injects `language_embd` through token attention and fuses it with normal memory-conditioned features.

**M5R mapping.**

- Implement an optional `SemanticHintProvider` interface, but keep it off in the first smoke unless the target is semantic-dominated.
- The more immediate borrow is not LVLM itself; it is the **trigger discipline**: expensive/global evidence only when uncertainty is high.

**Do not copy blindly.**  SeC is partly trained/fine-tuned and model-heavy.  Full SeC is outside our current training-free, minimal engineering constraint.

### 4. TEP: route by failure type

**Useful principle.**  Complex MOSE cases should not share one universal heuristic.  Tiny targets, regular targets, and semantic-dominated targets need different proposal sources.

**M5R mapping.**

- Add a lightweight target classifier from first-frame mask/video stats:
  - `tiny`: area <= about 1% or edge-heavy; allow crop/local-search proposals.
  - `occlusion_reappear`: disappearance gap detected; enable re-anchor branch logic.
  - `semantic_dominated/stuff_like`: avoid aggressive identity switching; optionally use semantic hint later.
  - `regular`: keep SAM2/M11 stable path.
- Store this route in audit JSON and comparison sheets.

**Do not copy blindly.**  TEP uses SAM3, external tracking, and MLLM prompts.  Since our experiments show SAM2 fits the first-mask VOS task better, borrow the routing structure, not the full SAM3-first stack.

### 5. OAMVOS: state machine + delayed promotion is the missing layer

**Useful principle.**  Gate is not just `write` or `do not write`; it should control mode transitions and delayed commitment.  OAMVOS explicitly separates stable propagation from ambiguous/recovery modes and commits memory only after reconfirmation.

**M5R mapping.**

- States: `STABLE`, `AMBIGUOUS`, `OCCLUDED`, `RECOVERY_BRANCH`, `RECONFIRMED`, `REJECTED`.
- In `AMBIGUOUS`/`RECOVERY_BRANCH`, branch outputs may be visualized or temporarily used, but must not overwrite main long-term anchors.
- Promotion requires at least two independent signals, e.g. temporal consistency + exemplar similarity, or bidirectional consistency + low distractor score.
- Keep first conditioning frame and selected pre-gap anchors always available.

**Do not copy blindly.**  The public source found is an arXiv report, not code.  Treat details as design evidence, not implementation reference.

### 6. Re-Prompting SAM3 via Object Retrieval: recovery needs a new evidence path

**Useful principle.**  The method's key contribution is not “SAM3 is better”; it is automatic later-frame pseudo-anchor retrieval.  Detector proposals plus object-level matching create evidence outside the propagation path.

**M5R mapping.**

- First M5R smoke can implement a simpler version:
  1. detect uncertain post-gap windows;
  2. generate candidate boxes/masks from existing SAM2/M11/M4/SAM3 candidate sources;
  3. extract masked crop descriptors from first anchor + selected pre-gap anchors;
  4. score candidates against positive anchors and a hard-negative bank;
  5. inject only reconfirmed pseudo anchors.
- If time allows, add DINO-like descriptors later; initial implementation can start with masked color/texture/shape descriptors plus short-window consistency, but the interface should allow replacing descriptors.

**Do not copy blindly.**  If SAM3 detector misses tiny/occluded instances or returns all same-class candidates, retrieval can be noisy.  Candidate retrieval must be guarded by negative-bank and delayed promotion.

### 7. SAM3-DMS: per-object memory control, not group-level average

**Useful principle.**  Memory quality must be object-specific.  Group-level or video-level average reliability can hide the failure of the exact tiny target we care about.

**Code observations.**

- `sam3/model/sam3_tracker_base.py` has `use_memory_selection`, `mf_threshold`, `use_decoupled_selection`, `cal_mem_score(object_score_logits, iou_score)`, and `frame_filter(...)`.
- `sam3/model/sam3_video_base.py` changes object addition under `use_decoupled_selection`: each object gets an independent tracker state instead of a coupled group state.

**M5R mapping.**

- Keep all audits and state transitions per video/object id, not per video.
- Never let one object's stable score authorize memory promotion for another object.
- If later extending to multi-object propagation, isolate object state by default.

## Concrete M5R architecture to implement next

```text
SAM2/M11 stable propagation
      │
      ▼
Per-object TargetProfile + ReferenceBank
      │
      ▼
UncertaintyStateMachine
      ├── STABLE: keep baseline/M11, collect high-quality references
      ├── OCCLUDED: output empty or conservative hold; freeze anchors
      ├── RECOVERY_BRANCH: generate candidate masks/boxes in post-gap windows
      └── RECONFIRMED: promote candidate as pseudo anchor after delayed verification
      │
      ▼
CandidateGenerator(s)
      ├── baseline/M11/M2/M4 existing masks
      ├── local motion/search crop proposals
      ├── optional SAM3 candidate proposals
      └── optional reverse propagation from high-confidence candidate
      │
      ▼
IdentityVerifier
      ├── positive anchor similarity: first GT + RCMS-like pre-gap anchors
      ├── negative bank similarity: known distractors/rejected candidates
      ├── short-window temporal consistency
      ├── bidirectional/cycle check when available
      └── mask quality sanity (area, stability, object score)
      │
      ▼
DelayedCommitPolicy
      ├── reject / keep empty
      ├── output-only candidate, no memory write
      └── promote pseudo-anchor and optionally rerun SAM2 in a bounded window
```

## First implementation slice: M5R-smoke, not full rerun

Target videos:

1. `r13u5z4y` — hard same-class strawberry reappearance; must not accept nearby wrong slice.
2. `8jsm23a7` — small object after occlusion; check recovery vs emptying.
3. `lcgc29va` — tiny/crowded target; check crop/local proposal benefit.
4. `q0sizv6m` or `msinig6m` — same-class/multi-instance stress.

Minimum deliverables:

- `tools/apply_m5r_reanchor.py` or equivalent, consuming existing baseline/M11/M2/M3/M4 outputs and audits.
- `docs/m5r_reanchor_experiment.md` with source-backed design, deviations, and per-video verdicts.
- `docs/assets/m5r_reanchor/smoke/` comparison sheets: RGB, baseline, M11, candidate, verifier scores, final.
- Submission zip only if smoke shows at least one real identity re-acquisition without damaging the provided 418 outputs.

## Risks and safeguards

| risk | why it matters | safeguard |
|---|---|---|
| Verifier is correlated with proposal source | stable wrong distractor can pass mask quality gates | require positive/negative identity evidence and delayed confirmation |
| Over-emptying improves safety but hurts J/F | M11 already tends toward safer rejection | track `empty_after_visible`, `false_absent_risk`, and compare with GT on train/contact where possible |
| Pseudo-anchor pollutes memory | main SAM2 weakness is autoregressive memory drift | output-only branch first; promote only after reconfirmation; keep rollback path |
| SAM3 detector/retrieval is expensive/noisy | prior simple SAM3.1 route was worse than SAM2 | use SAM3 only as optional candidate source, not as primary tracker |
| DINO/LVLM dependencies delay experiment | environment may become brittle | define descriptor/provider interface; start with cheap descriptors, allow later replacement |

## Updated stance

The strongest external agreement with our internal findings is: **identity recovery requires new anchors and memory promotion discipline**.  M5R should therefore be implemented as a structured re-anchor system, not as another postprocessing threshold sweep.

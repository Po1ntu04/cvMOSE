# M5R Reappearance-Aware ReAnchor Plan

Date: 2026-05-30
Status: Phase A/B scaffold implemented and reviewed; visual comparison shows no final-quality improvement yet

## One-line decision

Converge the next mainline to **RAR: Reappearance-Aware ReAnchor on top of SAM2**. SAM3/MLLM may enter later only as candidate/retrieval sources, not as the primary tracker.

## Why this supersedes more M11/M2/M3/M4 tuning

The observed wall is not a missing threshold. M11/M2/M3/M4 constrain SAM2 after, during, or around propagation, but they do not establish a new evidence loop for confirming identity after disappearance. RAR makes the missing loop explicit:

```text
SAM2 stable propagation
  -> uncertainty / disappearance detection
  -> freeze main memory writes
  -> recover from pre-disappearance anchors or later mined anchors
  -> verify identity over time
  -> delayed promotion into long-term anchor/memory
```

## System scope

RAR is deliberately small in v1. It has one main entrypoint and one thin primitive module:

- `tools/infer_mosev2_sam2_rar.py`
- `src/cvmose/reanchor.py`
- `scripts/run_b101_rar.sh`

No broad framework is introduced. Candidate retrieval, SAM3 detector use, DINO-like embedding, MLLM, and crop refinement remain staged extensions.

## Core objects

### `FrameAudit`

Every frame/object records:

- `state`: `stable`, `ambiguous`, or `recovery`;
- `base_mask_stats`: area, area fraction, bbox, centroid, edge touch;
- `quality_signals`: objectness, stability, area ratio, displacement, quality, reasons;
- `rcms_selected`: promoted pre-disappearance anchor frame ids;
- `candidate_count`, `best_candidate_score`: placeholders for later retrieval anchors;
- `commit_decision`: main write / provisional-only / RCMS promotion / empty output;
- `used_cond_frames`: current conditioned-memory frame ids;
- `output_policy` and notes.

### `AnchorBank`

Per-object anchor storage:

- `init_anchor`;
- `pre_disappearance_cond_reservoir`;
- `retrieved_anchors`;
- `negative_bank`.

The first implementation fills only `init_anchor` and `pre_disappearance_cond_reservoir`. Retrieved anchors and negatives are reserved for Ablation C.

### `StateMachine`

First version states:

- `stable`: normal SAM2 path; high-quality frames may enter the RCMS-lite reservoir;
- `ambiguous`: current output may be yielded, but the frame is provisional and does not enter main non-conditioning memory;
- `recovery`: disappearance/low-quality state; pre-disappearance anchors are promoted into conditioned memory.

### `CommitPolicy`

First version delayed commit:

- `stable` + high quality -> write main non-conditioning memory;
- `ambiguous` -> provisional only;
- `recovery` -> promote RCMS-lite anchors, no direct current-frame main memory write.

Later retrieved anchors must pass confirmation for multiple frames before becoming long-term anchors.

### `Selector`

Rule-based only. No weighted global score:

- accept main path;
- keep uncertain branch;
- trigger retrieval;
- output empty;
- request crop refinement.

## Ablation sequence

### Ablation A: RCMS-lite standalone

Command mode: `--rar-mode rcms`.

Purpose: test whether pre-disappearance conditioned reservoir + first-frame anchor preservation helps recovery without changing the normal SAM2 non-conditioning memory path.

Expected evidence:

- RCMS anchor insertions visible in audit;
- no provided-output changes when building a submission;
- key videos do not become more unstable.

### Ablation B: RCMS-lite + state machine + delayed commit

Command mode: default `--rar-mode state`.

Purpose: stop ambiguous/recovery frames from polluting main SAM2 memory, while still allowing provisional outputs for visual inspection.

Expected evidence:

- `commit_decision` separates `write_main_memory`, `provisional_only`, and `promote_rcms_anchors`;
- state timeline shows transition around empty streak/objectness/stability collapse;
- no catastrophic drift introduced on key videos.

### Ablation C: retrieval anchors

Not implemented in this scaffold.

Purpose: global/later-frame candidate mining and object-level matching. This is the decisive recovery stage for same-class distractors.

Target videos:

- `r13u5z4y`
- `q0sizv6m`
- `msinig6m`
- `1qlssuz2`
- `2smf7uq9`
- `8jsm23a7`

Candidate sources may include SAM3 detector, automatic mask proposals, searched windows, or existing M3/M4 masks. The key requirement is later-frame object-level candidates, not only current-path scoring.

### Ablation D: tiny-target refinement

Not implemented in this scaffold.

Purpose: use tiny-crop only after the target identity/location is already selected, as high-resolution local refinement.

Target videos:

- `lcgc29va`
- `4vznweiu`

## Visualization requirements

RAR experiments must produce two visual families before any final submission claim:

1. **Summary sheet per key video**: RGB frames, state timeline, anchor insertions, retrieval moments, commit moments.
2. **Recovery zoom sheet**: base path failure, candidate/retrieval anchor, verification signal, final output.

If these figures show only emptier/more conservative output rather than real recovery, the ablation is failed.

## Submission invariants

1. The 418 provided outputs must remain path-identical and byte-identical. Use `tools/validate_mose_submission.py` for full submission runs.
2. A new method cannot enter final if it creates a new catastrophic drift on key videos, even if a few frames look better.

## Current implementation status

Implemented locally in this commit:

- RAR primitive structures in `src/cvmose/reanchor.py`.
- SAM2 RAR entrypoint in `tools/infer_mosev2_sam2_rar.py`.
- b101 launcher in `scripts/run_b101_rar.sh`.
- Local syntax validation via `python3 -m py_compile` and `bash -n`.
- b101 15-video RCMS/state smoke evidence and key-video visual comparison sheets.

Not yet done:

- retrieval-anchor mining / Ablation C;
- tiny refinement after recovered anchor;
- final-quality improvement evidence;
- final zip candidate.

# Training-free Re-anchor Tracker Direction

Date: 2026-05-30
Status: next primary research direction after M11/M2/M3/M4

## One-line conclusion

The bottleneck is no longer threshold tuning; the system needs a new evidence path that can **recover instance identity after occlusion**, not only suppress wrong SAM2 outputs.

## Why the current methods hit the same wall

All current attempts constrain SAM2 failure at different layers, but none creates a sufficiently independent identity-reacquisition signal.

| method | intervention layer | what it does well | why it saturates |
|---|---|---|---|
| M11 cycle/tracklet gate | output / tracklet postprocessing | rejects suspicious post-gap tracklets and keeps baseline mostly intact | cannot write better history or recover a lost identity chain; it is a safer stop-gap, not a recovery mechanism |
| M2 memory gate | SAM2 memory-write control | prevents unreliable masks from entering non-conditioning memory | reliability/geometry stability is not identity; stable wrong same-class objects can become the new reference |
| M3 state selector | multi-source audit and output choice | makes candidate decisions explicit and auditable | candidate table improves inspection but does not by itself generate new correct evidence; conservative/balanced become over-empty or weak-switch |
| M4 tiny crop | proposal generation at higher relative resolution | helps diagnose token-resolution limits and can tighten small masks | crop solves “cannot see clearly,” not “which instance is this”; after occlusion it can follow correlated wrong guide/velocity evidence |

Therefore local score gains are hard because the system mostly performs **delete / skip / choose among existing masks**. The missing piece is **re-acquire / verify / then commit**.

## Correct intervention level

The next method should upgrade the pipeline from single-path propagation into a closed loop:

```text
first-frame GT anchor
        │
        ▼
SAM2 propagation ──► uncertainty detector ──► candidate retrieval/proposal
        │                       │                       │
        │                       ▼                       ▼
        │              hold / delay output        crop/SAM3/M2/M11 proposals
        │                                               │
        └──────────── identity verifier ◄───────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       reject / keep empty          accept re-anchor
                                          │
                                          ▼
                         controlled SAM2 re-prompt / memory update
```

Key distinction: proposals are not proof. A crop mask, SAM3 adapter mask, or M2/M11 continuation can only become a new anchor after an identity verifier accepts it.

## Required components

### 1. Uncertainty manager

Detect when propagation should stop being trusted:

- abrupt empty/non-empty transitions;
- post-occlusion reappearance;
- large motion or scale discontinuity;
- same-class/distractor proximity;
- tiny/edge target leaving the reliable field of view;
- disagreement among baseline, M2-light, M11, SAM3 adapter, and crop proposal.

Output states should remain explicit: `confirmed`, `uncertain`, `likely_occluded`, `candidate_reappearing`, `reanchored`.

### 2. Candidate retrieval / proposal

Generate possible target masks only when needed:

- baseline SAM2 continuation;
- M11-preserved/rejected tracklets;
- tiny-crop rerun proposals;
- SAM3.1 adapter proposals;
- local search around motion prior;
- possibly reverse-time propagation from a later high-confidence candidate.

The proposal set may be larger than final output, because verification is separate.

### 3. Identity verifier

The verifier must be less correlated with the proposal generator than another area/center threshold. Candidate signals:

- cycle-to-first-frame consistency;
- first-frame exemplar similarity on masked crop features;
- negative-bank similarity against known distractors / other ids;
- bidirectional consistency across a short segment;
- objectness/presence stability over a confirmation delay;
- cross-id competition if multiple objects coexist.

Minimum rule: a post-gap candidate should not become a memory anchor from single-source agreement alone.

### 4. Delayed commit

Do not immediately write a reappearing candidate into SAM2 memory or final submission as a confirmed object. Use staged commitment:

1. output-only or hold-empty while evidence accumulates;
2. require confirmation across several frames or bidirectional checks;
3. only then add the candidate as a pseudo anchor / memory update;
4. if later evidence contradicts it, roll back to empty or baseline-preserving output.

This addresses the core failure where a stable wrong object becomes the new reference.

## External method support

The source-backed study in `docs/external_method_study.md` confirms this direction from three independent lines: SAM2Long shows training-free branch search can reduce greedy memory accumulation; MOSEv2/RCMS shows high-quality pre-disappearance memories should be promoted into conditioned references at the moment of disappearance; and OAMVOS/Re-Prompting/SAM3-DMS show that uncertainty states, delayed promotion, object retrieval, and per-object memory governance are central for MOSEv2-style identity recovery.

## What to avoid

- More global threshold sweeps on M11/M2 without a new identity signal.
- Treating tiny-crop output as independent evidence; it often shares the same wrong guide.
- Treating SAM3/SAM3.1 public-simple output as primary; it solves concept discovery more than first-mask instance identity.
- Weighted final-score fusion where a wrong correlated source can overwhelm identity uncertainty.

## RAR implementation convergence

The accepted mainline is now **RAR: Reappearance-Aware ReAnchor**. Phase A/B should be implemented first as RCMS-lite + state machine + delayed commit in `tools/infer_mosev2_sam2_rar.py` and `src/cvmose/reanchor.py`. See `docs/m5r_rar_plan.md` for the locked ablation order, objects, visual evidence requirements, and submission invariants.

## Proposed next experiment: M5R re-anchor tracker smoke

A minimal next experiment should not be full 15-video first. It should target the hard cases that exposed the wall:

- `r13u5z4y`: strawberry, full hand occlusion, same-class reappearance;
- `8jsm23a7`: small tile/card-like object after occlusion;
- `lcgc29va`: tiny person/backpack in crowd;
- `q0sizv6m` or `msinig6m`: multi-instance same-class competition.

M5R should consume existing sources rather than rerun everything blindly:

1. detect uncertain/post-gap windows from M11/M2-light/M3 audits;
2. generate crop/SAM3/SAM2 local proposals only inside those windows;
3. verify candidates with cycle/exemplar/negative-bank checks;
4. produce contact sheets showing RGB, baseline, proposal, verifier verdict, and final output;
5. only package a full submission if the smoke shows actual recovery, not just cleaner empties.

## Success criterion

The method is worth full-run only if it demonstrates at least one true positive re-acquisition where previous methods either emptied the target or switched to a distractor, while not making `r13u5z4y`-style same-class reappearance worse.

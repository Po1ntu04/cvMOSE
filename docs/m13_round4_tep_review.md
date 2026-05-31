# M13 Round 4 TEP review and next structural implications

Source reviewed: `/home/yu/projects/cv/from fdu/MOSEv2/homework/1st-5th-MOSE.pdf` (`artifacts/m13_tep_1st5th.txt`).

## Relevant TEP claims

TEP's winning structure is not a threshold sweep. It is a three-stage prompt-enhancement system:

1. **Target classification**: route targets into regular / tiny / semantic-dominated using area and MLLM.
2. **Tracking enhancement**:
   - tiny targets get an external tracker bbox prompt;
   - semantic-dominated targets get MLLM-generated frame bboxes.
3. **Prompt fusion**: compare base-model bbox and auxiliary bbox using IoU/confidence/MLLM judge, then inject the better prompt.

The report's most relevant evidence is that the largest gains occur on reappearance (`J&F_reappear`), exactly where our `8jsm23a7`, `q0sizv6m`, and `r13u5z4y` failures sit.

## What M13 now matches

- Qwen3.6 split harness gives event classification and prompt plans without long-panel timeouts.
- `tools/infer_mosev2_sam2_teach_boxes.py` can inject MLLM boxes into SAM2 and now supports reverse propagation for late anchors.
- `tools/apply_m13_object_fusion.py` implements policy-level prompt fusion at object/frame range granularity.

## Remaining gap vs TEP

- We still lack a reliable external image-prompt tracker for tiny targets (TEP used SUTrack). Direct SAM2 box prompts are not enough for low-contrast/tiny cases like `z6dx46qr`.
- Qwen boxes are semantically useful but geometrically coarse. They must be treated as event anchors or bbox priors, not final masks.
- Same-class animal scenes (`q0sizv6m`) still need a hard-negative identity check before injection; otherwise backward propagation amplifies a plausible but wrong/composite anchor.

## Concrete next steps if hidden feedback is positive/negative

- If `m13_8rev_safe` improves: extend reverse-anchor windows to other semantic reappearance cases with visual proof only.
- If it regresses: keep Qwen event stories for routing, but require independent bbox source (SAM3 detector / DINO tracker / SUTrack-like image-prompt tracker) before injection.
- If `m13_8rev_zofficial_balanced` improves more than safe: fold official-large/long intervals into the conservative fusion table for low-contrast targets, but do not use direct MLLM boxes on low-contrast masks.
- If all M13 zips are flat: prioritize implementing an external image-prompt tracker candidate source over additional MLLM prompt engineering.

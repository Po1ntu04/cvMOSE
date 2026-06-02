# M9 Qwen-box recovery smoke

## Conclusion

M9 proved the new external-evidence route is executable, but the current Qwen coordinate-box variant is **not yet a final-quality improvement**: Qwen is useful as an absence/risk verifier, while its direct normalized boxes are too coarse for same-class/tiny recovery and SAM2 box prompts often leak into composite masks unless clipped.

## What was implemented

- `tools/mllm_recovery_box_proposals.py`
  - renders standardized REF/current-frame coordinate panels;
  - calls Qwen-VL through the cached `QwenVLClient`;
  - emits high-recall normalized boxes, not masks.
- `tools/apply_m9_box_proposals_sam2.py`
  - runs SAM2 `SAM2ImagePredictor` with Qwen boxes on b101 GPU;
  - can clip SAM2 masks back to the prompt box to suppress leakage;
  - writes rank-1/2/3 candidate roots from an M11 fallback root.
- `tools/build_m9_video_ablation_zips.py`
  - produces one-video/object-at-a-time submission zips for hidden-score isolation.
- `tools/apply_m8_candidate_fusion.py`
  - added opt-in `--allow-component-candidates` for reconstructing audited connected-component candidates; default remains safe/rejecting components.
- `scripts/run_b101_m9_qwen_box_sam2.sh`
  - reproducible b101 entry point for Qwen-box → SAM2 candidate-root generation.

## Evidence and paths

### Real Qwen proposal smoke

- Proposal JSON: `artifacts/m9_qwen_boxes/proposals_key_smoke.json` (local artifact, not committed)
- Summary: `docs/m9_qwen_box_proposals_smoke.md`
- Panels: `docs/assets/m9_qwen_boxes/proposal_panels/`
- Model actually used by records: mostly `qwen-vl-max`, one `qwen-vl-plus`; no API key is stored in repo.

Key observations:

| video/object | frame(s) | Qwen behavior | verdict |
| --- | ---: | --- | --- |
| `r13u5z4y/1` | 20,24,32 | Correctly says frame 20/32 target absent; frame 24 gives low-confidence composite strawberry cluster. | Good veto/absence signal; poor recovery box. |
| `q0sizv6m/1` | 23 | Returns multiple plausible guinea-pig boxes with same-class risk. | High recall but identity-unsafe. |
| `q0sizv6m/2` | 23 | Returns uncertain edge/background candidates. | Useful uncertainty, not promotion-ready. |
| `msinig6m/1` | 70 | Correctly says target not visible. | Good safety signal. |
| `lcgc29va/1` | 8 | Returns tiny low-confidence boxes; target interpretation remains ambiguous. | Not reliable enough for final. |

### SAM2 box-prompt roots generated on b101

Remote b101 was reachable and used successfully:

- Remote code root: `/data1/yuzhixiang/cv_mosev2/cvMOSE`
- Remote workspace: `/data1/yuzhixiang/cv_mosev2/MOSEv2`
- Conda env: `mose_sam2`
- GPU evidence: CUDA available, RTX 4090 visible.

Unclipped roots:

- `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m9_qwen_box_rank1`
- `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m9_qwen_box_rank2`
- `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m9_qwen_box_rank3`
- Audit: `/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m9_qwen_boxes/m9_box_sam2_rank_audit.json`
- Visuals: `docs/assets/m9_qwen_boxes/compare_box_roots/`

Clipped roots:

- `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m9_qwen_box_clip_rank1`
- `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m9_qwen_box_clip_rank2`
- `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m9_qwen_box_clip_rank3`
- Audit: `/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m9_qwen_boxes/m9_box_sam2_clip_rank_audit.json`
- Visuals: `docs/assets/m9_qwen_boxes/compare_box_clip_roots/`

Clipping reduced severe SAM2 leakage but did not create a trustworthy reappearance anchor. Example audit snippets:

| candidate | unclipped issue | clipped result | judgement |
| --- | --- | --- | --- |
| `r13u5z4y f24 rank1` | full left-side strawberry/hand composite, area 117319 | still large crop/component, area 29307 | reject; same-class composite |
| `q0sizv6m obj1 f23 rank2/3` | large composite guinea-pig regions | localized but still distractor-prone | probe only |
| `lcgc29va f8 rank1` | huge station-person mask | area 189 after clipping | still not visually reliable target recovery |

### Component probe

The M8 candidate pool found two high-scoring `q0sizv6m obj2 f23` connected components from clipped rank roots, but they are not safe: they improve localization shape relative to raw Qwen/SAM2 boxes while still being same-class ambiguous.

- Component probe root: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m9_box_clip_component_probe`
- Audit: `/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m9_qwen_boxes/m9_box_clip_component_probe.json`
- Visual: `docs/assets/m9_qwen_boxes/compare_component_probe/q0sizv6m_m6_compare.jpg`

## Single-object submission probes

All zips below validated with `tools/validate_mose_submission.py` and preserve the 418 provided outputs. They are intended for score diagnosis, not as recommended final submissions unless hidden feedback proves otherwise.

| zip | purpose | visual recommendation |
| --- | --- | --- |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_lcgc_pool.zip` | prior M8 pool-only tiny probe | score probe only |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_lcgc_dino.zip` | prior DINO reanchor tiny probe | score probe only |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_oneql_dino.zip` | prior DINO single-video probe | score probe only |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_twosmf_dino.zip` | prior DINO single-video probe | score probe only |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_fourv_dino.zip` | prior DINO single-video probe | score probe only |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_q0_obj1_dino.zip` | prior DINO q0 obj1 probe | score probe only |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_q0_obj2_dino.zip` | prior DINO q0 obj2 probe | score probe only |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_q0_obj2_component.zip` | new Qwen-box/SAM2 clipped component probe | risky; same-class ambiguity |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_lcgc_qwenclip1.zip` | new Qwen-box clipped tiny probe | visually weak |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_q0_obj1_qwenclip1.zip` | new Qwen-box clipped q0 obj1 probe | risky |
| `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m9_ablation_q0_obj2_qwenclip2.zip` | new Qwen-box clipped q0 obj2 probe | risky |

## Why this did not yet solve the 44-point bottleneck

The M9 smoke isolates the remaining bottleneck more sharply:

1. **MLLM semantic reasoning is stronger than MLLM coordinate localization.** Qwen often correctly says “absent / same-class risk / composite,” but its normalized boxes are not precise enough for direct SAM2 prompt promotion.
2. **SAM2 box prompts are not identity prompts.** A loose or ambiguous box in clutter makes SAM2 segment the dominant region inside/near the box; without a true object-level detector or verified crop candidate, this creates composite masks.
3. **Clipping controls leakage but not identity.** Intersecting with the prompt box converts catastrophic masks into smaller candidates, yet it cannot decide which guinea pig/strawberry/person instance is the reference.
4. **The current score plateau is therefore not a zip/fusion bug.** The system still lacks a high-recall, object-precise candidate source for hard reappearance frames.

## Next direction

Do not expand raw Qwen coordinate-box prompting. The next implementation should keep Qwen as a verifier/router and replace its coordinate source with an object-proposal source:

1. Generate many precise mask/box proposals in recovery frames using SAM2 automatic masks / detector proposals / object proposal models.
2. Render proposal crops to Qwen as labeled candidates, not ask Qwen to output coordinates.
3. Use DINO/SAM2 descriptors plus hard-negative margin to filter.
4. Only then call SAM2 `add_new_mask` or bounded repropagation.

This is more likely to move toward 44 than more direct MLLM coordinate calls.

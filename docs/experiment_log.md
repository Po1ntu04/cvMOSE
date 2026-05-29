# Experiment Log

## 2026-05-29 — M1 visibility/identity gate

- Branch: `method/m1-visibility-gate`
- Code commits: `6420d39` initial M1, `47db029` experiment-safety hardening, `0532fe8` corrupt-mask guard
- Method file: `tools/apply_visibility_gate.py`
- Launcher: `scripts/run_b101_m1_visibility_gate.sh`
- Baseline input: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_b101`
- Output masks: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m1_visibility`
- Submission dir: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/submission_433_m1_visibility`
- Submission zip: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/submission_mosev2_m1_visibility.zip`
- Audit JSON: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m1_visibility_gate_latest.json`

### Hypothesis

For MOSEv2 first-frame-mask tracking, an empty mask after severe occlusion is often less damaging than confidently switching to a wrong same-class instance. M1 therefore leaves SAM2 unchanged except when a mask reappears after an invisible gap with identity-risk signals:

- large centroid jump relative to target scale and image diagonal;
- far post-gap reappearance;
- abrupt area ratio change;
- severe fragmentation or implausibly large frame coverage;
- low RGB-histogram consistency when the target is large enough for that signal to be meaningful.

The first-frame GT mask is preserved exactly. The method is a reversible post-processing gate, not a model change.

### b101 execution

Command:

```bash
./scripts/sync_code_b101.sh
./scripts/run_b101_m1_visibility_gate.sh
```

Result:

- Videos processed: `15`
- Frames processed: `1004`
- Suppressed object-frames: `167`
- Gate runtime: `117.025 s` on latest validated rerun (`75.265 s` on first packaged run)
- Submission zip size: `71,211,856 bytes` (`68M`)
- Zip validation: `433` directories, `66,526` PNGs, `testzip_bad=None`
- Visual review: `docs/m1_visual_review.md`
- Code review: `docs/m1_code_review.md`
- GPU memory: not applicable; M1 is CPU post-processing over existing SAM2 predictions.

Per-video suppressions:

| video | object summary |
| --- | --- |
| `2smf7uq9` | id1 raw=23, gated=8, suppressed=15, span=28-42 |
| `4f98052b` | id1 raw=59, gated=58, suppressed=1, span=50; id2 raw=85, gated=10, suppressed=75, span=48-178 |
| `c8lutf29` | id1 raw=13, gated=2, suppressed=11, span=17-27 |
| `msinig6m` | id1 raw=49, gated=35, suppressed=14, span=50-74; id3 raw=40, gated=33, suppressed=7, span=93-99 |
| `pe0d85lk` | id1 raw=21, gated=2, suppressed=19, span=19-43 |
| `q0sizv6m` | id2 raw=32, gated=30, suppressed=2, span=13-14 |
| `r13u5z4y` | id1 raw=29, gated=6, suppressed=23, span=21-44 |

### Interpretation before Codabench

M1 is intentionally identity-protective and likely helps the canonical `r13u5z4y` strawberry occlusion/same-class-switch case. However, visual review found likely false-empty regressions in `c8lutf29` and `pe0d85lk`, where the target appears semantically visible but M1 suppresses it due to large post-gap movement. `4f98052b` id2 is also a high-risk long suppression span.

Decision: keep this zip as an ablation candidate only. Do not promote M1 as the default SAM2 replacement. The next useful step is M1.1: a narrower, visually gated identity-switch intervention before M2 pseudo-anchor re-prompting.

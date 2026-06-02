# M6 Phase 1 — Official MOSEv2 Checkpoint / Submission Ceiling

Date: 2026-05-30  
Branch: `method/m6_external_reid_ensemble`  
Code commit used on b101: `ff2ebf3d2fd8ebb0996883da357fb79dc5f8f277` plus local Phase-1 wrapper edits before the next commit.

## Purpose

This phase checks whether a strong ceiling comes from the public FudanCVL MOSEv2 baseline assets rather than from local 15-video-only post-hoc engineering. It is **diagnostic-first**: because these checkpoints are MOSEv2-specific / fine-tuned assets and the full submission zips are public benchmark outputs, final-submission eligibility depends on the homework rules. Under the strictest training-free interpretation, treat them as ceiling/diagnostic sources, not as the default final.

Sources:

- HF model repo: <https://huggingface.co/FudanCVL/MOSEv2_baseline/tree/main>
- SAM2 codebase used for inference: <https://github.com/facebookresearch/sam2>

## Download result and file hashes

Remote b101 could not reach HF directly (`Network is unreachable`), so files were downloaded locally and rsynced to:

```text
/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/external_checkpoints/FudanCVL_MOSEv2_baseline/
/home/yu/projects/cv/from fdu/MOSEv2/homework/external_checkpoints/FudanCVL_MOSEv2_baseline/
```

| file | size | sha256 |
|---|---:|---|
| `sam2.1_hiera_b+_MOSEv2_mss_lvt16.pt` | 323,651,479 | `4a559627b2a94fbb5b35316a57f1b366980ab14c07b72c965b06f61efd2b6ada` |
| `sam2.1_hiera_l_MOSEv2_mss_lvt16.pt` | 898,153,772 | `03fb442d2349679e20c1ae0fd5ddacf142fdac4796c17d23162c04589d17ab53` |
| `sam2_b+_MOSEv2_rcms_mqf_mss_lvt_submission.zip` | 79,011,033 | `dcca00f30f5d961218b3c7b4d340dbbfaf6c30e26ba751e576e4cb21a0902c0f` |
| `sam2_l_MOSEv2_rcms_mqf_mss_lvt_submission.zip` | 67,471,731 | `bc4bf899d9a4b7f2c1a5d070817aec7fe53883deee57983f24fd3e3b2d179f3a` |

## Commands

Checkpoint inference:

```bash
DOWNLOAD=0 INSTALL_HF_HUB=0 VARIANTS='bplus large' RUN_INFER=1 RUN_EXTRACT=1 \
  GPU=4 scripts/run_b101_official_ckpt.sh
```

After discovering that public full-submission zips did not preserve our local first-frame GT byte-exactly, extraction mode was fixed to overwrite `00000.png` from `homework/Annotations`, then rerun without checkpoint inference:

```bash
DOWNLOAD=0 INSTALL_HF_HUB=0 VARIANTS='bplus large' RUN_INFER=0 RUN_EXTRACT=1 \
  GPU=4 scripts/run_b101_official_ckpt.sh
```

Local compare/visual generation:

```bash
conda run -n cv-hw2 python tools/compare_candidate_roots.py \
  --workspace '/home/yu/projects/cv/from fdu/MOSEv2' \
  --baseline-root '/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_b101' \
  --m11-root artifacts/m5r_reanchor/source_preds/m11 \
  --roots baseline=... m11=... official_bplus=... official_large=... official_sub_bplus=... official_sub_large=... \
  --output-json artifacts/m6_phase1/compare_official.json \
  --output-csv artifacts/m6_phase1/compare_official.csv

conda run -n cv-hw2 python scripts/make_m6_compare.py \
  --workspace '/home/yu/projects/cv/from fdu/MOSEv2' \
  --baseline-root '/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_b101' \
  --m11-root artifacts/m5r_reanchor/source_preds/m11 \
  --candidate-roots official_bplus=... official_large=... official_sub_large=... \
  --out-dir docs/assets/m6_official/compare \
  --frames-per-video 8
```

## Candidate outputs and validation

| candidate | pred root | zip | audit / validation | validator |
|---|---|---|---|---|
| official b+ checkpoint | `/data1/.../pred_sam2_official_bplus_mosev2` | `/data1/.../submission_mosev2_official_bplus_15only.zip` | `/data1/.../logs/m6_official/validate_bplus.json` | ok |
| official large checkpoint | `/data1/.../pred_sam2_official_large_mosev2` | `/data1/.../submission_mosev2_official_large_15only.zip` | `/data1/.../logs/m6_official/validate_large.json` | ok |
| official b+ full-submission 15-only extract | `/data1/.../pred_official_submission_bplus_15only` | `/data1/.../submission_mosev2_official_submission_bplus_15only.zip` | `/data1/.../logs/m6_official/validate_submission_bplus.json` | ok |
| official large full-submission 15-only extract | `/data1/.../pred_official_submission_large_15only` | `/data1/.../submission_mosev2_official_submission_large_15only.zip` | `/data1/.../logs/m6_official/validate_submission_large.json` | ok |

All four zips contain 433 video directories / 66,526 PNGs, keep all 418 provided outputs unchanged, and now preserve the local first-frame GT for all 15 target videos.

Local copies of the zips are under:

```text
/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_official_*.zip
```

Visual sheets:

```text
docs/assets/m6_official/compare/{video}_m6_compare.jpg
```

## Format/agreement summary

| root | changed frames vs SAM2 | first-frame failures | mean per-video binary IoU vs SAM2 |
|---|---:|---:|---:|
| M11 | 60 | 0 | 0.950019 |
| official b+ checkpoint | 682 | 0 | 0.677876 |
| official large checkpoint | 726 | 0 | 0.715358 |
| official b+ submission extract | 700 | 0 | 0.679607 |
| official large submission extract | 708 | 0 | 0.674272 |

Interpretation: official candidates are much more different than M11. That is expected for a fine-tuned/RCMS/MQF/MSS/LVT system, but it also means they are not safe to fuse globally without visual proof.

## Per-video qualitative verdict

| video | visual verdict | recommended safe action |
|---|---|---|
| `r13u5z4y` | All official variants switch to empty after the occlusion/reappearance; this avoids the wrong strawberry that SAM2 follows, but still does not recover the intended slice. | Safe can use M11/official-empty only; do not treat as true recovery. |
| `q0sizv6m` | Checkpoint variants mostly behave like SAM2/M11 and continue same-class animal drift; official-sub-large becomes empty for object 2 later, safer but not a confident re-id. | Not safe as improvement except possible conservative emptying. |
| `msinig6m` | Official variants do not solve identity; large/sub-large introduce different koala/person/composite errors in some frames. | Reject for safe/balanced unless a narrow object-level segment is manually proven later. |
| `lcgc29va` | Official large/sub-large have partial tiny-target recall but still miss multiple frames; b+ loses the target around the hard interval. | Possible tiny diagnostic only; not a confident replacement. |
| `1qlssuz2` | Checkpoint variants are close to SAM2; official-sub-large shrinks/losses the car near the end. | Keep SAM2/M11 for safe. |
| `8jsm23a7` | Large/sub-large are visually close to SAM2; b+ agreement metric is low despite similar-looking thumbnails and should not be trusted blindly. | Keep M11/SAM2; stable guard applies. |
| `2smf7uq9` | b+ close to SAM2, large/sub-large more empty; no clear visual recovery. | Keep M11/SAM2. |
| `4vznweiu` | Very large divergence and tiny/edge uncertainty; no clear visual proof of improvement. | Keep baseline/M11. |
| `4f98052b` | Many empty frames; no visual proof that official variants improve both objects. | Keep baseline/M11 unless later per-object proof. |
| `3epdtmyr` | High IoU with SAM2 but many empty frames; likely not a re-id gain. | Keep baseline/M11. |
| `amfdu83t` | Mostly empty after early frames across all methods; no clear improvement. | Keep current safest. |
| `c8lutf29` | Mostly empty; no meaningful recovery. | Keep current safest. |
| `jadgtmfl` | large/sub-large less empty than b+, but still no documented target-level gain. | Candidate for later visual review, not automatic. |
| `pe0d85lk` | Checkpoint variants close to baseline; sub-large differs more. | Keep baseline/M11. |
| `z6dx46qr` | Low agreement and no visible robust target gain. | Keep baseline/M11. |

## Recommendation

- **As ceiling/diagnostic:** keep all four official outputs. They prove that stronger MOSEv2-specific systems can behave very differently from local SAM2/M11, but they still fail or choose emptiness on the hardest reappearance/same-class cases we care about.
- **As final submission:** only allowed if the assignment explicitly permits public MOSEv2-finetuned checkpoints or public official submission outputs. Otherwise, do **not** use them as final; use them as candidate roots for conservative per-video fusion/diagnostics.
- **For M6 fusion:** the official candidates can contribute only where visual evidence is strictly better and no stable-video regression appears. Default remains M11/SAM2.

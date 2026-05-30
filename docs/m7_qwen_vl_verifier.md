# M7 Qwen-VL verifier for M5R-C

**One-line verdict:** M7 is now implemented as a cached, conservative MLLM verification layer for M5R-C; this commit does not claim score gain yet because all local validation was dry-run/panel generation only, intentionally avoiding real API spend until the API key/model choice is confirmed.

## Why M7 exists

M5R-C already has the right structural loop:

```text
high-recall candidates -> object descriptor / hard negatives -> delayed anchor -> SAM2 add_new_mask repropagation
```

The current bottleneck is identity verification, not mask decoding.  In the hardest MOSEv2 cases (`r13u5z4y`, `q0sizv6m`, `msinig6m`, `lcgc29va`) geometric quality and DINO/SAM2 descriptors can still confuse same-class instances, tiny crops, or person/animal composites.  M7 adds Qwen-VL as a **verifier only**:

- target profiler;
- single-frame candidate judge;
- multi-frame tracklet / delayed-promotion judge;
- high-risk anchor veto/support signal.

It is explicitly **not** a mask source and cannot promote anchors by itself.

## API contract implemented

Official Aliyun Bailian / DashScope docs confirm:

- OpenAI-compatible SDK base URL for Beijing: `https://dashscope.aliyuncs.com/compatible-mode/v1`.
- Python OpenAI SDK calls use `OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=...)`.
- Vision input uses chat message content items with `type: "image_url"`; the image URL field can be a remote URL or a Base64 Data URL.

References:

- https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions
- https://help.aliyun.com/zh/model-studio/qwen-vl-compatible-with-openai
- https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope

Implementation:

- `src/cvmose/qwen_vl_client.py`
  - default `BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`;
  - reads `DASHSCOPE_API_KEY` only from environment;
  - default model `qwen-vl-max-latest`;
  - fallback models `qwen-vl-max,qwen2.5-vl-72b-instruct,qwen-vl-plus`;
  - local images are resized to max side 1600 and encoded as JPEG Base64 Data URLs;
  - strict JSON parsing with conservative parse-error fallback;
  - cache key = model + schema + prompt + image byte hashes;
  - cache directory: `artifacts/m7_qwen_vl/cache/`;
  - dry-run returns conservative JSON stubs and writes cache records.

## Panel design

Implemented in `src/cvmose/mllm_panels.py`.

### A. Target Profile Panel

One panel per video/object:

- frame-0 RGB + yellow bbox;
- frame-0 GT mask overlay;
- target crop raw;
- target crop + mask overlay;
- extra zoom when tiny.

### B. Candidate Judge Panel

One panel per high-risk video/object/frame:

- REF crop from frame 0;
- optional pre-gap context placeholder;
- current wide context with candidate letters;
- each candidate crop as raw | overlay;
- explicit `EMPTY / TARGET ABSENT OPTION` tile.

### C. Tracklet Judge Panel

One panel per candidate tracklet:

- REF crop;
- candidate raw | overlay for 2-3 consecutive frames;
- wide context around the anchor frame.

## Tools added

| file | role |
| --- | --- |
| `tools/test_qwen_vl_bailian.py` | smoke-test Bailian/OpenAI-compatible JSON vision call; writes setup doc |
| `tools/mllm_target_profile.py` | render profile panels and produce per-object routing profiles |
| `tools/mllm_candidate_judge.py` | render high-risk candidate panels and cache candidate judgments |
| `tools/mllm_tracklet_judge.py` | render delayed-promotion tracklet panels and cache tracklet judgments |
| `scripts/make_m7_qwen_compare.py` | compare SAM2/M11/M5R/M7 roots when real M7 prediction roots exist |

## M5R-C integration

`tools/infer_mosev2_sam2_reanchor.py` now accepts:

```text
--target-profiles-json
--mllm-candidate-judgments-json
--mllm-tracklet-judgments-json
--mllm-policy off|veto_only|support_and_veto|semantic_support
--mllm-min-veto-confidence 0.55
--mllm-min-support-confidence 0.70
--mllm-require-tracklet-for-anchor
--mllm-uncertain-action keep_original|empty|output_only
```

Policy behavior:

- `off`: original M5R-C.
- `veto_only`: Qwen can reject high-risk anchors if confidence crosses the veto threshold; it cannot add anchors.
- `support_and_veto`: Qwen support can slightly boost already-valid descriptor candidates, but promotion still requires descriptor/temporal evidence and optional tracklet confirmation.
- `semantic_support`: same as support/veto only for target profiles marked semantic-dominated; otherwise falls back to veto-only behavior.

Audit fields added to each candidate:

```text
mllm_policy
mllm_candidate_decision
mllm_tracklet_decision
mllm_veto_applied
mllm_support_applied
mllm_confidence
mllm_risk_tags
mllm_cache_key
```

`run_b101_m5r_reanchor.sh` allowlists the new MLLM flags for safe remote execution.

## Dry-run evidence collected

Commands run locally with `conda run -n cv-hw2`:

```bash
python tools/test_qwen_vl_bailian.py --dry-run
python tools/mllm_target_profile.py \
  --workspace '/home/yu/projects/cv/from fdu/MOSEv2' \
  --videos r13u5z4y q0sizv6m msinig6m 1qlssuz2 2smf7uq9 8jsm23a7 lcgc29va 4vznweiu 4f98052b c8lutf29 pe0d85lk \
  --dry-run
python tools/mllm_candidate_judge.py \
  --workspace '/home/yu/projects/cv/from fdu/MOSEv2' \
  --target-profiles-json artifacts/m7_qwen_vl/target_profiles.json \
  --m5r-audit-json artifacts/m5r_reanchor/m5r_reanchor_key.json \
  --pred-roots 'baseline=/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_b101,m11=artifacts/m5r_reanchor/source_preds/m11,m5r=artifacts/m5r_reanchor/key_pred,dino_m5r=/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m5r_dino_key,rar_state=artifacts/m5r_reanchor/source_preds/rar_state' \
  --videos r13u5z4y q0sizv6m msinig6m 1qlssuz2 2smf7uq9 8jsm23a7 lcgc29va 4vznweiu 4f98052b c8lutf29 pe0d85lk \
  --dry-run --max-calls 80
python tools/mllm_tracklet_judge.py \
  --workspace '/home/yu/projects/cv/from fdu/MOSEv2' \
  --target-profiles-json artifacts/m7_qwen_vl/target_profiles.json \
  --candidate-judgments-json artifacts/m7_qwen_vl/candidate_judgments.json \
  --dry-run --max-calls 40
python -m py_compile src/cvmose/qwen_vl_client.py src/cvmose/mllm_panels.py \
  tools/test_qwen_vl_bailian.py tools/mllm_target_profile.py \
  tools/mllm_candidate_judge.py tools/mllm_tracklet_judge.py \
  tools/infer_mosev2_sam2_reanchor.py scripts/make_m7_qwen_compare.py
bash -n scripts/run_b101_m5r_reanchor.sh
```

Generated docs:

- `docs/m7_qwen_vl_setup.md`
- `docs/m7_target_profile_summary.md`
- `docs/m7_candidate_judge_summary.md`
- `docs/m7_tracklet_judge_summary.md`

Generated local artifacts (ignored by git):

- `artifacts/m7_qwen_vl/profile_panels/`
- `artifacts/m7_qwen_vl/candidate_panels/`
- `artifacts/m7_qwen_vl/tracklet_panels/`
- `artifacts/m7_qwen_vl/cache/`
- `artifacts/m7_qwen_vl/target_profiles.json`
- `artifacts/m7_qwen_vl/candidate_judgments.json`
- `artifacts/m7_qwen_vl/tracklet_judgments.json`

Dry-run counts:

- target profiles: 16 objects;
- candidate judgments: 80 high-risk panels;
- tracklet judgments: 40 panels;
- all dry-run judgments are conservative: no support, no promotion.

## Current route table from dry-run heuristic profiles

| route | videos / objects | intended policy |
| --- | --- | --- |
| same-class dense | `r13u5z4y`, `q0sizv6m`, `msinig6m`, `2smf7uq9`, `8jsm23a7` | primarily `veto_only`; do not accept single-frame same-class matches |
| tiny | `1qlssuz2`, `lcgc29va`, `4vznweiu` | require zoom/crop evidence; Qwen cannot promote alone |
| semantic-dominated | `4f98052b`, `c8lutf29`, `pe0d85lk` and parts of `4vznweiu`/`8jsm23a7` | allow support only after descriptor/tracklet confirmation |

These are dry-run heuristics, not semantic Qwen conclusions.

## What to run once API/model is confirmed

Real API smoke:

```bash
python tools/test_qwen_vl_bailian.py
```

Then regenerate the three MLLM JSON files **without** `--dry-run`.  Keep `--max-calls` bounded.  If parse failures exceed 30% or Qwen confidently picks wrong same-class candidates on `r13u5z4y/q0sizv6m/msinig6m`, stop and keep M7 as audit-only.

Remote M5R-C smoke after real judgments:

```bash
VIDEOS='r13u5z4y q0sizv6m msinig6m 1qlssuz2 2smf7uq9 8jsm23a7 lcgc29va 4vznweiu 4f98052b c8lutf29 pe0d85lk' \
MAKE_SUBMISSION=0 GPU=${GPU:-4} \
PRED_ROOT='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m7_qwen_veto_key' \
AUDIT_JSON='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m7_qwen_veto_key.json' \
AUDIT_DIR='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m7_qwen_veto_key_by_video' \
M5R_EXTRA_ARGS='--mllm-policy veto_only --target-profiles-json artifacts/m7_qwen_vl/target_profiles.json --mllm-candidate-judgments-json artifacts/m7_qwen_vl/candidate_judgments.json --mllm-tracklet-judgments-json artifacts/m7_qwen_vl/tracklet_judgments.json --mllm-min-veto-confidence 0.55 --mllm-require-tracklet-for-anchor' \
 scripts/run_b101_m5r_reanchor.sh
```

Support mode should be run only after candidate/tracklet real judgments show useful non-erroneous support on semantic-dominated objects.

## Submission decision status

No M7 submission zip was generated in this implementation pass because:

1. the user said the API key/model would be provided later;
2. all judgments here were forced dry-run;
3. dry-run stubs are designed to be conservative audit scaffolding, not final prediction evidence.

M7 should produce final/balanced zips only after real Qwen calls and visual review show:

- at least three key videos are visually better;
- no wrong-strawberry recovery on `r13u5z4y`;
- no large wrong same-class/composite masks on `q0sizv6m`/`msinig6m`;
- no regression on stable videos such as `8jsm23a7`;
- 418 provided outputs unchanged via `tools/validate_mose_submission.py`.

## Real API update: qwen3.5-plus q0 smoke

After the API key/model were provided, I ran a real OpenAI-compatible Bailian smoke using `QWEN_VL_MODEL=qwen3.5-plus`.

Real API evidence:

- smoke test: `status=ok`, `model_used=qwen3.5-plus`, elapsed about 7-12 seconds depending on cache/image sizing;
- target profiles: 16/16 objects returned `status=ok`;
- q0sizv6m candidate judge subset: 8 real candidate calls;
- q0sizv6m b101 M5R-veto smoke v1 exposed an implementation issue: judged anchors were rejected, but unjudged high-risk same-class anchors could still be selected;
- fixed policy: with `--mllm-require-tracklet-for-anchor`, same-class/tiny/edge high-risk candidates without an MLLM judgment are rejected unless their descriptor margin is very strong.

q0sizv6m real Qwen candidate outcome:

| frame | best | veto | conf | short reading |
| ---: | --- | --- | ---: | --- |
| 34 | uncertain | false | 0.25 | dense identical guinea-pig cluster; cannot confirm instance |
| 17 | none | true | 0.90 | candidate is same-class distractor |
| 36 | uncertain | false | 0.25 | cannot verify specific identity |
| 35 | uncertain | false | 0.30 | dense crowd prevents re-ID |
| 40 | uncertain | false | 0.55 | one candidate color/position plausible but not enough to promote |
| 16 | none | true | 0.10 | same-class neighbor, not the reference instance |

b101 q0 smoke after the high-risk-unjudged fix:

```text
pred_root=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m7_qwen_veto_q0_real_v2
audit_json=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m7_qwen_veto_q0_real_v2.json
accepted_anchor_count=0
changed_vs_baseline=0
```

Interpretation:

- Qwen3.5-plus is useful as a **safety verifier** on q0: it refuses to certify same-class guinea-pig anchors and explicitly identifies some wrong-neighbor candidates.
- It does **not** yet provide recovery evidence on q0; the safe outcome is baseline/M11-equivalent rather than a positive improvement.
- This supports keeping `veto_only` for same-class dense cases and reserving `support_and_veto` for semantic-dominated objects only after real support judgments are available.

Visual sheet:

- `docs/assets/m7_qwen_vl/q0_real_v2_compare/q0sizv6m_m7_qwen_compare.jpg`

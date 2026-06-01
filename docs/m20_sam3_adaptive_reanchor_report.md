# M20 SAM3 adaptive re-anchor report

## Goal

Implement the planned **SAM3-assisted, rollback-safe re-anchor layer** without
making SAM3 the default tracker.  The base remains the strongest local
SAM2/M17 result:

```text
/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m17_q0_full_box
```

SAM3 is treated as a candidate/proof source only.  A candidate must pass the
M18-style descriptor/negative-bank/story checks before it can affect a new
prediction root.

## Implemented pieces

- `src/cvmose/sam3_m20.py`
  - `PromptPack`: category text, semantic text, positive first-frame visual
    prompt, negative visual prompts, and story constraints.
  - `M20Sam3Adapter.sam3_image_propose(...)`: live SAM3.1 proposal generation
    through the official `handle_request(add_prompt)` path, with replay support
    for precomputed SAM3 roots.
  - `M20Sam3Adapter.sam3_video_branch(...)`: bounded branch audit/rollback
    record.  Live multi-anchor bounded propagation remains an explicit upstream
    branch step; replay mode records the available `pred_sam31_b101` branch.
  - `sam3_audit_export()`: prompt/candidate/branch audit export.

- `tools/build_m20_sam3_adaptive_reanchor.py`
  - builds prompt packs for the five M20 targets:
    `8jsm23a7`, `4vznweiu`, `1qlssuz2`, `r13u5z4y`, `q0sizv6m:obj2`;
  - replays `pred_sam31_b101` as the SAM3 branch when live CUDA SAM3 is not
    available;
  - scores SAM3 candidates with first-frame positive memory, M18 hard-negative
    boxes, temporal story compatibility, area risk, temporal votes, and
    source-independent agreement;
  - writes safe/balanced/aggressive prediction roots and submission zips with
    audit JSON, source tables, candidate cards, and validator output.

## Runtime note

This local session has no CUDA SAM3 runtime (`cv-hw2` is CPU-only), and a quick
batch SSH check to `b101` timed out.  Therefore the completed run used replay
mode over the already available b101 SAM3.1 GT-mask branch:

```text
/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam31_b101
```

The live code path is still present behind `--run-live-sam3` for a CUDA b101
rerun.

## Candidate retrieval result

Artifact: `artifacts/m20_sam3_adaptive/sam3_candidate_cards.json`

- SAM3 replay cards scored: `18`
- M20 decisions:
  - `reject`: `14`
  - `needs_more_evidence`: `4`
  - `promote`: `0`

Interpretation: the available SAM3.1 GT-mask branch is useful as a negative
audit/control but did **not** produce any independently verified re-anchor.  In
particular, same-class/dense targets were blocked by hard-negative similarity,
low margin, absent-window story constraints, or lack of source-independent
confirmation.

## Generated profiles

All three zips passed `tools/validate_mose_submission.py` with:

- `433` video dirs;
- `66526` PNGs;
- `418` provided videos unchanged (`provided_changed_count=0`);
- `15` predicted videos structurally valid.

| profile | zip | changed frames vs M17 base | policy |
| --- | --- | ---: | --- |
| safe | `submission_mosev2_m20_sam3_safe.zip` | `0` | Exact M17 full-box base; no SAM3 candidate had enough evidence to replace. |
| balanced | `submission_mosev2_m20_sam3_balanced.zip` | `45` | Story-verified bounded pseudo-anchors: `8jsm23a7` frames `6-48`, `1qlssuz2` frames `13-14`; SAM3 replay remains audit/proof only. |
| aggressive | `submission_mosev2_m20_sam3_aggressive.zip` | `70` | Probe-only broader story windows: `8jsm23a7` frames `2-48`, `4vznweiu` frames `14-34`, `1qlssuz2` frames `13-14`. |

The safe zip is the default non-regression candidate.  Balanced/aggressive are
score probes, not claims that SAM3 alone found a correct instance.

## Key artifacts

```text
artifacts/m20_sam3_adaptive/
├── prompt_packs.json
├── sam3_candidate_cards.json
├── sam3_adapter_audit.json
├── audit_safe.json
├── audit_balanced.json
├── audit_aggressive.json
├── source_table_safe.csv
├── source_table_balanced.csv
├── source_table_aggressive.csv
├── compare_m20_candidates.json
├── compare_m20_candidates.csv
├── validate_safe.json
├── validate_balanced.json
├── validate_aggressive.json
└── run_summary.json
```

## Verification commands

```bash
conda run -n cv-hw2 python -m py_compile \
  src/cvmose/sam3_m20.py \
  tools/build_m20_sam3_adaptive_reanchor.py

conda run -n cv-hw2 python tools/build_m20_sam3_adaptive_reanchor.py --overwrite

conda run -n cv-hw2 python tools/compare_candidate_roots.py \
  --workspace "/home/yu/projects/cv/from fdu/MOSEv2" \
  --baseline-root "/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m17_q0_full_box" \
  --m11-root "/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m15_layered_safe" \
  --roots \
    m20_safe=/home/yu/projects/cv/from\ fdu/MOSEv2/homework/pred_m20_sam3_safe \
    m20_balanced=/home/yu/projects/cv/from\ fdu/MOSEv2/homework/pred_m20_sam3_balanced \
    m20_aggressive=/home/yu/projects/cv/from\ fdu/MOSEv2/homework/pred_m20_sam3_aggressive \
    sam31=/home/yu/projects/cv/from\ fdu/MOSEv2/homework/pred_sam31_b101 \
  --videos 8jsm23a7 4vznweiu 1qlssuz2 r13u5z4y q0sizv6m \
  --output-json artifacts/m20_sam3_adaptive/compare_m20_candidates.json \
  --output-csv artifacts/m20_sam3_adaptive/compare_m20_candidates.csv

for mode in safe balanced aggressive; do
  conda run -n cv-hw2 python tools/validate_mose_submission.py \
    --workspace "/home/yu/projects/cv/from fdu/MOSEv2" \
    --zip-path "/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m20_sam3_${mode}.zip" \
    --output-json "artifacts/m20_sam3_adaptive/validate_${mode}.json"
done
```

## Failure mode classification

For this local replay run, the dominant failure class is:

1. **SAM3 proposal / branch did not recall a verified target** (`promote=0`).
2. Some proposals were present but the verifier rejected or downgraded them
   because they matched hard negatives, lacked independent confirmation, or
   appeared inside absent/occluded story windows.

No tracker-injection drift was accepted into final M20 outputs because all SAM3
branch outputs remained isolated until verification.

## Review / adversarial confirmation update (2026-06-01)

Independent review found the first M20 implementation was **not merge-ready** as a SAM3 method: balanced/aggressive outputs were still driven by M19 story roots, while SAM3 candidate cards were only audited.  The live path also incorrectly reused first-frame GT boxes as later-frame image prompts, even though SAM3 boxes are frame-local.

Fixes applied:

- Final M20 profiles are now **SAM3-candidate gated by default**:
  - `safe`: only `promote` candidates;
  - `balanced`: `promote` / `output_only` candidates;
  - `aggressive`: also allows `needs_more_evidence` probes.
- Legacy M19 story-root probes are excluded unless `--include-story-probes` is explicitly passed.
- Live SAM3 proposal ablation now tries prompt variants and filters visual boxes to the same frame; frame-0 GT boxes are no longer sent as geometry on later frames.
- Candidate masks are materialized under the artifact directory, and final source tables carry `candidate_id` provenance.
- Zip validation now fails closed before reporting/copying; existing zips are not silently overwritten without `--overwrite`.

Remote b101 verification after fixes:

```text
artifact: artifacts/m20_sam3_strict_remote/
base: /data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_m17_q0_full_box
SAM3 replay source: pred_sam31_b101
cards: 18 = 14 reject + 4 needs_more_evidence + 0 promote/output_only
safe: 0 edits, validator ok
balanced: 0 edits, validator ok
aggressive: 3 frame-level SAM3-gated probes on 8jsm23a7, validator ok
```

Local copied zips:

```text
/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m20_sam3_strict_safe.zip
/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m20_sam3_strict_balanced.zip
/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m20_sam3_strict_aggressive.zip
```

Live SAM3 smoke on b101:

- `4vznweiu:1`, one sampled frame: 6 live semantic candidates, all rejected by identity/negative-bank checks.
- `8jsm23a7:1`, two sampled frames: 12 live candidates, 1 `output_only`, 10 `needs_more_evidence`, 1 rejected.  The accepted `output_only` candidate overlaps current M17 strongly at frame 48, so it is not new recovery evidence.

Adversarial conclusion: after fixing provenance, M20 currently provides **safety and audit value but no proven SAM3-driven score-improving candidate**.  The next SAM3 work should target true later-frame proposal recall and multi-anchor propagation rather than reusing replay roots or story masks.

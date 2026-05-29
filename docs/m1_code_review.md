# M1 Code Review

Date: 2026-05-29  
Branch: `method/m1-visibility-gate`  
Scope: `tools/apply_visibility_gate.py`, `scripts/run_b101_m1_visibility_gate.sh`, submission packaging behavior

## Independent review lanes

- Code-reviewer lane: **REQUEST CHANGES**
- Architect / devil's-advocate lane: **BLOCK**
- Final synthesis: **REQUEST CHANGES** for merge/default-method approval

This does not mean the M1 experiment artifact is useless. It means M1 should remain an ablation/probe until the method-level risks are resolved.

## Fixed after review

| issue | status | commit |
| --- | --- | --- |
| Unsafe output deletion if `out_pred_root` overlaps raw/JPEG/annotation roots | fixed by protected-root validation | `47db029` |
| Unknown raw label IDs could pass through | fixed by raw-label subset validation | `47db029` |
| Runner inferred dry-run from `$*` substring | fixed by explicit remote `DRY_RUN_FLAG` | `47db029` |
| Submission wrapper did not check final zip integrity | fixed by post-build `433` dirs / `66526` PNGs / `testzip` validation | `47db029` |
| Unused `--copy-raw-first` flag | removed | `47db029` |
| Corrupt-mask signals could suppress with no gap or corroboration | guarded by requiring prior gap or multiple anomaly signals | `0532fe8` |

Validation after fixes:

- `python3 -m py_compile tools/apply_visibility_gate.py`
- `bash -n scripts/run_b101_m1_visibility_gate.sh`
- b101 dry-run after safety fixes: `1004` frames, `167` suppressions
- b101 non-dry after safety fixes: generated `433` video dirs, `66,526` PNGs, validated zip size `71,211,856` bytes
- b101 dry-run after corrupt-mask guard: `1004` frames, `167` suppressions

## Remaining blockers

### 1. Suppression latch / no recovery path

When a candidate is suppressed, `last_confirmed` is not updated and the invisible gap grows. If the true target reappears far away because of camera motion or long occlusion, it can be repeatedly compared against a stale pre-occlusion anchor and remain empty for a long span.

This is visible in M1 audit spans such as:

- `4f98052b` id2: 75 suppressed frames;
- `r13u5z4y` id1: 23 suppressed frames;
- `pe0d85lk` id1: 19 suppressed frames.

For `r13u5z4y` this may be desirable identity protection. For `pe0d85lk` it is likely a false-empty regression. The current code has no mechanism to distinguish the two.

### 2. No camera-motion compensation

Centroid distance is measured in image coordinates. In high camera-motion scenes (`pe0d85lk`, `4f98052b`) a true same object can move far in image space even if identity is preserved.

### 3. No semantic identity confirmation

M1 uses area, centroid, RGB histogram, and fragmentation. These are useful audit signals but weak evidence for “same first-frame instance,” especially with:

- same-class objects (`r13u5z4y`, `2smf7uq9`, `q0sizv6m`);
- repeated textures/chairs (`4f98052b`);
- moving people near edges (`c8lutf29`, `pe0d85lk`).

### 4. No cross-id / hard-negative reasoning

Multi-object videos need competition constraints. M1 evaluates each object independently and cannot detect two ids converging on the same instance, or use likely distractors as negative evidence.

## Final review decision

M1 is code-safe enough to keep as a reproducible experiment artifact after `47db029` and `0532fe8`, but it is **not method-ready** as a default submission path.

Recommendation: do **M1.1** before M2. M1.1 should be narrower and semantic-risk-aware rather than a global post-gap suppression rule.

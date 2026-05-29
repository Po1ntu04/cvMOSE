# Experiment Protocol

## 1. Versioning

Every experiment must have:

- branch or commit id;
- method id (`M1`, `M2`, ...);
- affected videos;
- remote command;
- output path;
- validation evidence;
- Codabench score if submitted.

Suggested branch names:

- `method/m1-visibility-gate`
- `method/m2-multi-anchor`
- `exp/r13u5z4y-reanchor-v1`

## 2. Commit message protocol

Use decision-record style commits:

```text
<why this change was made>

Constraint: <external constraint>
Rejected: <alternative> | <reason>
Confidence: <low|medium|high>
Scope-risk: <narrow|moderate|broad>
Directive: <warning for future changes>
Tested: <what was verified>
Not-tested: <known gaps>
```

## 3. Remote b101 policy

Remote is for code execution, not for storing analysis artifacts.

Sync only:

- `tools/`
- `scripts/`
- `src/`
- `configs/`
- `pyproject.toml`
- `README.md`

Do not rsync:

- `docs/`
- contact sheets / target zooms;
- extracted paper text;
- local analysis images;
- data, checkpoints, submissions, logs unless explicitly needed as experiment outputs.

## 4. Parallel inference policy

Use concurrency to reduce wall time, but keep outputs disjoint by video.

Priority:

1. multi-GPU partition if multiple GPUs are intentionally exposed;
2. multiple workers on one GPU only when memory headroom is known;
3. write to one shared `pred_root` only when video partitions are disjoint;
4. build submission once after all workers complete.

For SAM2 on RTX 4090, prior memory use was about 1 GiB allocated, so multiple workers may be feasible. For SAM3.1 adapter, prior peak was about 9.5 GiB reserved, so concurrency must be more conservative.

## 5. Experiment log template

```markdown
## YYYY-MM-DD HH:MM — <experiment id>

- Commit:
- Method:
- Hypothesis:
- Videos:
- Command:
- Output:
- Runtime / GPU:
- Qualitative evidence:
- Quantitative evidence:
- Decision:
- Next:
```

# M7 candidate judge summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.5-plus`
- API key present: `True`
- records: `8`
- support/veto/uncertain: `6/1/1`

| video | obj | frame | candidates | best | support | veto | conf | reason | panel |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| lcgc29va | 1 | 8 | 1 | A | True | False | 0.90 | Candidate A matches REF instance appearance (dark jacket) and local spatial cont | `artifacts/m7_qwen_vl/candidate_panels_outline_lcgc/lcgc29va/obj1_f00008_candidate_judge.jpg` |
| lcgc29va | 1 | 9 | 1 | A | True | False | 0.85 | Candidate A preserves the unique spatial relationship with the neighbor in the r | `artifacts/m7_qwen_vl/candidate_panels_outline_lcgc/lcgc29va/obj1_f00009_candidate_judge.jpg` |
| lcgc29va | 1 | 4 | 2 | none | False | True | 0.90 | Candidates A/B mask a person in a red jacket, while the REF target (blue outline | `artifacts/m7_qwen_vl/candidate_panels_outline_lcgc/lcgc29va/obj1_f00004_candidate_judge.jpg` |
| lcgc29va | 1 | 12 | 2 | uncertain | False | False | 0.40 | Candidate B tracks red jacket (distractor, REF is dark jacket). Candidate A trac | `artifacts/m7_qwen_vl/candidate_panels_outline_lcgc/lcgc29va/obj1_f00012_candidate_judge.jpg` |
| lcgc29va | 1 | 1 | 4 | A | True | False | 0.85 | Candidates A, B, D show high consensus on location and mask extent (upper body), | `artifacts/m7_qwen_vl/candidate_panels_outline_lcgc/lcgc29va/obj1_f00001_candidate_judge.jpg` |
| lcgc29va | 1 | 2 | 3 | A | True | False | 0.85 | Candidate A matches REF location and appearance (person in red at turnstile). Al | `artifacts/m7_qwen_vl/candidate_panels_outline_lcgc/lcgc29va/obj1_f00002_candidate_judge.jpg` |
| lcgc29va | 1 | 3 | 3 | C | True | False | 0.85 | All candidates consistently mask the same person's head located immediately left | `artifacts/m7_qwen_vl/candidate_panels_outline_lcgc/lcgc29va/obj1_f00003_candidate_judge.jpg` |
| lcgc29va | 1 | 5 | 3 | C | True | False | 0.70 | Visual features (red scarf, dark coat, profile pose) match REF. Candidate C prov | `artifacts/m7_qwen_vl/candidate_panels_outline_lcgc/lcgc29va/obj1_f00005_candidate_judge.jpg` |

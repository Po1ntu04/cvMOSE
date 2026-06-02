# M7 candidate judge summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.5-plus`
- API key present: `True`
- records: `8`
- support/veto/uncertain: `0/4/6`

| video | obj | frame | candidates | best | support | veto | conf | reason | panel |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| q0sizv6m | 1 | 34 | 2 | uncertain | False | False | 0.25 | Dense cluster of visually identical guinea pigs makes instance re-identification | `artifacts/m7_qwen_vl/candidate_panels_q0_real/q0sizv6m/obj1_f00034_candidate_judge.jpg` |
| q0sizv6m | 1 | 17 | 1 | none | False | True | 0.90 | Candidate A masks a different guinea pig (same class distractor) than the instan | `artifacts/m7_qwen_vl/candidate_panels_q0_real/q0sizv6m/obj1_f00017_candidate_judge.jpg` |
| q0sizv6m | 1 | 36 | 2 | uncertain | False | False | 0.25 | Dense cluster of visually identical guinea pigs; impossible to verify specific i | `artifacts/m7_qwen_vl/candidate_panels_q0_real/q0sizv6m/obj1_f00036_candidate_judge.jpg` |
| q0sizv6m | 1 | 35 | 2 | uncertain | False | False | 0.30 | Dense crowd of visually identical guinea pigs makes instance re-identification f | `artifacts/m7_qwen_vl/candidate_panels_q0_real/q0sizv6m/obj1_f00035_candidate_judge.jpg` |
| q0sizv6m | 1 | 40 | 3 | uncertain | False | False | 0.55 | Candidates A/B are noise (tiny area on ground). Candidate C matches target color | `artifacts/m7_qwen_vl/candidate_panels_q0_real/q0sizv6m/obj1_f00040_candidate_judge.jpg` |
| q0sizv6m | 1 | 10 | 1 | uncertain | False | True | 0.00 | all models failed | `artifacts/m7_qwen_vl/candidate_panels_q0_real/q0sizv6m/obj1_f00010_candidate_judge.jpg` |
| q0sizv6m | 1 | 16 | 1 | none | False | True | 0.10 | Candidate A masks a same-class neighbor (tan guinea pig on right edge near pole) | `artifacts/m7_qwen_vl/candidate_panels_q0_real/q0sizv6m/obj1_f00016_candidate_judge.jpg` |
| q0sizv6m | 1 | 11 | 2 | uncertain | False | True | 0.00 | all models failed | `artifacts/m7_qwen_vl/candidate_panels_q0_real/q0sizv6m/obj1_f00011_candidate_judge.jpg` |

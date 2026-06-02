# M7 candidate judge summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.5-plus`
- API key present: `True`
- records: `16`
- support/veto/uncertain: `12/4/3`

| video | obj | frame | candidates | best | support | veto | conf | reason | panel |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| 4vznweiu | 1 | 16 | 5 | B | True | False | 0.85 | Candidate B matches the green 'T' block in shape, color, and position relative t | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/4vznweiu/obj1_f00016_candidate_judge.jpg` |
| 4vznweiu | 1 | 8 | 4 | A | True | False | 0.95 | All candidates consistently identify the unique green die with letter 'T', match | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/4vznweiu/obj1_f00008_candidate_judge.jpg` |
| 4vznweiu | 1 | 9 | 4 | A | True | False | 0.95 | Target is the unique green die with 'T'. All candidates consistently segment thi | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/4vznweiu/obj1_f00009_candidate_judge.jpg` |
| 4vznweiu | 1 | 10 | 4 | A | True | False | 0.95 | All candidates consistently segment the unique green 'T' die, which matches the  | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/4vznweiu/obj1_f00010_candidate_judge.jpg` |
| 4vznweiu | 1 | 15 | 4 | empty | False | True | 0.95 | REF target is a green die with letter 'T'. All candidates (B, C, D) segment whit | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/4vznweiu/obj1_f00015_candidate_judge.jpg` |
| 4vznweiu | 1 | 11 | 5 | A | True | False | 0.95 | Target (green 'T' die) is clearly visible and uniquely distinguished by color fr | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/4vznweiu/obj1_f00011_candidate_judge.jpg` |
| lcgc29va | 1 | 8 | 1 | A | True | False | 0.90 | Candidate A shows a green backpack matching REF color and shape. Crucially, the  | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/lcgc29va/obj1_f00008_candidate_judge.jpg` |
| lcgc29va | 1 | 9 | 1 | uncertain | False | True | 0.25 | Candidate A shows a green backpack on a person positioned to the RIGHT of a red- | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/lcgc29va/obj1_f00009_candidate_judge.jpg` |
| lcgc29va | 1 | 4 | 2 | A | True | False | 0.75 | Distinctive green backpack visible in crop matches REF appearance and location c | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/lcgc29va/obj1_f00004_candidate_judge.jpg` |
| lcgc29va | 1 | 12 | 2 | B | True | True | 0.80 | Candidate B matches the specific instance features (green backpack on dark jacke | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/lcgc29va/obj1_f00012_candidate_judge.jpg` |
| lcgc29va | 1 | 1 | 4 | A | True | False | 0.85 | Distinctive green backpack clearly visible; strong consensus among A, B, D on ti | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/lcgc29va/obj1_f00001_candidate_judge.jpg` |
| lcgc29va | 1 | 2 | 3 | A | True | False | 0.85 | Distinctive green backpack feature matches REF perfectly. All candidates agree o | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/lcgc29va/obj1_f00002_candidate_judge.jpg` |
| lcgc29va | 1 | 3 | 3 | uncertain | False | False | 0.45 | Target is tiny and not clearly visible in crop; candidates show similar small gr | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/lcgc29va/obj1_f00003_candidate_judge.jpg` |
| lcgc29va | 1 | 5 | 3 | C | True | True | 0.85 | Target (green backpack) is clearly visible and matches REF color/shape. Candidat | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/lcgc29va/obj1_f00005_candidate_judge.jpg` |
| 8jsm23a7 | 1 | 6 | 4 | uncertain | False | False | 0.45 | REF shows a small green tile being held; candidates show similar tiles but no cl | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/8jsm23a7/obj1_f00006_candidate_judge.jpg` |
| 8jsm23a7 | 1 | 4 | 5 | B | True | False | 0.85 | Candidate B isolates the specific single green tile interacted with by the hand, | `artifacts/m7_qwen_vl/candidate_panels_real_tiny_semantic/8jsm23a7/obj1_f00004_candidate_judge.jpg` |

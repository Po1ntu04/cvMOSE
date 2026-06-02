# M7 candidate judge summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.5-plus`
- API key present: `True`
- records: `16`
- support/veto/uncertain: `14/2/1`

| video | obj | frame | candidates | best | support | veto | conf | reason | panel |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| 4vznweiu | 1 | 16 | 5 | B | True | False | 0.85 | Candidate B (and E) localize a green die with 'T' visible on top, matching REF i | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/4vznweiu/obj1_f00016_candidate_judge.jpg` |
| 4vznweiu | 1 | 8 | 4 | A | True | False | 0.95 | All candidates (A/B/C/D) consistently identify the same green die with 'T' as in | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/4vznweiu/obj1_f00008_candidate_judge.jpg` |
| 4vznweiu | 1 | 9 | 4 | A | True | False | 0.95 | Target (green 'T' die) is clearly visible and matches REF features (unique green | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/4vznweiu/obj1_f00009_candidate_judge.jpg` |
| 4vznweiu | 1 | 10 | 4 | A | True | False | 0.95 | All candidates (A/B/C/D) consistently identify the unique green 'T' die, which m | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/4vznweiu/obj1_f00010_candidate_judge.jpg` |
| 4vznweiu | 1 | 15 | 4 | empty | False | True | 0.80 | REF target is a green die with letter 'T'. Candidates B and D segment a green di | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/4vznweiu/obj1_f00015_candidate_judge.jpg` |
| 4vznweiu | 1 | 11 | 5 | A | True | False | 0.95 | Target (green die with 'T') is clearly visible and distinct by color from surrou | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/4vznweiu/obj1_f00011_candidate_judge.jpg` |
| lcgc29va | 1 | 8 | 1 | A | True | False | 0.85 | Distinctive green backpack matches REF color and shape. Spatial relationship wit | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/lcgc29va/obj1_f00008_candidate_judge.jpg` |
| lcgc29va | 1 | 9 | 1 | A | True | False | 0.72 | Target is small but spatial relationship with distinct red-jacket companion matc | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/lcgc29va/obj1_f00009_candidate_judge.jpg` |
| lcgc29va | 1 | 4 | 2 | uncertain | False | False | 0.30 | Target is tiny (area ~400px); visual features (backpack) not clearly distinguish | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/lcgc29va/obj1_f00004_candidate_judge.jpg` |
| lcgc29va | 1 | 12 | 2 | B | True | True | 0.75 | Candidate B shows a person with a green backpack (matching REF) standing next to | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/lcgc29va/obj1_f00012_candidate_judge.jpg` |
| lcgc29va | 1 | 1 | 4 | A | True | False | 0.90 | All candidates consistently identify the person with the distinct green backpack | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/lcgc29va/obj1_f00001_candidate_judge.jpg` |
| lcgc29va | 1 | 2 | 3 | A | True | False | 0.90 | Target (person with distinctive green backpack) is clearly visible and matches R | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/lcgc29va/obj1_f00002_candidate_judge.jpg` |
| lcgc29va | 1 | 3 | 3 | A | True | False | 0.85 | Target (person with distinctive green backpack) is clearly visible in candidate  | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/lcgc29va/obj1_f00003_candidate_judge.jpg` |
| lcgc29va | 1 | 5 | 3 | C | True | False | 0.75 | Target is the person with backpack seen in REF, now further away behind pillars. | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/lcgc29va/obj1_f00005_candidate_judge.jpg` |
| 8jsm23a7 | 1 | 6 | 4 | A | True | False | 0.75 | Target is a single Mahjong tile being handled. Candidate A localizes a single ti | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/8jsm23a7/obj1_f00006_candidate_judge.jpg` |
| 8jsm23a7 | 1 | 4 | 5 | A | True | False | 0.85 | REF shows a green tile back held by hand. Candidates A-D correctly identify a gr | `artifacts/m7_qwen_vl/candidate_panels_overlayfix_tiny/8jsm23a7/obj1_f00004_candidate_judge.jpg` |

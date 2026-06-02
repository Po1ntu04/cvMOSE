# M7 candidate judge summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.6-plus`
- API key present: `True`
- records: `13`
- support/veto/uncertain: `7/6/6`

| video | obj | frame | candidates | best | support | veto | conf | reason | panel |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| r13u5z4y | 1 | 24 | 2 | uncertain | False | True | 0.40 | Target strawberry has been sliced into multiple indistinguishable pieces. Candid | `artifacts/m10_qwen36/candidate_panels_key/r13u5z4y/obj1_f00024_candidate_judge.jpg` |
| q0sizv6m | 1 | 23 | 3 | uncertain | False | True | 0.35 | Dense cluster of visually identical guinea pigs; Candidate A matches class/appea | `artifacts/m10_qwen36/candidate_panels_key/q0sizv6m/obj1_f00023_candidate_judge.jpg` |
| q0sizv6m | 2 | 23 | 5 | A | True | False | 0.85 | Candidate A matches the reference guinea pig (black head, light body) in positio | `artifacts/m10_qwen36/candidate_panels_key/q0sizv6m/obj2_f00023_candidate_judge.jpg` |
| msinig6m | 1 | 70 | 1 | A | True | False | 0.95 | Candidate A precisely matches the reference koala's position and unique spatial  | `artifacts/m10_qwen36/candidate_panels_key/msinig6m/obj1_f00070_candidate_judge.jpg` |
| 1qlssuz2 | 1 | 10 | 2 | A | True | False | 0.85 | Candidate A and B both identify the same white car in the current frame, which v | `artifacts/m10_qwen36/candidate_panels_key/1qlssuz2/obj1_f00010_candidate_judge.jpg` |
| 1qlssuz2 | 1 | 34 | 4 | A | True | False | 0.85 | Candidate A (and B/C) shows a white wagon/SUV in the right lane with a guardrail | `artifacts/m10_qwen36/candidate_panels_key/1qlssuz2/obj1_f00034_candidate_judge.jpg` |
| 2smf7uq9 | 1 | 30 | 4 | C | True | False | 0.85 | Target flamingo on right edge is clearly visible and consistent with REF positio | `artifacts/m10_qwen36/candidate_panels_key/2smf7uq9/obj1_f00030_candidate_judge.jpg` |
| 2smf7uq9 | 1 | 31 | 3 | uncertain | False | True | 0.50 | Multiple similar flamingos in dense group; no clear visual evidence linking any  | `artifacts/m10_qwen36/candidate_panels_key/2smf7uq9/obj1_f00031_candidate_judge.jpg` |
| 8jsm23a7 | 1 | 10 | 3 | A | True | False | 0.70 | Candidate A correctly segments the single Mahjong tile being placed, consistent  | `artifacts/m10_qwen36/candidate_panels_key/8jsm23a7/obj1_f00010_candidate_judge.jpg` |
| 8jsm23a7 | 1 | 15 | 3 | uncertain | False | True | 0.50 | Target is a tiny Mahjong tile with highly repetitive texture. REF crop is blurry | `artifacts/m10_qwen36/candidate_panels_key/8jsm23a7/obj1_f00015_candidate_judge.jpg` |
| lcgc29va | 1 | 8 | 2 | uncertain | False | True | 0.30 | REF shows a person in a dark jacket. Candidate B is a pink jacket (color mismatc | `artifacts/m10_qwen36/candidate_panels_key/lcgc29va/obj1_f00008_candidate_judge.jpg` |
| 4vznweiu | 1 | 9 | 3 | A | True | False | 0.90 | Target 'T' block is clearly visible. Candidate A matches REF with consistent spa | `artifacts/m10_qwen36/candidate_panels_key/4vznweiu/obj1_f00009_candidate_judge.jpg` |
| 4vznweiu | 1 | 15 | 3 | uncertain | False | True | 0.30 | REF is a specific 'T' bead. Candidate B selects a 'U' bead (wrong letter). Candi | `artifacts/m10_qwen36/candidate_panels_key/4vznweiu/obj1_f00015_candidate_judge.jpg` |

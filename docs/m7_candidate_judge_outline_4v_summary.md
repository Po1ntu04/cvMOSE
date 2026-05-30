# M7 candidate judge summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.5-plus`
- API key present: `True`
- records: `6`
- support/veto/uncertain: `3/2/1`

| video | obj | frame | candidates | best | support | veto | conf | reason | panel |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| 4vznweiu | 1 | 16 | 5 | uncertain | False | True | 0.30 | REF target is a specific 'T'/'T' die. Candidates B/D/E track 'T'/'O' dice, which | `artifacts/m7_qwen_vl/candidate_panels_outline_4v/4vznweiu/obj1_f00016_candidate_judge.jpg` |
| 4vznweiu | 1 | 8 | 4 | A | True | False | 0.95 | Target is a specific die with 'T' on top and front faces. All candidates correct | `artifacts/m7_qwen_vl/candidate_panels_outline_4v/4vznweiu/obj1_f00008_candidate_judge.jpg` |
| 4vznweiu | 1 | 9 | 4 | A | True | False | 0.95 | Target 'T' tile is clearly visible in all candidate crops. Text 'T' (top/bottom) | `artifacts/m7_qwen_vl/candidate_panels_outline_4v/4vznweiu/obj1_f00009_candidate_judge.jpg` |
| 4vznweiu | 1 | 10 | 4 | A | False | False | 0.65 | Target appearance (T/T die) matches REF, but left neighbor context differs (REF= | `artifacts/m7_qwen_vl/candidate_panels_outline_4v/4vznweiu/obj1_f00010_candidate_judge.jpg` |
| 4vznweiu | 1 | 15 | 4 | empty | False | True | 0.30 | REF is a 'T' bead next to an 'R'. Candidate A masks a 'T' bead next to an 'O' (d | `artifacts/m7_qwen_vl/candidate_panels_outline_4v/4vznweiu/obj1_f00015_candidate_judge.jpg` |
| 4vznweiu | 1 | 11 | 5 | A | True | False | 0.95 | All candidates consistently identify the tile with 'T'/'T' text. Neighbor tiles  | `artifacts/m7_qwen_vl/candidate_panels_outline_4v/4vznweiu/obj1_f00011_candidate_judge.jpg` |

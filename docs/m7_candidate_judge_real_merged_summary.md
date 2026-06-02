# M7 real candidate judgment merged summary

- model: `qwen3.5-plus`
- records: `24`

| video | records | ok | api_error | support | veto | uncertain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1qlssuz2 | 6 | 4 | 2 | 3 | 3 | 2 |
| msinig6m | 5 | 2 | 3 | 0 | 4 | 5 |
| q0sizv6m | 8 | 6 | 2 | 0 | 4 | 6 |
| r13u5z4y | 5 | 2 | 3 | 0 | 5 | 4 |

## Records

| video | obj | frame | best | support | veto | conf | status | reason |
| --- | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| 1qlssuz2 | 1 | 9 | none | False | True | 0.95 | ok | REF is a green car. Both candidates A and B mask a white car adjacent to the green one. Cl |
| 1qlssuz2 | 1 | 10 | A | True | False | 0.90 | ok | Green car instance clearly visible and matches REF color/shape; white car distractor corre |
| 1qlssuz2 | 1 | 16 | uncertain | False | True | 0.00 | api_error | all models failed |
| 1qlssuz2 | 1 | 33 | A | True | False | 0.90 | ok | Green car matches REF color and shape. Candidates A, B, D show strong consensus on locatio |
| 1qlssuz2 | 1 | 34 | A | True | False | 0.95 | ok | Strong consensus (A,B,C,D,F) on green car matching REF color and class. Candidate E is out |
| 1qlssuz2 | 1 | 35 | uncertain | False | True | 0.00 | api_error | all models failed |
| msinig6m | 1 | 25 | uncertain | False | True | 0.00 | api_error | all models failed |
| msinig6m | 1 | 26 | uncertain | False | True | 0.00 | api_error | all models failed |
| msinig6m | 1 | 27 | uncertain | False | False | 0.45 | ok | High risk of same-class distractor (multiple koalas in crowd) and edge_partial (bbox clipp |
| msinig6m | 1 | 28 | uncertain | False | True | 0.00 | api_error | all models failed |
| msinig6m | 1 | 29 | uncertain | False | True | 0.10 | ok | Reference missing (grey box); cannot verify identity in crowd; candidate bbox (width 9px)  |
| q0sizv6m | 1 | 10 | uncertain | False | True | 0.00 | api_error | all models failed |
| q0sizv6m | 1 | 11 | uncertain | False | True | 0.00 | api_error | all models failed |
| q0sizv6m | 1 | 16 | none | False | True | 0.10 | ok | Candidate A masks a same-class neighbor (tan guinea pig on right edge near pole). The true |
| q0sizv6m | 1 | 17 | none | False | True | 0.90 | ok | Candidate A masks a different guinea pig (same class distractor) than the instance indicat |
| q0sizv6m | 1 | 34 | uncertain | False | False | 0.25 | ok | Dense cluster of visually identical guinea pigs makes instance re-identification impossibl |
| q0sizv6m | 1 | 35 | uncertain | False | False | 0.30 | ok | Dense crowd of visually identical guinea pigs makes instance re-identification from Frame  |
| q0sizv6m | 1 | 36 | uncertain | False | False | 0.25 | ok | Dense cluster of visually identical guinea pigs; impossible to verify specific instance id |
| q0sizv6m | 1 | 40 | uncertain | False | False | 0.55 | ok | Candidates A/B are noise (tiny area on ground). Candidate C matches target color (tan) but |
| r13u5z4y | 1 | 1 | uncertain | False | True | 0.00 | api_error | all models failed |
| r13u5z4y | 1 | 2 | uncertain | False | True | 0.00 | api_error | all models failed |
| r13u5z4y | 1 | 3 | uncertain | False | True | 0.35 | ok | REF shows strawberry with green calyx; Candidate shows hulled strawberry without calyx. An |
| r13u5z4y | 1 | 4 | none | False | True | 0.90 | ok | Candidates A/B track a strawberry slice without the distinctive green gummy bear; the actu |
| r13u5z4y | 1 | 21 | uncertain | False | True | 0.00 | api_error | all models failed |

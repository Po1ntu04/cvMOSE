# M7 target profile summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.6-plus`
- API key present: `True`
- dry_run requested: `False`
- objects profiled: `16`
- type counts: `{"edge_partial": 4, "same_class_dense": 7, "semantic_dominated": 1, "tiny": 4}`

| video | obj | type | tiny | edge/partial | semantic | same-class | policy | confidence | panel |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | --- |
| r13u5z4y | 1 | same_class_dense | False | False | True | True | veto_only | 0.85 | `artifacts/m10_qwen36/profile_panels/r13u5z4y/obj1_target_profile.jpg` |
| q0sizv6m | 1 | same_class_dense | False | False | True | True | avoid_reanchor | 0.85 | `artifacts/m10_qwen36/profile_panels/q0sizv6m/obj1_target_profile.jpg` |
| q0sizv6m | 2 | edge_partial | True | True | False | True | veto_only | 0.60 | `artifacts/m10_qwen36/profile_panels/q0sizv6m/obj2_target_profile.jpg` |
| msinig6m | 1 | same_class_dense | False | False | False | True | veto_only | 0.90 | `artifacts/m10_qwen36/profile_panels/msinig6m/obj1_target_profile.jpg` |
| msinig6m | 2 | same_class_dense | False | True | False | True | veto_only | 0.85 | `artifacts/m10_qwen36/profile_panels/msinig6m/obj2_target_profile.jpg` |
| msinig6m | 3 | same_class_dense | False | False | False | True | avoid_reanchor | 0.90 | `artifacts/m10_qwen36/profile_panels/msinig6m/obj3_target_profile.jpg` |
| 1qlssuz2 | 1 | tiny | True | False | False | True | tiny_requires_crop | 0.85 | `artifacts/m10_qwen36/profile_panels/1qlssuz2/obj1_target_profile.jpg` |
| 2smf7uq9 | 1 | same_class_dense | True | False | True | True | avoid_reanchor | 0.85 | `artifacts/m10_qwen36/profile_panels/2smf7uq9/obj1_target_profile.jpg` |
| 8jsm23a7 | 1 | tiny | True | False | False | True | veto_only | 0.75 | `artifacts/m10_qwen36/profile_panels/8jsm23a7/obj1_target_profile.jpg` |
| lcgc29va | 1 | tiny | True | False | False | True | tiny_requires_crop | 0.50 | `artifacts/m10_qwen36/profile_panels/lcgc29va/obj1_target_profile.jpg` |
| 4vznweiu | 1 | tiny | True | False | True | True | veto_only | 0.90 | `artifacts/m10_qwen36/profile_panels/4vznweiu/obj1_target_profile.jpg` |
| 4f98052b | 1 | edge_partial | False | True | False | True | allow_support | 0.90 | `artifacts/m10_qwen36/profile_panels/4f98052b/obj1_target_profile.jpg` |
| 4f98052b | 2 | same_class_dense | False | False | True | True | avoid_reanchor | 0.90 | `artifacts/m10_qwen36/profile_panels/4f98052b/obj2_target_profile.jpg` |
| c8lutf29 | 1 | edge_partial | False | True | False | True | avoid_reanchor | 0.85 | `artifacts/m10_qwen36/profile_panels/c8lutf29/obj1_target_profile.jpg` |
| pe0d85lk | 1 | edge_partial | False | True | False | True | avoid_reanchor | 0.85 | `artifacts/m10_qwen36/profile_panels/pe0d85lk/obj1_target_profile.jpg` |
| pe0d85lk | 2 | semantic_dominated | False | True | True | True | allow_support | 0.90 | `artifacts/m10_qwen36/profile_panels/pe0d85lk/obj2_target_profile.jpg` |

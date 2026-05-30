# M7 target profile summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen-vl-max-latest`
- API key present: `True`
- dry_run requested: `True`
- objects profiled: `16`
- type counts: `{"same_class_dense": 8, "semantic_dominated": 5, "tiny": 3}`

| video | obj | type | tiny | edge/partial | semantic | same-class | policy | confidence | panel |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | --- |
| r13u5z4y | 1 | same_class_dense | False | True | False | True | veto_only | 0.00 | `artifacts/m7_qwen_vl/profile_panels/r13u5z4y/obj1_target_profile.jpg` |
| q0sizv6m | 1 | same_class_dense | False | False | False | True | veto_only | 0.00 | `artifacts/m7_qwen_vl/profile_panels/q0sizv6m/obj1_target_profile.jpg` |
| q0sizv6m | 2 | same_class_dense | True | True | False | True | avoid_reanchor | 0.00 | `artifacts/m7_qwen_vl/profile_panels/q0sizv6m/obj2_target_profile.jpg` |
| msinig6m | 1 | same_class_dense | False | False | False | True | veto_only | 0.00 | `artifacts/m7_qwen_vl/profile_panels/msinig6m/obj1_target_profile.jpg` |
| msinig6m | 2 | same_class_dense | False | True | False | True | veto_only | 0.00 | `artifacts/m7_qwen_vl/profile_panels/msinig6m/obj2_target_profile.jpg` |
| msinig6m | 3 | same_class_dense | False | False | False | True | veto_only | 0.00 | `artifacts/m7_qwen_vl/profile_panels/msinig6m/obj3_target_profile.jpg` |
| 1qlssuz2 | 1 | tiny | True | False | False | False | tiny_requires_crop | 0.00 | `artifacts/m7_qwen_vl/profile_panels/1qlssuz2/obj1_target_profile.jpg` |
| 2smf7uq9 | 1 | same_class_dense | True | False | False | True | avoid_reanchor | 0.00 | `artifacts/m7_qwen_vl/profile_panels/2smf7uq9/obj1_target_profile.jpg` |
| 8jsm23a7 | 1 | same_class_dense | True | False | True | True | avoid_reanchor | 0.00 | `artifacts/m7_qwen_vl/profile_panels/8jsm23a7/obj1_target_profile.jpg` |
| lcgc29va | 1 | tiny | True | True | False | False | tiny_requires_crop | 0.00 | `artifacts/m7_qwen_vl/profile_panels/lcgc29va/obj1_target_profile.jpg` |
| 4vznweiu | 1 | tiny | True | False | True | False | tiny_requires_crop | 0.00 | `artifacts/m7_qwen_vl/profile_panels/4vznweiu/obj1_target_profile.jpg` |
| 4f98052b | 1 | semantic_dominated | False | True | True | False | allow_support | 0.00 | `artifacts/m7_qwen_vl/profile_panels/4f98052b/obj1_target_profile.jpg` |
| 4f98052b | 2 | semantic_dominated | False | False | True | False | allow_support | 0.00 | `artifacts/m7_qwen_vl/profile_panels/4f98052b/obj2_target_profile.jpg` |
| c8lutf29 | 1 | semantic_dominated | False | True | True | False | allow_support | 0.00 | `artifacts/m7_qwen_vl/profile_panels/c8lutf29/obj1_target_profile.jpg` |
| pe0d85lk | 1 | semantic_dominated | False | True | True | False | allow_support | 0.00 | `artifacts/m7_qwen_vl/profile_panels/pe0d85lk/obj1_target_profile.jpg` |
| pe0d85lk | 2 | semantic_dominated | False | True | True | False | allow_support | 0.00 | `artifacts/m7_qwen_vl/profile_panels/pe0d85lk/obj2_target_profile.jpg` |

# M7 target profile summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.5-plus`
- API key present: `True`
- dry_run requested: `False`
- objects profiled: `3`
- type counts: `{"tiny": 3}`

| video | obj | type | tiny | edge/partial | semantic | same-class | policy | confidence | panel |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | --- |
| 4vznweiu | 1 | tiny | True | False | True | True | veto_only | 0.90 | `artifacts/m7_qwen_vl/profile_panels_overlayfix_subset/4vznweiu/obj1_target_profile.jpg` |
| lcgc29va | 1 | tiny | True | False | False | True | tiny_requires_crop | 0.85 | `artifacts/m7_qwen_vl/profile_panels_overlayfix_subset/lcgc29va/obj1_target_profile.jpg` |
| 8jsm23a7 | 1 | tiny | True | False | False | True | avoid_reanchor | 0.75 | `artifacts/m7_qwen_vl/profile_panels_overlayfix_subset/8jsm23a7/obj1_target_profile.jpg` |

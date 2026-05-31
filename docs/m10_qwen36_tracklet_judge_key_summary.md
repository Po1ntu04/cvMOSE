# M7 tracklet judge summary

- workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- model: `qwen3.6-plus`
- API key present: `True`
- records: `10`
- promote_anchor=true: `7`

| video | obj | anchor | source | promote | conf | same-ref | risk | panel |
| --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| r13u5z4y | 1 | 24 | baseline | False | 0.40 | uncertain | same_class_switch,composite | `artifacts/m10_qwen36/tracklet_panels_key/r13u5z4y/obj1_f00024_baseline_tracklet.jpg` |
| q0sizv6m | 1 | 23 | dam | False | 0.35 | uncertain | same_class_switch,occlusion | `artifacts/m10_qwen36/tracklet_panels_key/q0sizv6m/obj1_f00023_dam_tracklet.jpg` |
| q0sizv6m | 2 | 23 | baseline | True | 0.95 | yes |  | `artifacts/m10_qwen36/tracklet_panels_key/q0sizv6m/obj2_f00023_baseline_tracklet.jpg` |
| msinig6m | 1 | 70 | baseline | True | 0.95 | yes |  | `artifacts/m10_qwen36/tracklet_panels_key/msinig6m/obj1_f00070_baseline_tracklet.jpg` |
| 1qlssuz2 | 1 | 10 | baseline | True | 0.85 | yes | tiny_unreadable | `artifacts/m10_qwen36/tracklet_panels_key/1qlssuz2/obj1_f00010_baseline_tracklet.jpg` |
| 1qlssuz2 | 1 | 34 | baseline | True | 0.90 | yes |  | `artifacts/m10_qwen36/tracklet_panels_key/1qlssuz2/obj1_f00034_baseline_tracklet.jpg` |
| 2smf7uq9 | 1 | 30 | official_large | True | 0.92 | yes |  | `artifacts/m10_qwen36/tracklet_panels_key/2smf7uq9/obj1_f00030_official_large_tracklet.jpg` |
| 2smf7uq9 | 1 | 31 | baseline | False | 0.35 | uncertain | same_class_switch,unstable_tracklet | `artifacts/m10_qwen36/tracklet_panels_key/2smf7uq9/obj1_f00031_baseline_tracklet.jpg` |
| 8jsm23a7 | 1 | 10 | baseline | True | 0.75 | yes |  | `artifacts/m10_qwen36/tracklet_panels_key/8jsm23a7/obj1_f00010_baseline_tracklet.jpg` |
| 8jsm23a7 | 1 | 15 | baseline | True | 0.75 | yes | same_class_distractor,tiny_unreadable | `artifacts/m10_qwen36/tracklet_panels_key/8jsm23a7/obj1_f00015_baseline_tracklet.jpg` |

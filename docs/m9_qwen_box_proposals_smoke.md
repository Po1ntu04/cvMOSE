# M9 Qwen recovery box proposal summary

- Workspace: `/home/yu/projects/cv/from fdu/MOSEv2`
- Model requested: `Qwen3.5plus`
- Calls: 7
- Dry run: False
- Records with candidates: 4 / 7
- JSON: `artifacts/m9_qwen_boxes/proposals_key_smoke.json`
- Panels: `docs/assets/m9_qwen_boxes/proposal_panels`

| video | obj | frame | visible | candidates | model | notes |
| --- | ---: | ---: | --- | ---: | --- | --- |
| lcgc29va | 1 | 8 | uncertain | 2 | qwen-vl-max | Target is tiny and likely occluded or out of view; no clear bright green backpac |
| msinig6m | 1 | 70 | no | 0 | qwen-vl-max | REF 中的 bright green plush toy 在当前帧中未出现，所有可见对象均为灰色考拉或绿色制服人员，无绿色毛绒玩具迹象。 |
| q0sizv6m | 1 | 23 | yes | 3 | qwen-vl-plus | Multiple plausible candidates exist due to high density of same-class distractor |
| q0sizv6m | 2 | 23 | uncertain | 2 | qwen-vl-max | target likely partially occluded or out of frame; multiple similar guinea pigs p |
| r13u5z4y | 1 | 20 | no | 0 | qwen-vl-max | 目标草莓已被切开，当前帧中无完整或可识别的同一物理实例；仅见手部遮挡和残余果肉痕迹，无法确认原目标位置 |
| r13u5z4y | 1 | 24 | uncertain | 1 | qwen-vl-max | target likely obscured by hand; left-side strawberries are plausible but not uni |
| r13u5z4y | 1 | 32 | no | 0 | qwen-vl-max | 目标草莓已被切开，原实例已不存在；当前帧中仅剩切片和残余汁液，无完整或可识别的同一物理实例。 |

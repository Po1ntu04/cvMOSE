# M11 / M2 Visual Comparison Figures

This page tracks the key comparison figures committed to the repository for the M11 cycle gate and M2 reliable-memory-gate experiments.

## M11 cycle-gate figures

These figures compare RGB / SAM2 baseline / M11 cycle-gate behavior on representative cases.

- [`r13u5z4y_cycle_positive.jpg`](assets/m11_cycle/r13u5z4y_cycle_positive.jpg) — strawberry occlusion case; M11 suppresses post-occlusion candidate tracklets.
- [`msinig6m_cycle_candidate.jpg`](assets/m11_cycle/msinig6m_cycle_candidate.jpg) — crowded koala/person case; cycle gate rejects likely wrong reappearing clusters.
- [`4f98052b_cycle_mixed.jpg`](assets/m11_cycle/4f98052b_cycle_mixed.jpg) — repeated red/black chair/logo scene; mixed keep/reject behavior.
- [`c8lutf29_cycle_preserve.jpg`](assets/m11_cycle/c8lutf29_cycle_preserve.jpg) — edge/large-object protection case.
- [`pe0d85lk_cycle_preserve.jpg`](assets/m11_cycle/pe0d85lk_cycle_preserve.jpg) — large/edge person-region protection case.

## M2 summary and zoom figures

- [`summary_table.jpg`](assets/m2_memory_gate/summary_table.jpg) — mask-difference summary between SAM2 baseline and M2.
- [`r13u5z4y_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/r13u5z4y_m2_zoom_sheet.jpg) — strawberry long-occlusion/same-class slice case.
- [`q0sizv6m_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/q0sizv6m_m2_zoom_sheet.jpg) — crowded guinea-pig same-class drift case.
- [`lcgc29va_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/lcgc29va_m2_zoom_sheet.jpg) — tiny pedestrian over-suppression case.
- [`msinig6m_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/msinig6m_m2_zoom_sheet.jpg) — crowded koala/person wrong-recovery case.
- [`3epdtmyr_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/3epdtmyr_m2_zoom_sheet.jpg) — foreground animal/distractor case.
- [`1qlssuz2_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/1qlssuz2_m2_zoom_sheet.jpg) — small vehicle under bridge / re-identification case.
- [`4vznweiu_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/4vznweiu_m2_zoom_sheet.jpg) — tiny letter bead over-suppression case.
- [`4f98052b_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/4f98052b_m2_zoom_sheet.jpg) — repeated red/black room distractor case.
- [`z6dx46qr_m2_zoom_sheet.jpg`](assets/m2_memory_gate/zooms/z6dx46qr_m2_zoom_sheet.jpg) — low-contrast underwater/background false-positive case.

## M2 full-frame sheets

Full-frame sheets are included for traceability under [`assets/m2_memory_gate/full/`](assets/m2_memory_gate/full/):

- [`1qlssuz2_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/1qlssuz2_sam2_vs_m2_sheet.jpg)
- [`3epdtmyr_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/3epdtmyr_sam2_vs_m2_sheet.jpg)
- [`4f98052b_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/4f98052b_sam2_vs_m2_sheet.jpg)
- [`4vznweiu_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/4vznweiu_sam2_vs_m2_sheet.jpg)
- [`c8lutf29_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/c8lutf29_sam2_vs_m2_sheet.jpg)
- [`lcgc29va_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/lcgc29va_sam2_vs_m2_sheet.jpg)
- [`msinig6m_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/msinig6m_sam2_vs_m2_sheet.jpg)
- [`q0sizv6m_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/q0sizv6m_sam2_vs_m2_sheet.jpg)
- [`r13u5z4y_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/r13u5z4y_sam2_vs_m2_sheet.jpg)
- [`z6dx46qr_sam2_vs_m2_sheet.jpg`](assets/m2_memory_gate/full/z6dx46qr_sam2_vs_m2_sheet.jpg)

Interpretation details are in [`m2_visual_analysis.md`](m2_visual_analysis.md). M11/M2 mechanism-level comparison is summarized in the conversation and reflected by the figure names above.

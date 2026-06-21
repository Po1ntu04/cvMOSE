# MOSEv2 paper-like report

This directory contains the formal Chinese report and reproducible paper figures.

## Build

```bash
cd paper
/home/yu/miniconda3/envs/cv-hw2/bin/python make_figures.py
latexmk -xelatex -interaction=nonstopmode main.tex
```

The submission-ready PDF is `MOSEv2_training_free_reanchor_report.pdf`.

Before final submission, replace the explicit `请填写` fields on the cover with the student's college, major, and name.

## Evidence boundaries

- Scores are copied from user-provided CodaBench feedback and the preserved object-level metric memory in `docs/test_latest_metric_memory.md`.
- Qualitative figures are generated from the local MOSEv2 RGB frames and the preserved SAM2/final prediction roots.
- No later-frame ground truth is used by the method or the figures.

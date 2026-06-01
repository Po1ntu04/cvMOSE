# `test_latest.log` metric memory

This note preserves full-test numbers parsed from `/home/yu/projects/cv/from fdu/MOSEv2/homework/test_latest.log`, because the CodaBench leaderboard rounds to two decimals and hides sub-0.01 effects.

## Parsed runs

| label | source | rows | J | F_new | J&F_new (recomputed) | printed J&F_new | disappear J&F_new | reappear J&F_new | F | J&F | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `m10_q04v` | previous `test_latest.log` parse, submission `submission_m10_q04v.zip` | 575 | 41.55977 | 45.03115 | 43.29546 | 43.30 | 60.43 | 20.62 | 47.21 | 44.38 | q0+4v probe; rounds to 43.30. |
| `m7_part_balanced_43p30` | previous confirmed 43.30 M7 detail log parse | 575 | 41.56043 | 45.03151 | 43.29597 | 43.30 | 60.43 | 20.62 | 47.21 | 44.38 | Confirmed detailed log matches the leaderboard 43.30 family. |
| `m7_part_conservative_43p30` | previous current `test_latest.log` parse after user replaced with conservative log | 575 | 41.56043 | 45.03151 | 43.29597 | 43.30 | 60.43 | 20.62 | 47.21 | 44.38 | Metric-equivalent to confirmed M7 part-balanced at aggregate and key-row precision. |
| `claimed_m6_balanced_current_log` | current `test_latest.log` parse after user labeled it M6 balanced | 575 | 41.56043 | 45.03151 | 43.29597 | 43.30 | 60.43 | 20.62 | 47.21 | 44.38 | Identical to M7 balanced/conservative at aggregate and key-row precision; inconsistent with leaderboard row `submission_m6_balanced.zip=43.29` unless rounded/display mismatch or copied detail is not M6 balanced. |
| `previous_current_detail_43p25` | earlier accidental/current detail block before replacement | 575 | 41.51414 | 44.98529 | 43.24971 | 43.25 | 60.40 | 20.55 | 47.16 | 44.34 | Kept for context only; it was not the true 43.30 M7 detail block. |

## Delta: `m10_q04v` minus confirmed `m7_part_balanced_43p30`

| metric | delta |
| --- | ---: |
| J | -0.00066 |
| F_new | -0.00037 |
| J&F_new | -0.00051 |
| F | +0.00056 |
| J&F | -0.00005 |

Interpretation: `m10_q04v` and confirmed `m7_part_balanced_43p30` are essentially tied. At the true unrounded J&F_new level, q04v is **slightly lower** than M7 by about `0.0005`, which is far below leaderboard precision.

## Key homework-video rows: q04v vs confirmed M7

| video | rows | q04v J&F_new | M7 J&F_new | q04v - M7 | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `q0sizv6m` | 2 | 43.7925 | 43.7925 | 0.0000 | q0only/q04v does not improve q0 relative to confirmed M7 detail. |
| `4vznweiu` | 1 | 34.6450 | 34.9400 | -0.2950 | q04v is slightly worse on 4v; global effect is only about `-0.00051` over 575 rows. |
| `1qlssuz2` | 1 | 38.6200 | 38.6200 | 0.0000 | identical in the stored key row. |
| `2smf7uq9` | 1 | 91.6750 | 91.6750 | 0.0000 | identical; earlier apparent huge delta came from comparing q04v to the wrong 43.25 detail block. |
| `lcgc29va` | 1 | 53.8300 | 53.8300 | 0.0000 | identical; earlier apparent delta came from the wrong 43.25 detail block. |
| `r13u5z4y` | 1 | 39.1850 | 39.1850 | 0.0000 | unchanged. |
| `msinig6m` | 3 | 89.0200 | 89.0200 | 0.0000 | unchanged. |
| `8jsm23a7` | 1 | 2.1300 | 2.1300 | 0.0000 | unchanged in the key row. |

## Confirmed M7 key rows

| video | object rows | avg J | avg F_new | avg J&F_new | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `q0sizv6m` | 2 | 43.4100 | 44.1750 | 43.7925 | One object remains very low; q0 remains unsolved. |
| `4vznweiu` | 1 | 33.5000 | 36.3800 | 34.9400 | Slightly better than q04v; still a weak tiny/semantic target. |
| `1qlssuz2` | 1 | 36.8800 | 40.3600 | 38.6200 | Same as q04v stored row. |
| `2smf7uq9` | 1 | 89.9700 | 93.3800 | 91.6750 | Strong row; not a q04v-specific gain. |
| `lcgc29va` | 1 | 51.6100 | 56.0500 | 53.8300 | Same as q04v stored row. |
| `r13u5z4y` | 1 | 38.8400 | 39.5300 | 39.1850 | Reappearance remains 0/0. |
| `msinig6m` | 3 | 88.5833 | 89.4567 | 89.0200 | Same as q04v stored row. |

## Corrected inference

- The earlier conclusion that q04v gained mainly from `2smf7uq9` and `lcgc29va` was an artifact of comparing q04v against a 43.25 detail block, not against true M7 part-balanced.
- Against confirmed M7 part-balanced, q04v contributes no measurable q0 improvement and is slightly worse on `4vznweiu`.
- Therefore `submission_m10_q04v.zip`, `submission_m10_q0only.zip`, and `submission_mosev2_m7_part_balanced.zip` all landing at 43.30 is not hiding a meaningful q04v win; they are effectively the same score family.
- The dominant unresolved hard rows in this score family include `q0sizv6m` object row 2, `r13u5z4y` reappearance, very low `8jsm23a7`, and weak tiny/semantic cases such as `4vznweiu`.


## Delta: conservative minus balanced

At the available printed-detail precision, the current conservative detail log is identical to the confirmed balanced detail log:

| metric | conservative - balanced |
| --- | ---: |
| J | 0.00000 |
| F_new | 0.00000 |
| J&F_new | 0.00000 |
| disappear J&F_new | 0.00 |
| reappear J&F_new | 0.00 |
| F | 0.00000 |
| J&F | 0.00000 |

Key rows (`q0sizv6m`, `4vznweiu`, `1qlssuz2`, `2smf7uq9`, `8jsm23a7`, `r13u5z4y`, `msinig6m`, `lcgc29va`, etc.) also match the stored balanced key rows exactly at two-decimal per-row precision. Therefore conservative and balanced appear metric-equivalent, not merely leaderboard-rounded equivalent.

Correct implication: the balanced additions did not produce measurable full-test benefit over conservative; if they changed masks, their net effect was exactly neutral at the reported per-row precision or affected only pixels/frames not changing rounded row metrics.


## Claimed M6 balanced log check

The current file was labeled as M6 balanced by the user, but the parsed detail block is numerically identical to the stored M7 balanced/conservative logs:

- rows: 575
- J&F_new: `43.29597`
- J: `41.56043`
- F_new: `45.03151`
- key rows (`q0sizv6m`, `4vznweiu`, `1qlssuz2`, `2smf7uq9`, `8jsm23a7`, `r13u5z4y`, `msinig6m`, `lcgc29va`) match M7 at current precision.

Because the leaderboard row for `submission_m6_balanced.zip` was `43.29`, this log does not currently reveal a distinct M6-balanced metric signature. Treat it as metric-equivalent to M7 unless a new detail log shows different rows.

## Score leverage from current 43.30 family

To move from `43.296` to `44.0`, the system needs about `+0.704` global J&F_new, equivalent to about `+405` summed object-row J&F_new points over 575 rows. The highest-leverage known homework rows are:

| row | current J&F_new | gain if raised to 70 | gain if raised to 90 | implication |
| --- | ---: | ---: | ---: | --- |
| `8jsm23a7` objrow1 | 2.130 | +0.1180 | +0.1528 | Huge hidden-GT failure despite visually being treated as stable; should be re-opened first. |
| `q0sizv6m` objrow2 | 12.420 | +0.1001 | +0.1349 | Dense same-class case remains unsolved. |
| `amfdu83t` objrow1 | 28.175 | +0.0727 | +0.1075 | Under-discussed; likely high-value for inspection. |
| `4vznweiu` objrow1 | 34.940 | +0.0610 | +0.0958 | Tiny/semantic tile object; Qwen support was not enough. |
| `1qlssuz2` objrow1 | 38.620 | +0.0546 | +0.0894 | Prior balanced refinement is not enough. |
| `r13u5z4y` objrow1 | 39.185 | +0.0536 | +0.0884 | Reappearance remains zero; needs true re-acquisition, not more veto. |
| `z6dx46qr` objrow1 | 40.470 | +0.0514 | +0.0861 | Another under-discussed medium-low row. |

Even lifting all known 15-video target rows below 70 up to 70 yields only about `+0.562`, so reaching 44 from the 15 editable videos likely requires either (a) multiple low rows improved close to 90, or (b) finding additional non-obvious low rows among the 15 beyond the already-discussed cases.

## Most useful next log

Since conservative now appears metric-equivalent to balanced, the most useful next detailed log is `submission_m10_q0only.zip` if we want to isolate whether the q0-only anchor changes any row at all. If the goal is to find a route toward a real gain rather than diagnose M10, the next most useful detailed log is a genuinely different 43.29/43.30 family candidate such as `submission_m6_agg.zip` / `submission_m6_balanced.zip`, because it may reveal which object rows can move without hurting the stable M7 rows.

## M13 zofficial-balanced hidden feedback: 43.34

Latest `/home/yu/projects/cv/from fdu/MOSEv2/homework/test_latest.log` corresponds to `submission_mosev2_m13_8rev_zofficial_balanced.zip` / zofficial-balanced feedback:

- J&F_new: `43.34`
- J: `41.60`
- F_new: `45.07`
- disappear J&F_new: `60.43`
- reappear J&F_new: `20.62`
- F: `47.27`
- J&F: `44.44`

Compared with the confirmed 43.30 M7 family (`J&F_new≈43.29597`), this is about `+0.044` global J&F_new. The row-level evidence indicates the gain is almost entirely from `z6dx46qr`:

| video row | M7 family J&F_new | latest M13 J&F_new | delta | interpretation |
| --- | ---: | ---: | ---: | --- |
| `z6dx46qr` | `40.47` | `65.14` | `+24.67` | This explains roughly `+24.67 / 575 ≈ +0.043` global J&F_new, matching the total improvement. The useful component is the official-large interval replacement, not direct MLLM box prompting. |
| `8jsm23a7` | `2.13` | `2.13` | `0.00` | Despite visual improvement in the M13 reverse-anchor sheet, hidden metric did not improve at reported row precision. The mask is still not aligned with GT, or the visually plausible object hypothesis does not match the annotation. |
| `q0sizv6m` rows | `74.48/12.34` | `74.48/12.34` | `0.00` | q0 remains unchanged because the unsafe reverse-anchor/composite candidate was correctly excluded from this fusion. |

Conclusion: `43.34` is a **real small leaderboard improvement** and the first non-flat gain above the 43.30 family, but it is **not** the expected 44-level breakthrough. It validates that carefully selected per-object interval fusion can move hidden score, while the MLLM reverse-anchor path still lacks hidden-metric proof.

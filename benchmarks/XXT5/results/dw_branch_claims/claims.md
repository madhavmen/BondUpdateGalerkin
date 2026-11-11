# Claims of experiments/oldcap-fast-rsvd-xx at larger L

Accuracy is 'no worse' within a factor 1.5 of the reference arm's worst sampled S^z error.

## domain_wall, d = 2, L = 32, tol_basis = 0.0001

| claim | arm | measured | verdict |
|---|---|---|---|
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot2_Kadapt | columns x0.68, S^z error x1.00, wall x1.04 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 126) | RSVD_pilot2_Kadapt | max per entry 26 (max chi_old 36; rank_adaptive max 35) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_Kadapt | columns x0.72, S^z error x1.00, wall x1.04 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_Kadapt | max per entry 42 (max chi_old 38; rank_adaptive max 35) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot4_Kadapt | columns x0.76, S^z error x1.00, wall x1.05 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 124) | RSVD_pilot4_Kadapt | max per entry 29 (max chi_old 38; rank_adaptive max 35) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_margin8 | columns x0.71, S^z error x1.00, wall x0.99 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_margin8 | max per entry 36 (max chi_old 38; rank_adaptive max 35) | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor01 | mean probes per entry x1.00 of the fixed arm (max 16 vs 16), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor01 | selected 176 vs 2304, S^z error x1.00, unmet steps 0 | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor005 | mean probes per entry x1.00 of the fixed arm (max 16 vs 16), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor005 | selected 208 vs 2304, S^z error x1.00, unmet steps 0 | PASS |
| C3 same 8-column sketch as RSVD_fixed8rank4_Kadapt, no holdout | RSVD_tail8_factor01 | mean probes per entry x1.01 of the fixed arm (max 12 vs 12), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed8rank4_Kadapt, accuracy kept | RSVD_tail8_factor01 | selected 176 vs 1548, S^z error x1.00, unmet steps 0 | PASS |
| C4 per-bond cap 1.2 d chi_old | all BUG arms | 0 frames over the cap | PASS |

## domain_wall, d = 2, L = 32, tol_basis = 0.001

| claim | arm | measured | verdict |
|---|---|---|---|
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot2_Kadapt | columns x0.67, S^z error x1.31, wall x1.01 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 126) | RSVD_pilot2_Kadapt | max per entry 22 (max chi_old 29; rank_adaptive max 105) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_Kadapt | columns x0.70, S^z error x1.27, wall x0.97 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_Kadapt | max per entry 57 (max chi_old 29; rank_adaptive max 105) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot4_Kadapt | columns x0.75, S^z error x1.31, wall x0.98 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 124) | RSVD_pilot4_Kadapt | max per entry 70 (max chi_old 29; rank_adaptive max 105) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_margin8 | columns x0.69, S^z error x1.63, wall x0.97 | FAIL |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_margin8 | max per entry 41 (max chi_old 29; rank_adaptive max 105) | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor01 | mean probes per entry x0.99 of the fixed arm (max 16 vs 16), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor01 | selected 163 vs 1931, S^z error x2.19, unmet steps 0 | FAIL |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor005 | mean probes per entry x1.00 of the fixed arm (max 16 vs 16), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor005 | selected 188 vs 1931, S^z error x0.99, unmet steps 0 | PASS |
| C3 same 8-column sketch as RSVD_fixed8rank4_Kadapt, no holdout | RSVD_tail8_factor01 | mean probes per entry x1.00 of the fixed arm (max 12 vs 12), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed8rank4_Kadapt, accuracy kept | RSVD_tail8_factor01 | selected 166 vs 1338, S^z error x2.11, unmet steps 0 | FAIL |
| C4 per-bond cap 1.2 d chi_old | all BUG arms | 0 frames over the cap | PASS |

## domain_wall, d = 2, L = 32, tol_basis = 0.01

| claim | arm | measured | verdict |
|---|---|---|---|
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot2_Kadapt | columns x0.70, S^z error x0.57, wall x1.04 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 126) | RSVD_pilot2_Kadapt | max per entry 37 (max chi_old 22; rank_adaptive max 34) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_Kadapt | columns x0.74, S^z error x1.26, wall x0.98 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_Kadapt | max per entry 47 (max chi_old 22; rank_adaptive max 34) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot4_Kadapt | columns x0.78, S^z error x1.19, wall x0.98 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 124) | RSVD_pilot4_Kadapt | max per entry 35 (max chi_old 21; rank_adaptive max 34) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_margin8 | columns x0.72, S^z error x0.58, wall x0.97 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_margin8 | max per entry 26 (max chi_old 22; rank_adaptive max 34) | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor01 | mean probes per entry x1.00 of the fixed arm (max 15 vs 15), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor01 | selected 156 vs 1538, S^z error x1.00, unmet steps 0 | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor005 | mean probes per entry x1.00 of the fixed arm (max 16 vs 15), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor005 | selected 186 vs 1538, S^z error x1.03, unmet steps 0 | PASS |
| C3 same 8-column sketch as RSVD_fixed8rank4_Kadapt, no holdout | RSVD_tail8_factor01 | mean probes per entry x0.99 of the fixed arm (max 12 vs 12), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed8rank4_Kadapt, accuracy kept | RSVD_tail8_factor01 | selected 160 vs 1110, S^z error x2.02, unmet steps 0 | FAIL |
| C4 per-bond cap 1.2 d chi_old | all BUG arms | 0 frames over the cap | PASS |

## domain_wall, d = 2, L = 64, tol_basis = 0.0001

| claim | arm | measured | verdict |
|---|---|---|---|
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot2_Kadapt | columns x0.70, S^z error x1.00, wall x0.99 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 126) | RSVD_pilot2_Kadapt | max per entry 25 (max chi_old 40; rank_adaptive max 31) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_Kadapt | columns x0.74, S^z error x1.00, wall x0.96 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_Kadapt | max per entry 51 (max chi_old 39; rank_adaptive max 31) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot4_Kadapt | columns x0.78, S^z error x1.00, wall x1.04 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 124) | RSVD_pilot4_Kadapt | max per entry 61 (max chi_old 39; rank_adaptive max 31) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_margin8 | columns x0.74, S^z error x1.00, wall x0.96 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_margin8 | max per entry 93 (max chi_old 40; rank_adaptive max 31) | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor01 | mean probes per entry x1.00 of the fixed arm (max 16 vs 17), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor01 | selected 176 vs 2402, S^z error x1.00, unmet steps 0 | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor005 | mean probes per entry x1.01 of the fixed arm (max 17 vs 17), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor005 | selected 206 vs 2402, S^z error x1.00, unmet steps 0 | PASS |
| C3 same 8-column sketch as RSVD_fixed8rank4_Kadapt, no holdout | RSVD_tail8_factor01 | mean probes per entry x0.99 of the fixed arm (max 12 vs 12), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed8rank4_Kadapt, accuracy kept | RSVD_tail8_factor01 | selected 181 vs 1624, S^z error x1.00, unmet steps 0 | PASS |
| C4 per-bond cap 1.2 d chi_old | all BUG arms | 0 frames over the cap | PASS |

## domain_wall, d = 2, L = 64, tol_basis = 0.001

| claim | arm | measured | verdict |
|---|---|---|---|
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot2_Kadapt | columns x0.71, S^z error x0.94, wall x1.00 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 126) | RSVD_pilot2_Kadapt | max per entry 22 (max chi_old 30; rank_adaptive max 71) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_Kadapt | columns x0.74, S^z error x1.19, wall x1.00 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_Kadapt | max per entry 27 (max chi_old 29; rank_adaptive max 71) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot4_Kadapt | columns x0.78, S^z error x1.19, wall x0.99 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 124) | RSVD_pilot4_Kadapt | max per entry 35 (max chi_old 30; rank_adaptive max 71) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_margin8 | columns x0.74, S^z error x0.88, wall x1.02 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_margin8 | max per entry 30 (max chi_old 29; rank_adaptive max 71) | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor01 | mean probes per entry x1.00 of the fixed arm (max 16 vs 16), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor01 | selected 159 vs 2046, S^z error x0.91, unmet steps 0 | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor005 | mean probes per entry x1.01 of the fixed arm (max 16 vs 16), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor005 | selected 190 vs 2046, S^z error x0.91, unmet steps 0 | PASS |
| C3 same 8-column sketch as RSVD_fixed8rank4_Kadapt, no holdout | RSVD_tail8_factor01 | mean probes per entry x0.99 of the fixed arm (max 12 vs 12), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed8rank4_Kadapt, accuracy kept | RSVD_tail8_factor01 | selected 156 vs 1410, S^z error x0.94, unmet steps 0 | PASS |
| C4 per-bond cap 1.2 d chi_old | all BUG arms | 0 frames over the cap | PASS |

## domain_wall, d = 2, L = 64, tol_basis = 0.01

| claim | arm | measured | verdict |
|---|---|---|---|
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot2_Kadapt | columns x0.73, S^z error x0.95, wall x1.00 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 126) | RSVD_pilot2_Kadapt | max per entry 22 (max chi_old 22; rank_adaptive max 37) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_Kadapt | columns x0.76, S^z error x1.12, wall x0.99 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_Kadapt | max per entry 49 (max chi_old 23; rank_adaptive max 37) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot4_Kadapt | columns x0.80, S^z error x1.12, wall x0.97 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 124) | RSVD_pilot4_Kadapt | max per entry 26 (max chi_old 22; rank_adaptive max 37) | PASS |
| C1 fewer sketch columns than rank_adaptive, no worse | RSVD_pilot3_margin8 | columns x0.75, S^z error x1.44, wall x0.98 | PASS |
| C2 each draw at most 64 columns (per-entry sum <= 157) | RSVD_pilot3_margin8 | max per entry 24 (max chi_old 22; rank_adaptive max 37) | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor01 | mean probes per entry x0.98 of the fixed arm (max 16 vs 16), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor01 | selected 151 vs 1676, S^z error x1.33, unmet steps 0 | PASS |
| C3 same 12-column sketch as RSVD_fixed12rank8_Kadapt, no holdout | RSVD_tail12_factor005 | mean probes per entry x0.99 of the fixed arm (max 16 vs 16), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed12rank8_Kadapt, accuracy kept | RSVD_tail12_factor005 | selected 188 vs 1676, S^z error x1.36, unmet steps 0 | PASS |
| C3 same 8-column sketch as RSVD_fixed8rank4_Kadapt, no holdout | RSVD_tail8_factor01 | mean probes per entry x0.99 of the fixed arm (max 12 vs 12), holdout 0 | PASS |
| C3 fewer kept directions than RSVD_fixed8rank4_Kadapt, accuracy kept | RSVD_tail8_factor01 | selected 152 vs 1216, S^z error x1.81, unmet steps 0 | FAIL |
| C4 per-bond cap 1.2 d chi_old | all BUG arms | 0 frames over the cap | PASS |


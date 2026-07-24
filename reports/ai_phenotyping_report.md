# AI phenotyping workflow report

This report is generated from a synthetic dataset and demonstrates an AI/ML workflow for phenotyping traits.

## Test-set model performance

| Metric | Value |
|---|---:|
| RMSE | 1.119 |
| MAE | 0.8995 |
| R² | 0.9428 |

## Top features by permutation-style importance

| Rank | Feature | Mean R² drop |
|---:|---|---:|
| 1 | canopy_cover_pct | 0.2829 |
| 2 | ndvi_proxy | 0.1624 |
| 3 | disease_score_1_9 | 0.1245 |
| 4 | spad_chlorophyll_proxy | 0.0277 |
| 5 | thermal_depression_c | 0.0126 |
| 6 | red_edge_index_proxy | 0.0015 |

## Top candidate genotypes from AI selection-support score

| Rank | Genotype | AI selection-support score | Mean yield-potential score |
|---:|---|---:|---:|
| 1 | AIPHENO_G01 | 0.998 | 36.061 |
| 2 | AIPHENO_G20 | 0.951 | 35.151 |
| 3 | AIPHENO_G07 | 0.711 | 31.981 |
| 4 | AIPHENO_G04 | 0.71 | 30.959 |
| 5 | AIPHENO_G10 | 0.644 | 30.024 |
| 6 | AIPHENO_G17 | 0.627 | 30.289 |

## Interpretation

The workflow shows how phenotyping features can be used in a transparent predictive model and then translated into a candidate-ranking table.

The dataset is synthetic, so rankings should be interpreted only as workflow demonstration.

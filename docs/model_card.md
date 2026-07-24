# Model card

## Intended use

Demonstration of a transparent AI/ML workflow for plant phenotyping features after trait extraction.

## Data

Synthetic plot-level data with simulated image-derived and field-derived phenotyping traits.

## Model

Ridge regression implemented with NumPy.

## Validation

The model is evaluated on held-out genotypes to reduce plot-level leakage.

## Test performance

- RMSE: 1.119
- MAE: 0.8995
- R²: 0.9428

## Limitations

- Synthetic data only.
- Not an image-segmentation model.
- Not a UAV-processing pipeline.
- Not validated for real breeding decisions.
- Intended as a reproducible portfolio example, not a scientific claim.

"""
AI Plant Phenotyping Workflow in Python

Author: Mokkala Siva Prasad

Purpose:
    Demonstrate a transparent AI/ML workflow for plant phenotyping data.

Important:
    The data are synthetic. The image-derived traits are simulated proxies,
    not outputs from real UAV, drone, or image-processing pipelines.

Dependencies:
    Python standard library + NumPy
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260724)

FEATURES = [
    "canopy_cover_pct",
    "ndvi_proxy",
    "red_edge_index_proxy",
    "texture_uniformity",
    "thermal_depression_c",
    "plant_height_cm",
    "spad_chlorophyll_proxy",
    "disease_score_1_9",
    "days_to_maturity",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_numeric(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for key, value in row.items():
                if key in {"genotype", "replication"}:
                    converted[key] = value
                else:
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def generate_synthetic_phenotyping_data() -> Path:
    rows: list[dict] = []

    for genotype_index in range(1, 25):
        genotype = f"AIPHENO_G{genotype_index:02d}"

        growth_potential = RNG.normal(0.0, 1.0)
        stress_sensitivity = RNG.normal(0.0, 0.8)
        maturity_type = RNG.normal(0.0, 0.7)
        canopy_architecture = RNG.normal(0.0, 0.8)

        for replication in range(1, 5):
            noise = RNG.normal(0.0, 1.0)

            canopy_cover = 58 + 7.0 * growth_potential + 4.0 * canopy_architecture - 3.0 * stress_sensitivity + RNG.normal(0, 2.8)
            ndvi = 0.62 + 0.055 * growth_potential - 0.035 * stress_sensitivity + RNG.normal(0, 0.018)
            red_edge = 0.38 + 0.040 * growth_potential + 0.020 * canopy_architecture - 0.020 * stress_sensitivity + RNG.normal(0, 0.015)
            texture_uniformity = 0.52 + 0.050 * canopy_architecture - 0.040 * stress_sensitivity + RNG.normal(0, 0.020)
            thermal_depression = 4.2 + 0.55 * growth_potential - 0.75 * stress_sensitivity + RNG.normal(0, 0.35)
            plant_height = 47 + 5.5 * growth_potential + 3.0 * maturity_type + RNG.normal(0, 2.1)
            spad = 40 + 4.8 * growth_potential - 2.2 * stress_sensitivity + RNG.normal(0, 1.4)
            disease = 3.1 + 0.85 * stress_sensitivity - 0.45 * growth_potential + RNG.normal(0, 0.35)
            maturity = 82 + 3.5 * maturity_type + 0.8 * growth_potential + RNG.normal(0, 1.2)

            yield_potential = (
                14.0
                + 0.15 * canopy_cover
                + 12.0 * (ndvi - 0.62)
                + 5.0 * (red_edge - 0.38)
                + 0.13 * spad
                + 0.55 * thermal_depression
                - 1.10 * disease
                - 0.10 * max(0.0, maturity - 84)
                + RNG.normal(0, 1.1)
            )

            rows.append(
                {
                    "genotype": genotype,
                    "replication": replication,
                    "canopy_cover_pct": round(float(np.clip(canopy_cover, 35, 90)), 3),
                    "ndvi_proxy": round(float(np.clip(ndvi, 0.42, 0.90)), 3),
                    "red_edge_index_proxy": round(float(np.clip(red_edge, 0.25, 0.55)), 3),
                    "texture_uniformity": round(float(np.clip(texture_uniformity, 0.30, 0.78)), 3),
                    "thermal_depression_c": round(float(np.clip(thermal_depression, 1.5, 7.5)), 3),
                    "plant_height_cm": round(float(np.clip(plant_height, 30, 75)), 3),
                    "spad_chlorophyll_proxy": round(float(np.clip(spad, 25, 60)), 3),
                    "disease_score_1_9": round(float(np.clip(disease, 1, 9)), 3),
                    "days_to_maturity": round(float(np.clip(maturity, 72, 96)), 3),
                    "yield_potential_score": round(float(np.clip(yield_potential, 12, 42)), 3),
                }
            )

    path = DATA_DIR / "synthetic_image_derived_phenotyping.csv"
    write_csv(path, rows, ["genotype", "replication", *FEATURES, "yield_potential_score"])
    return path


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def fit_ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[float, np.ndarray]:
    x_aug = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)
    intercept = float(beta[0])
    coefficients = beta[1:]
    return intercept, coefficients


def predict_ridge(x: np.ndarray, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    return intercept + x @ coefficients


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(float(r2), 4)}


def permutation_importance(
    x_test: np.ndarray,
    y_test: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
    feature_names: list[str],
    repeats: int = 30,
) -> list[dict]:
    baseline = metrics(y_test, predict_ridge(x_test, intercept, coefficients))["r2"]
    rows: list[dict] = []

    for col, name in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            permuted = x_test.copy()
            RNG.shuffle(permuted[:, col])
            permuted_score = metrics(y_test, predict_ridge(permuted, intercept, coefficients))["r2"]
            drops.append(baseline - permuted_score)
        rows.append(
            {
                "feature": name,
                "mean_r2_drop": round(float(np.mean(drops)), 4),
                "std_r2_drop": round(float(np.std(drops, ddof=1)), 4),
            }
        )

    return sorted(rows, key=lambda r: r["mean_r2_drop"], reverse=True)


def aggregate_by_genotype(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["genotype"], []).append(row)

    genotype_rows = []
    for genotype, values in grouped.items():
        out = {"genotype": genotype}
        for feature in [*FEATURES, "yield_potential_score"]:
            out[feature] = round(float(np.mean([v[feature] for v in values])), 3)
        genotype_rows.append(out)

    return sorted(genotype_rows, key=lambda r: r["yield_potential_score"], reverse=True)


def scale(values: Iterable[float]) -> list[float]:
    vals = np.array(list(values), dtype=float)
    low, high = float(vals.min()), float(vals.max())
    if math.isclose(low, high):
        return [0.5 for _ in vals]
    return [float((v - low) / (high - low)) for v in vals]


def write_bar_svg(path: Path, rows: list[dict], value_key: str, label_key: str, title: str, color: str = "#2E7D32") -> None:
    width, height = 900, 520
    margin_left, margin_top, margin_bottom = 240, 70, 60
    plot_width = width - margin_left - 40
    bar_height = 28
    gap = 12
    ordered = rows[:10]
    values = [max(0.0, float(r[value_key])) for r in ordered]
    max_val = max(values) if values else 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{title}</text>',
    ]

    for i, row in enumerate(ordered):
        y = margin_top + i * (bar_height + gap)
        bar_width = (float(row[value_key]) / max_val) * plot_width if max_val else 0
        lines.append(f'<text x="{margin_left-12}" y="{y+20}" text-anchor="end" font-family="Arial" font-size="13">{row[label_key]}</text>')
        lines.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{margin_left+bar_width+8}" y="{y+20}" font-family="Arial" font-size="13">{float(row[value_key]):.3f}</text>')

    lines.append(f'<text x="{width/2}" y="{height-margin_bottom/2}" text-anchor="middle" font-family="Arial" font-size="12">Synthetic data demonstration</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scatter_svg(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    width, height = 720, 620
    margin = 80
    min_val = float(min(y_true.min(), y_pred.min())) - 1
    max_val = float(max(y_true.max(), y_pred.max())) + 1

    def xy(xv: float, yv: float) -> tuple[float, float]:
        x = margin + (xv - min_val) / (max_val - min_val) * (width - 2 * margin)
        y = height - margin - (yv - min_val) / (max_val - min_val) * (height - 2 * margin)
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]

    x1, y1 = xy(min_val, min_val)
    x2, y2 = xy(max_val, max_val)
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#999" stroke-dasharray="5,5"/>')

    for observed, predicted in zip(y_true, y_pred):
        x, y = xy(float(observed), float(predicted))
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#1565C0" opacity="0.75"/>')

    lines.append(f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="14">Observed yield-potential score</text>')
    lines.append(f'<text x="24" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24 {height/2})">Predicted yield-potential score</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data_path = generate_synthetic_phenotyping_data()
    rows = read_csv_numeric(data_path)

    genotypes = np.array([row["genotype"] for row in rows])
    unique_genotypes = sorted(set(genotypes))
    test_genotypes = set(unique_genotypes[::4])

    x = np.array([[row[f] for f in FEATURES] for row in rows], dtype=float)
    y = np.array([row["yield_potential_score"] for row in rows], dtype=float)

    test_mask = np.array([g in test_genotypes for g in genotypes])
    train_mask = ~test_mask

    x_train, x_test, _, _ = standardize_train_test(x[train_mask], x[test_mask])
    y_train, y_test = y[train_mask], y[test_mask]

    intercept, coefficients = fit_ridge_regression(x_train, y_train, alpha=1.2)
    y_pred = predict_ridge(x_test, intercept, coefficients)
    model_metrics = metrics(y_test, y_pred)

    importance_rows = permutation_importance(x_test, y_test, intercept, coefficients, FEATURES)

    coefficient_rows = [
        {"feature": feature, "standardized_coefficient": round(float(coef), 4)}
        for feature, coef in sorted(zip(FEATURES, coefficients), key=lambda item: abs(item[1]), reverse=True)
    ]

    prediction_rows = []
    for row, observed, predicted in zip([r for r, is_test in zip(rows, test_mask) if is_test], y_test, y_pred):
        prediction_rows.append(
            {
                "genotype": row["genotype"],
                "replication": row["replication"],
                "observed_yield_potential_score": round(float(observed), 3),
                "predicted_yield_potential_score": round(float(predicted), 3),
                "residual": round(float(observed - predicted), 3),
            }
        )

    genotype_rows = aggregate_by_genotype(rows)
    feature_scaled = {
        "yield_potential_score": scale([r["yield_potential_score"] for r in genotype_rows]),
        "ndvi_proxy": scale([r["ndvi_proxy"] for r in genotype_rows]),
        "spad_chlorophyll_proxy": scale([r["spad_chlorophyll_proxy"] for r in genotype_rows]),
        "disease_score_1_9": scale([-r["disease_score_1_9"] for r in genotype_rows]),
        "thermal_depression_c": scale([r["thermal_depression_c"] for r in genotype_rows]),
    }

    for i, row in enumerate(genotype_rows):
        row["ai_selection_support_score"] = round(
            0.40 * feature_scaled["yield_potential_score"][i]
            + 0.20 * feature_scaled["ndvi_proxy"][i]
            + 0.15 * feature_scaled["spad_chlorophyll_proxy"][i]
            + 0.15 * feature_scaled["disease_score_1_9"][i]
            + 0.10 * feature_scaled["thermal_depression_c"][i],
            3,
        )

    top_candidates = sorted(genotype_rows, key=lambda r: r["ai_selection_support_score"], reverse=True)[:8]

    write_csv(OUTPUT_DIR / "predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    write_csv(OUTPUT_DIR / "feature_importance.csv", importance_rows, ["feature", "mean_r2_drop", "std_r2_drop"])
    write_csv(OUTPUT_DIR / "model_coefficients.csv", coefficient_rows, ["feature", "standardized_coefficient"])
    write_csv(OUTPUT_DIR / "top_candidate_genotypes.csv", top_candidates, ["genotype", *FEATURES, "yield_potential_score", "ai_selection_support_score"])

    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(
            {
                "model": "ridge regression implemented with NumPy",
                "split_strategy": "held-out genotypes",
                "target": "yield_potential_score",
                "n_train_plots": int(train_mask.sum()),
                "n_test_plots": int(test_mask.sum()),
                "test_genotypes": sorted(test_genotypes),
                **model_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_scatter_svg(OUTPUT_DIR / "predicted_vs_observed.svg", y_test, y_pred, "Predicted vs Observed Yield-Potential Score")
    write_bar_svg(OUTPUT_DIR / "feature_importance.svg", importance_rows, "mean_r2_drop", "feature", "Permutation-style Feature Importance", "#2E7D32")

    model_card = [
        "# Model card",
        "",
        "## Intended use",
        "",
        "Demonstration of a transparent AI/ML workflow for plant phenotyping features after trait extraction.",
        "",
        "## Data",
        "",
        "Synthetic plot-level data with simulated image-derived and field-derived phenotyping traits.",
        "",
        "## Model",
        "",
        "Ridge regression implemented with NumPy.",
        "",
        "## Validation",
        "",
        "The model is evaluated on held-out genotypes to reduce plot-level leakage.",
        "",
        "## Test performance",
        "",
        f"- RMSE: {model_metrics['rmse']}",
        f"- MAE: {model_metrics['mae']}",
        f"- R²: {model_metrics['r2']}",
        "",
        "## Limitations",
        "",
        "- Synthetic data only.",
        "- Not an image-segmentation model.",
        "- Not a UAV-processing pipeline.",
        "- Not validated for real breeding decisions.",
        "- Intended as a reproducible portfolio example, not a scientific claim.",
    ]
    (DOCS_DIR / "model_card.md").write_text("\n".join(model_card), encoding="utf-8")

    report = [
        "# AI phenotyping workflow report",
        "",
        "This report is generated from a synthetic dataset and demonstrates an AI/ML workflow for phenotyping traits.",
        "",
        "## Test-set model performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| RMSE | {model_metrics['rmse']} |",
        f"| MAE | {model_metrics['mae']} |",
        f"| R² | {model_metrics['r2']} |",
        "",
        "## Top features by permutation-style importance",
        "",
        "| Rank | Feature | Mean R² drop |",
        "|---:|---|---:|",
    ]
    for rank, row in enumerate(importance_rows[:6], start=1):
        report.append(f"| {rank} | {row['feature']} | {row['mean_r2_drop']} |")

    report.extend(
        [
            "",
            "## Top candidate genotypes from AI selection-support score",
            "",
            "| Rank | Genotype | AI selection-support score | Mean yield-potential score |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_candidates[:6], start=1):
        report.append(f"| {rank} | {row['genotype']} | {row['ai_selection_support_score']} | {row['yield_potential_score']} |")

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "The workflow shows how phenotyping features can be used in a transparent predictive model and then translated into a candidate-ranking table.",
            "",
            "The dataset is synthetic, so rankings should be interpreted only as workflow demonstration.",
        ]
    )
    (REPORT_DIR / "ai_phenotyping_report.md").write_text("\n".join(report), encoding="utf-8")

    print("AI phenotyping workflow complete.")
    print(json.dumps(model_metrics, indent=2))
    print("Top candidate genotypes:")
    for row in top_candidates[:5]:
        print(row["genotype"], row["ai_selection_support_score"], row["yield_potential_score"])


if __name__ == "__main__":
    main()

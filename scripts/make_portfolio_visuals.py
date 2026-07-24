"""
Create polished SVG visuals for the AI plant phenotyping portfolio repository.

The visuals are generated only from synthetic demonstration outputs already
present in the repository.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

FONT = "Inter, Segoe UI, Arial, sans-serif"
INK = "#17312B"
MUTED = "#60746C"
GRID = "#DDE8E2"
GREEN = "#2E7D32"
TEAL = "#009688"
BLUE = "#1E88E5"
AMBER = "#F9A825"
PAPER = "#FBFDF9"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(value: str) -> float:
    return float(value)


def write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def defs() -> list[str]:
    return [
        "<defs>",
        '<linearGradient id="heroGradient" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#0B3D2E"/>',
        '<stop offset="55%" stop-color="#1B7F5A"/>',
        '<stop offset="100%" stop-color="#63B967"/>',
        "</linearGradient>",
        '<linearGradient id="softGreen" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0%" stop-color="#2E7D32"/>',
        '<stop offset="100%" stop-color="#00A896"/>',
        "</linearGradient>",
        '<filter id="softShadow" x="-10%" y="-10%" width="120%" height="130%">',
        '<feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#163B2B" flood-opacity="0.14"/>',
        "</filter>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#2E7D32"/></marker>',
        "</defs>",
    ]


def text(x: float, y: float, value: str, size: int = 16, weight: int = 400, fill: str = INK, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def rounded_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", radius: int = 22) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def portfolio_overview() -> None:
    metrics = json.loads((OUTPUT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    top_features = read_csv(OUTPUT_DIR / "feature_importance.csv")[:5]
    top_candidates = read_csv(OUTPUT_DIR / "top_candidate_genotypes.csv")[:4]

    width, height = 1180, 560
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">AI plant phenotyping portfolio overview</title>",
        "<desc id=\"desc\">Synthetic phenotyping workflow summary with model metrics, top features, candidate genotypes, and workflow stages.</desc>",
        *defs(),
        f'<rect width="{width}" height="{height}" rx="28" fill="{PAPER}"/>',
        '<rect x="28" y="28" width="1124" height="160" rx="28" fill="url(#heroGradient)" filter="url(#softShadow)"/>',
        text(62, 84, "AI plant phenotyping workflow", 30, 500, "#FFFFFF"),
        text(62, 126, "Image-derived traits → validated prediction → candidate ranking", 17, 400, "#E6F4EA"),
        text(62, 162, "Portfolio demonstration for AI-enabled crop phenotyping roles", 14, 400, "#D3EFE0"),
    ]

    metric_cards = [
        ("RMSE", f"{metrics['rmse']:.3f}", "prediction error"),
        ("MAE", f"{metrics['mae']:.4f}", "mean absolute error"),
        ("R²", f"{metrics['r2']:.4f}", "held-out genotypes"),
    ]
    for i, (label, value, note) in enumerate(metric_cards):
        x = 678 + i * 150
        lines.append(rounded_rect(x, 70, 126, 88, "#FFFFFF", "none", 18))
        lines.append(text(x + 18, 100, label, 13, 500, MUTED))
        lines.append(text(x + 18, 130, value, 24, 500, GREEN))
        lines.append(text(x + 18, 150, note, 11, 400, MUTED))

    lines.extend([
        rounded_rect(36, 220, 525, 284, "#FFFFFF", "#DCEBE2", 24),
        text(66, 264, "Top image-derived features", 20, 500),
        text(66, 291, "Permutation-style importance from the synthetic model", 13, 400, MUTED),
    ])
    max_drop = max(num(row["mean_r2_drop"]) for row in top_features)
    for i, row in enumerate(top_features):
        y = 330 + i * 34
        value = num(row["mean_r2_drop"])
        bar_w = 315 * value / max_drop
        label = row["feature"].replace("_", " ")
        lines.append(text(70, y + 18, label[:24], 13, 400, INK))
        lines.append(f'<rect x="238" y="{y}" width="245" height="18" rx="9" fill="#E9F5EE"/>')
        lines.append(f'<rect x="238" y="{y}" width="{245 * value / max_drop:.1f}" height="18" rx="9" fill="url(#softGreen)"/>')
        lines.append(text(500, y + 15, f"{value:.4f}", 12, 500, MUTED))

    lines.extend([
        rounded_rect(610, 220, 532, 284, "#FFFFFF", "#DCEBE2", 24),
        text(640, 264, "Candidate ranking preview", 20, 500),
        text(640, 291, "Top genotypes by AI selection-support score", 13, 400, MUTED),
    ])
    for i, row in enumerate(top_candidates):
        y = 326 + i * 42
        score = num(row["ai_selection_support_score"])
        lines.append(f'<circle cx="662" cy="{y+10}" r="16" fill="#E8F5E9" stroke="#B9DEC4"/>')
        lines.append(text(662, y + 16, str(i + 1), 13, 500, GREEN, "middle"))
        lines.append(text(696, y + 15, row["genotype"], 15, 500, INK))
        lines.append(f'<rect x="850" y="{y}" width="220" height="20" rx="10" fill="#EEF6F2"/>')
        lines.append(f'<rect x="850" y="{y}" width="{220*score:.1f}" height="20" rx="10" fill="#1E88E5" opacity="0.85"/>')
        lines.append(text(1084, y + 16, f"{score:.3f}", 12, 500, MUTED))

    lines.append("</svg>")
    write(OUTPUT_DIR / "portfolio_overview.svg", lines)


def predicted_vs_observed() -> None:
    rows = read_csv(OUTPUT_DIR / "predictions.csv")
    metrics = json.loads((OUTPUT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    obs = [num(r["observed_yield_potential_score"]) for r in rows]
    pred = [num(r["predicted_yield_potential_score"]) for r in rows]
    low = min(min(obs), min(pred)) - 1.0
    high = max(max(obs), max(pred)) + 1.0
    width, height = 920, 640
    left, top, plot_w, plot_h = 96, 96, 700, 420

    def xy(xv: float, yv: float) -> tuple[float, float]:
        x = left + (xv - low) / (high - low) * plot_w
        y = top + plot_h - (yv - low) / (high - low) * plot_h
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Predicted versus observed yield-potential score</title>",
        "<desc id=\"desc\">Scatter plot comparing observed and predicted yield-potential scores for held-out synthetic phenotyping observations.</desc>",
        *defs(),
        f'<rect width="{width}" height="{height}" rx="28" fill="{PAPER}"/>',
        text(58, 50, "Predicted vs observed yield-potential score", 24, 500),
        text(58, 78, "Held-out genotype validation using synthetic phenotyping traits", 14, 400, MUTED),
    ]
    for i in range(6):
        val = low + i * (high - low) / 5
        x, y = xy(val, val)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(text(left - 12, y + 5, f"{val:.0f}", 12, 400, MUTED, "end"))
        lines.append(text(x, top + plot_h + 26, f"{val:.0f}", 12, 400, MUTED, "middle"))
    x1, y1 = xy(low, low)
    x2, y2 = xy(high, high)
    lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#8CBBA3" stroke-width="2" stroke-dasharray="8 8"/>')
    for o, p in zip(obs, pred):
        x, y = xy(o, p)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{TEAL}" opacity="0.72" stroke="#FFFFFF" stroke-width="1.2"/>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="{INK}" stroke-width="1.4"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="{INK}" stroke-width="1.4"/>')
    lines.append(text(left + plot_w / 2, 585, "Observed yield-potential score", 14, 500, INK, "middle"))
    lines.append(f'<text x="26" y="{top+plot_h/2:.1f}" font-family="{FONT}" font-size="14" font-weight="500" fill="{INK}" text-anchor="middle" transform="rotate(-90 26 {top+plot_h/2:.1f})">Predicted yield-potential score</text>')
    lines.append(rounded_rect(690, 118, 170, 104, "#FFFFFF", "#DCEBE2", 18))
    lines.append(text(715, 149, "Model fit", 13, 500, MUTED))
    lines.append(text(715, 184, f"R² {metrics['r2']:.4f}", 28, 500, GREEN))
    lines.append(text(715, 210, f"RMSE {metrics['rmse']:.3f}", 13, 400, MUTED))
    lines.append(text(732, 538, "Dashed line = perfect prediction", 12, 400, MUTED, "middle"))
    lines.append("</svg>")
    write(OUTPUT_DIR / "predicted_vs_observed.svg", lines)


def feature_importance() -> None:
    rows = read_csv(OUTPUT_DIR / "feature_importance.csv")[:8]
    max_drop = max(num(row["mean_r2_drop"]) for row in rows)
    width, height = 920, 560
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Feature-importance ranking</title>",
        "<desc id=\"desc\">Horizontal bar chart showing the most influential synthetic phenotyping variables by mean R squared drop.</desc>",
        *defs(),
        f'<rect width="{width}" height="{height}" rx="28" fill="{PAPER}"/>',
        text(58, 55, "Feature importance", 25, 500),
        text(58, 84, "Mean R² drop when each synthetic phenotyping feature is permuted", 14, 400, MUTED),
    ]
    for i, row in enumerate(rows):
        y = 125 + i * 46
        label = row["feature"].replace("_", " ")
        value = num(row["mean_r2_drop"])
        bar_w = 520 * value / max_drop
        lines.append(text(62, y + 22, label, 14, 500, INK))
        lines.append(f'<rect x="330" y="{y}" width="520" height="24" rx="12" fill="#E8F3EE"/>')
        lines.append(f'<rect x="330" y="{y}" width="{bar_w:.1f}" height="24" rx="12" fill="url(#softGreen)"/>')
        lines.append(text(865, y + 18, f"{value:.4f}", 13, 500, MUTED))
    lines.append(text(58, 515, "Higher value means the model lost more predictive power when that feature was disrupted.", 13, 400, MUTED))
    lines.append("</svg>")
    write(OUTPUT_DIR / "feature_importance.svg", lines)


def workflow_diagram() -> None:
    width, height = 1100, 330
    stages = [
        ("1", "Phenotyping inputs", "Canopy, NDVI proxy, SPAD, disease, maturity"),
        ("2", "Model training", "Ridge regression using synthetic plot-level traits"),
        ("3", "Validation", "Held-out genotype test set with RMSE, MAE, R²"),
        ("4", "Interpretation", "Feature importance and candidate ranking"),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">AI phenotyping workflow diagram</title>",
        "<desc id=\"desc\">Four stage workflow from synthetic phenotyping inputs to model training, validation, and interpretation.</desc>",
        *defs(),
        f'<rect width="{width}" height="{height}" rx="28" fill="{PAPER}"/>',
        text(50, 55, "AI phenotyping workflow", 25, 500),
        text(50, 84, "A transparent pipeline from crop traits to explainable prediction outputs", 14, 400, MUTED),
    ]
    for i, (number, title, body) in enumerate(stages):
        x = 54 + i * 260
        y = 130
        lines.append(rounded_rect(x, y, 220, 128, "#FFFFFF", "#DCEBE2", 22))
        lines.append(f'<circle cx="{x+34}" cy="{y+35}" r="18" fill="#E8F5E9" stroke="#B5DABF"/>')
        lines.append(text(x + 34, y + 41, number, 15, 500, GREEN, "middle"))
        lines.append(text(x + 64, y + 40, title, 16, 500, INK))
        lines.append(text(x + 24, y + 78, body[:31], 12, 400, MUTED))
        lines.append(text(x + 24, y + 99, body[31:], 12, 400, MUTED))
        if i < len(stages) - 1:
            ax = x + 226
            lines.append(f'<path d="M {ax} {y+64} L {ax+36} {y+64}" stroke="{GREEN}" stroke-width="2.2" marker-end="url(#arrow)"/>')
    lines.append("</svg>")
    write(OUTPUT_DIR / "workflow_diagram.svg", lines)


def main() -> None:
    portfolio_overview()
    predicted_vs_observed()
    feature_importance()
    workflow_diagram()
    print("Portfolio visuals refreshed for AI plant phenotyping repository.")


if __name__ == "__main__":
    main()

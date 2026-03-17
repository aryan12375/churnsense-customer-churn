"""
src/shap_explain.py
───────────────────
SHAP-based explainability for the champion model.

Produces:
  1. Global summary plot  — feature importance across all predictions
  2. Global bar plot      — mean |SHAP| per feature, top-20
  3. Dependence plots     — top-5 features: SHAP value vs raw feature value
  4. Local force plots    — per-customer explanation as interactive HTML
  5. High-risk customer report — top-N at-risk customers with SHAP breakdowns

Usage:
    python src/shap_explain.py
    python src/shap_explain.py --top-n 20 --export-html
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from src.preprocessing import ChurnPreprocessor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Chart theme ────────────────────────────────────────────────────────────────
BG_DARK    = "#0F1117"
BG_PANEL   = "#161B27"
ACCENT     = "#6C63FF"
CHAMPION   = "#00E5A0"
WARN       = "#FF6B6B"
TEXT_WHITE = "#EAEAEA"
TEXT_MUTED = "#808898"
GRID_COLOR = "#232838"


class ShapExplainer:

    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.feature_names: list[str] = []

    def run(
        self,
        csv_path: str = "data/telco_churn.csv",
        top_n: int = 10,
        export_html: bool = False,
    ) -> None:
        # ── Load data and model ────────────────────────────────────────────
        prep = ChurnPreprocessor()
        _, X_test, _, y_test, feature_names = prep.run(csv_path)
        self.feature_names = feature_names

        champion = joblib.load(MODELS_DIR / "champion.pkl")
        logger.info(f"Loaded champion model: {type(champion).__name__}")

        # ── Build SHAP explainer ───────────────────────────────────────────
        logger.info("Computing SHAP values (this may take ~30s)...")
        self.explainer = shap.TreeExplainer(champion)
        self.shap_values = self.explainer.shap_values(X_test)

        # shap_values is (n_samples, n_features) for binary classification
        if isinstance(self.shap_values, list):
            sv = self.shap_values[1]   # positive class
        else:
            sv = self.shap_values

        # ── Generate all plots ─────────────────────────────────────────────
        self._plot_summary_beeswarm(sv, X_test)
        self._plot_bar_importance(sv)
        self._plot_dependence_grid(sv, X_test, top_n=5)
        self._generate_high_risk_report(sv, X_test, y_test, top_n=top_n)

        if export_html:
            self._export_force_plots_html(sv, X_test, top_n=top_n)

        logger.info("── SHAP analysis complete ────────────────────────────────")

    # ──────────────────────────────────────────────────────────────────────────
    # Plot: summary beeswarm
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_summary_beeswarm(self, sv: np.ndarray, X_test: np.ndarray) -> None:
        """
        Beeswarm plot: each dot = one customer.
        x-axis = SHAP value (impact on churn probability).
        Color = feature value (red = high, blue = low).
        """
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG_DARK)
        ax.set_facecolor(BG_PANEL)

        shap.summary_plot(
            sv,
            X_test,
            feature_names=self._clean_names(self.feature_names),
            max_display=20,
            show=False,
            plot_type="dot",
        )

        ax = plt.gca()
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.xaxis.label.set_color(TEXT_MUTED)
        ax.set_xlabel("SHAP value (impact on churn probability)", color=TEXT_MUTED, fontsize=11)

        plt.title(
            "Feature Impact on Churn Predictions — SHAP Summary",
            color=TEXT_WHITE, fontsize=13, fontweight="bold", pad=14,
        )
        plt.gcf().set_facecolor(BG_DARK)
        plt.tight_layout()

        out = REPORTS_DIR / "shap_summary_beeswarm.png"
        plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
        plt.close()
        logger.info(f"Saved → {out}")

    # ──────────────────────────────────────────────────────────────────────────
    # Plot: bar importance
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_bar_importance(self, sv: np.ndarray) -> None:
        mean_abs = np.abs(sv).mean(axis=0)
        indices  = np.argsort(mean_abs)[::-1][:20]

        names  = [self._clean_names(self.feature_names)[i] for i in indices]
        values = mean_abs[indices]

        plt.rcParams.update({"font.family": "DejaVu Sans"})
        fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG_DARK)
        ax.set_facecolor(BG_PANEL)

        # Color gradient: top features glow warmer
        colors = [
            CHAMPION if i == 0 else ACCENT if i < 5 else "#4A4580"
            for i in range(len(names))
        ]

        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.6)

        for bar, val in zip(bars, values[::-1]):
            ax.text(
                val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                color=TEXT_WHITE, fontsize=9.5,
            )

        ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
        ax.set_xlabel("Mean |SHAP value|", color=TEXT_MUTED, fontsize=11)
        ax.set_title(
            "Top-20 Features by Global SHAP Importance",
            color=TEXT_WHITE, fontsize=13, fontweight="bold", pad=14,
        )

        legend_patches = [
            mpatches.Patch(color=CHAMPION, label="#1 driver"),
            mpatches.Patch(color=ACCENT,   label="Top 5"),
            mpatches.Patch(color="#4A4580", label="Remaining"),
        ]
        legend = ax.legend(handles=legend_patches, loc="lower right",
                           frameon=True, fontsize=10)
        legend.get_frame().set_facecolor(BG_DARK)
        for t in legend.get_texts():
            t.set_color(TEXT_WHITE)

        plt.tight_layout()
        out = REPORTS_DIR / "shap_bar_importance.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
        plt.close()
        logger.info(f"Saved → {out}")

    # ──────────────────────────────────────────────────────────────────────────
    # Plot: dependence plots for top-5 features
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_dependence_grid(self, sv: np.ndarray, X_test: np.ndarray, top_n: int = 5) -> None:
        mean_abs = np.abs(sv).mean(axis=0)
        top_idxs = np.argsort(mean_abs)[::-1][:top_n]
        names    = self._clean_names(self.feature_names)

        fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 5), facecolor=BG_DARK)
        if top_n == 1:
            axes = [axes]

        for ax, idx in zip(axes, top_idxs):
            feat_name = names[idx]
            x_vals    = X_test[:, idx]
            s_vals    = sv[:, idx]

            ax.set_facecolor(BG_PANEL)
            sc = ax.scatter(
                x_vals, s_vals,
                c=s_vals, cmap="RdYlGn_r",
                alpha=0.4, s=12, linewidths=0,
            )
            ax.axhline(0, color=TEXT_MUTED, linewidth=0.8, linestyle="--")
            ax.set_xlabel(feat_name, color=TEXT_MUTED, fontsize=9)
            ax.set_ylabel("SHAP value", color=TEXT_MUTED, fontsize=9)
            ax.set_title(feat_name, color=TEXT_WHITE, fontsize=10, fontweight="bold")
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)

        fig.suptitle(
            "Dependence Plots — Top-5 Features",
            color=TEXT_WHITE, fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        out = REPORTS_DIR / "shap_dependence_grid.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
        plt.close()
        logger.info(f"Saved → {out}")

    # ──────────────────────────────────────────────────────────────────────────
    # High-risk customer report
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_high_risk_report(
        self,
        sv: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        top_n: int = 10,
    ) -> None:
        """
        Identify the top-N highest-risk customers and explain each one.
        Saves a JSON report that the dashboard reads.
        """
        # Expected value + sum of SHAP values ≈ log-odds → convert to probability
        base_val   = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1]

        log_odds   = base_val + sv.sum(axis=1)
        churn_prob = 1 / (1 + np.exp(-log_odds))

        top_indices = np.argsort(churn_prob)[::-1][:top_n]
        names       = self._clean_names(self.feature_names)

        customers = []
        for rank, idx in enumerate(top_indices):
            # Top-5 drivers for this customer (positive = push toward churn)
            feature_contribs = sorted(
                zip(names, sv[idx].tolist(), X_test[idx].tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]

            customers.append({
                "rank":        rank + 1,
                "customer_id": f"CUST_{idx:04d}",
                "churn_prob":  round(float(churn_prob[idx]), 4),
                "actual_churn": int(y_test[idx]) if hasattr(y_test, "__iter__") else None,
                "risk_tier":   (
                    "critical" if churn_prob[idx] >= 0.80 else
                    "high"     if churn_prob[idx] >= 0.65 else
                    "medium"
                ),
                "top_drivers": [
                    {
                        "feature": f,
                        "shap_value": round(s, 4),
                        "feature_value": round(v, 4),
                        "direction": "increases_churn" if s > 0 else "reduces_churn",
                    }
                    for f, s, v in feature_contribs
                ],
            })

        report = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "total_customers_scored": int(len(churn_prob)),
            "high_risk_threshold": 0.65,
            "customers": customers,
        }

        out = REPORTS_DIR / "high_risk_customers.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved high-risk report → {out}  ({len(customers)} customers)")

    # ──────────────────────────────────────────────────────────────────────────
    # Optional: interactive HTML force plots
    # ──────────────────────────────────────────────────────────────────────────

    def _export_force_plots_html(
        self, sv: np.ndarray, X_test: np.ndarray, top_n: int = 5
    ) -> None:
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1]

        names = self._clean_names(self.feature_names)

        shap.initjs()
        log_odds  = base_val + sv.sum(axis=1)
        top_idxs  = np.argsort(log_odds)[::-1][:top_n]

        html_parts = [shap.getjs()]
        for i, idx in enumerate(top_idxs):
            plot = shap.force_plot(
                base_val,
                sv[idx],
                X_test[idx],
                feature_names=names,
                show=False,
            )
            html_parts.append(f"<h3>Customer rank #{i+1} (index {idx})</h3>")
            html_parts.append(plot.html())

        full_html = "<html><body style='background:#0F1117;color:#EAEAEA;font-family:sans-serif;padding:24px'>"
        full_html += "<h1>ChurnSense — SHAP Force Plots (Top At-Risk Customers)</h1>"
        full_html += "".join(html_parts)
        full_html += "</body></html>"

        out = REPORTS_DIR / "shap_force_plots.html"
        out.write_text(full_html)
        logger.info(f"Saved interactive SHAP plots → {out}")

    # ──────────────────────────────────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_names(names: list[str]) -> list[str]:
        """Convert one-hot encoded names like 'cat__Contract_Month-to-month' → 'Contract: Month-to-month'."""
        cleaned = []
        for n in names:
            n = n.replace("cat__", "").replace("num__", "")
            if "_" in n:
                parts = n.split("_", 1)
                n = f"{parts[0]}: {parts[1].replace('_', ' ')}"
            cleaned.append(n)
        return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/telco_churn.csv")
    parser.add_argument("--top-n",       type=int, default=10)
    parser.add_argument("--export-html", action="store_true")
    args = parser.parse_args()

    explainer = ShapExplainer()
    explainer.run(csv_path=args.data, top_n=args.top_n, export_html=args.export_html)

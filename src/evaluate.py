"""
src/evaluate.py
───────────────
Comprehensive model evaluation with charts + JSON export for dashboard.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

BG_DARK    = "#0F1117"
BG_PANEL   = "#161B27"
BG_CARD    = "#1C2130"
ACCENT     = "#6C63FF"
CHAMPION   = "#00E5A0"
WARN       = "#FF6B6B"
TEXT_WHITE = "#EAEAEA"
TEXT_MUTED = "#808898"
GRID_COLOR = "#232838"


def _apply_dark_theme(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT_WHITE)
    ax.set_axisbelow(True)
    ax.grid(color=GRID_COLOR, linewidth=0.6, zorder=0)


class ModelEvaluator:

    def compute(self, model, X_test, y_test, name):
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        f1      = f1_score(y_test, y_pred, average="binary")
        avg_prc = average_precision_score(y_test, y_proba)

        report = classification_report(y_test, y_pred, output_dict=True)
        churn_metrics = report.get("1", report.get("1.0", {}))

        return {
            "roc_auc": round(roc_auc, 4),
            "f1": round(f1, 4),
            "avg_precision": round(avg_prc, 4),
            "precision": round(churn_metrics.get("precision", 0), 4),
            "recall": round(churn_metrics.get("recall", 0), 4),
            "accuracy": round(report.get("accuracy", 0), 4),
        }

    def plot_all(self, model, X_test, y_test, name, save=True):
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fig = plt.figure(figsize=(18, 6), facecolor=BG_DARK)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        self._plot_confusion_matrix(fig.add_subplot(gs[0]), y_test, y_pred)
        self._plot_roc_curve(fig.add_subplot(gs[1]), y_test, y_proba)
        self._plot_pr_curve(fig.add_subplot(gs[2]), y_test, y_proba)

        fig.suptitle(f"Evaluation Report — {name}",
                     color=TEXT_WHITE, fontsize=15, fontweight="bold")

        if save:
            out = REPORTS_DIR / "eval_champion.png"
            fig.savefig(out, bbox_inches="tight", dpi=180)
            print(f"Saved chart → {out}")

        plt.close()

    def _plot_confusion_matrix(self, ax, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")

    def _plot_roc_curve(self, ax, y_test, y_proba):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr)
        ax.set_title("ROC Curve")

    def _plot_pr_curve(self, ax, y_test, y_proba):
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(recall, precision)
        ax.set_title("PR Curve")


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib
    from src.preprocessing import ChurnPreprocessor

    prep = ChurnPreprocessor()
    _, X_test, _, y_test, feature_names = prep.run("data/telco_churn.csv")

    champion = joblib.load("models/champion.pkl")

    evaluator = ModelEvaluator()
    metrics = evaluator.compute(champion, X_test, y_test, "Champion")
    evaluator.plot_all(champion, X_test, y_test, "Champion")

    # 🔥 NEW: Generate JSON for frontend
    y_proba = champion.predict_proba(X_test)[:, 1]

    df = pd.DataFrame(X_test, columns=feature_names)
    df["churn_probability"] = y_proba

    high_risk = df[df["churn_probability"] > 0.6]

    out_path = REPORTS_DIR / "high_risk_customers.json"
    high_risk.to_json(out_path, orient="records", indent=2)

    print(f"\nSaved high-risk customers → {out_path}")

    print("\nChampion metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
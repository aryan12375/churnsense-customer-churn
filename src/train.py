"""
src/train.py
────────────
Trains all 5 classification models, evaluates them, logs to MLflow,
and saves the champion model.

Models:
  • Logistic Regression  (baseline, interpretable)
  • Random Forest        (ensemble bagging)
  • Gradient Boosting    (ensemble boosting — champion)
  • SVM                  (margin-based baseline)
  • ANN                  (PyTorch deep learning — challenger)

Usage:
    python src/train.py
    python src/train.py --skip-ann      # skip the PyTorch model
"""

import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.preprocessing import ChurnPreprocessor
from src.evaluate import ModelEvaluator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

def build_model_registry() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class ChurnTrainer:
    def __init__(self, skip_ann: bool = False):
        self.skip_ann = skip_ann
        self.evaluator = ModelEvaluator()
        self.results: dict[str, dict] = {}
        self.champion_name: str = ""
        self.champion_model = None

    def run(self, csv_path: str = "data/telco_churn.csv") -> None:
        prep = ChurnPreprocessor(random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test, feature_names = prep.run(csv_path)

        mlflow.set_experiment("ChurnSense")

        models = build_model_registry()

        for name, model in models.items():
            self._train_and_log(name, model, X_train, X_test, y_train, y_test, feature_names)

        if not self.skip_ann:
            self._train_ann(X_train, X_test, y_train, y_test, feature_names)

        self._select_champion()
        self._plot_comparison()
        self._save_champion_metadata(feature_names)

        logger.info(f"\n🏆  Champion: {self.champion_name}")
        for name, r in sorted(self.results.items(), key=lambda x: -x[1]["roc_auc"]):
            logger.info(f"   {name:<25} ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")

    # ──────────────────────────────────────────────────────────────────────────

    def _train_and_log(self, name, model, X_train, X_test, y_train, y_test, feature_names):
        logger.info(f"Training: {name}")
        t0 = time.time()

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            elapsed = time.time() - t0

            metrics = self.evaluator.compute(model, X_test, y_test, name)
            metrics["train_time_s"] = round(elapsed, 2)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.sklearn.log_model(model, name.replace(" ", "_"))

        self.results[name] = metrics

        joblib.dump(model, MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl")
        logger.info(f"  → ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  ({elapsed:.1f}s)")

    def _train_ann(self, X_train, X_test, y_train, y_test, feature_names):
        try:
            from src.ann_model import ANNChurnModel
            logger.info("Training: ANN (PyTorch)")
            t0 = time.time()

            ann = ANNChurnModel(input_dim=X_train.shape[1])
            ann.fit(X_train, y_train)
            elapsed = time.time() - t0

            metrics = self.evaluator.compute(ann, X_test, y_test, "ANN")
            metrics["train_time_s"] = round(elapsed, 2)
            self.results["ANN"] = metrics
            logger.info(f"  → ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  ({elapsed:.1f}s)")

        except Exception as e:
            logger.warning(f"ANN training failed (PyTorch not available?): {e}")

    def _select_champion(self) -> None:
        self.champion_name = max(self.results, key=lambda n: self.results[n]["roc_auc"])
        champ_path = MODELS_DIR / f"{self.champion_name.replace(' ', '_').lower()}.pkl"
        if champ_path.exists():
            self.champion_model = joblib.load(champ_path)
            # Also save as canonical champion
            joblib.dump(self.champion_model, MODELS_DIR / "champion.pkl")

    def _plot_comparison(self) -> None:
        """
        Generates a polished model comparison chart.
        """
        names = list(self.results.keys())
        roc_scores = [self.results[n]["roc_auc"] for n in names]
        f1_scores  = [self.results[n]["f1"] for n in names]

        # ── Style ────────────────────────────────────────────────────────────
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0F1117")

        palette_base  = "#1C2130"
        palette_highlight = "#6C63FF"
        champion_color    = "#00E5A0"

        champion_idx = roc_scores.index(max(roc_scores))

        for ax, scores, title, ylabel in zip(
            axes,
            [roc_scores, f1_scores],
            ["ROC-AUC Score", "F1-Score"],
            ["ROC-AUC", "F1-Score"],
        ):
            ax.set_facecolor("#0F1117")
            colors = [
                champion_color if i == champion_idx else palette_highlight
                for i in range(len(names))
            ]
            bars = ax.barh(names, scores, color=colors, height=0.55, zorder=2)

            # Value labels
            for bar, score in zip(bars, scores):
                ax.text(
                    score + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", ha="left",
                    color="white", fontsize=11, fontweight="bold",
                )

            ax.set_xlim(0.5, 1.0)
            ax.set_xlabel(ylabel, color="#A0A8B8", fontsize=12)
            ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=14)
            ax.tick_params(colors="#A0A8B8", labelsize=11)
            ax.xaxis.set_tick_params(color="#A0A8B8")
            for spine in ax.spines.values():
                spine.set_color("#2A3045")
            ax.set_axisbelow(True)
            ax.xaxis.grid(True, color="#1C2130", linewidth=0.8, zorder=0)

        legend_patches = [
            mpatches.Patch(color=champion_color, label="Champion model"),
            mpatches.Patch(color=palette_highlight, label="Challenger models"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=2,
                   frameon=False, fontsize=11,
                   labelcolor="white", bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            "ChurnSense — Model Performance Comparison",
            color="white", fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        out = REPORTS_DIR / "model_comparison.png"
        fig.savefig(out, bbox_inches="tight", dpi=180, facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved comparison chart → {out}")

    def _save_champion_metadata(self, feature_names: list[str]) -> None:
        meta = {
            "champion": self.champion_name,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "metrics": self.results.get(self.champion_name, {}),
            "feature_count": len(feature_names),
            "all_results": self.results,
        }
        path = MODELS_DIR / "champion_metadata.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved metadata → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ChurnSense models")
    parser.add_argument("--data", default="data/telco_churn.csv", help="Path to CSV")
    parser.add_argument("--skip-ann", action="store_true", help="Skip PyTorch ANN training")
    args = parser.parse_args()

    trainer = ChurnTrainer(skip_ann=args.skip_ann)
    trainer.run(csv_path=args.data)


if __name__ == "__main__":
    main()

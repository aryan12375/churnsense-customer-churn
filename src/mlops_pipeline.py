"""
src/mlops_pipeline.py
─────────────────────
Production MLOps retraining pipeline.

Features:
  1. Population Stability Index (PSI) — detects feature drift vs training baseline
  2. Automatic retrain trigger — fires when PSI > threshold on key features
  3. Model versioning — timestamped .pkl files + metadata JSON
  4. Champion promotion gate — new model must beat champion on F1 before deploy
  5. Scheduled runs via cron or manual trigger

Usage:
    # Check for drift only
    python src/mlops_pipeline.py --check-drift --data data/new_data.csv

    # Full retrain
    python src/mlops_pipeline.py --retrain --data data/new_data.csv

    # Force retrain regardless of drift
    python src/mlops_pipeline.py --retrain --force
"""

import argparse
import json
import logging
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
VERSIONS_DIR = Path("models/versions")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
VERSIONS_DIR.mkdir(exist_ok=True)

PSI_WARNING_THRESHOLD = 0.10   # Mild drift
PSI_RETRAIN_THRESHOLD = 0.20   # Severe drift — trigger retrain


# ──────────────────────────────────────────────────────────────────────────────
# Population Stability Index
# ──────────────────────────────────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-8,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI < 0.10  → no significant change (safe)
    PSI 0.10–0.20 → moderate change (monitor)
    PSI > 0.20  → significant change (retrain recommended)

    Parameters
    ----------
    expected : array — reference distribution (training data)
    actual   : array — new data distribution
    n_bins   : int   — number of percentile bins
    eps      : float — small value to avoid log(0)
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Normalise to proportions
    exp_pct = (expected_counts / (len(expected) + eps)) + eps
    act_pct = (actual_counts   / (len(actual)   + eps)) + eps

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


# ──────────────────────────────────────────────────────────────────────────────
# Drift Monitor
# ──────────────────────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Loads the training baseline distribution and compares it to new data.
    """

    def __init__(self, baseline_path: str = "models/feature_baseline.json"):
        self.baseline_path = Path(baseline_path)
        self.baseline: dict | None = None

    def save_baseline(self, X_train: np.ndarray, feature_names: list[str]) -> None:
        """Called once after initial training to save the reference distribution."""
        baseline = {}
        for i, name in enumerate(feature_names):
            col = X_train[:, i]
            baseline[name] = {
                "mean":  float(np.mean(col)),
                "std":   float(np.std(col)),
                "p10":   float(np.percentile(col, 10)),
                "p25":   float(np.percentile(col, 25)),
                "p50":   float(np.percentile(col, 50)),
                "p75":   float(np.percentile(col, 75)),
                "p90":   float(np.percentile(col, 90)),
                "values": col.tolist(),  # store raw for PSI
            }
        self.baseline_path.write_text(json.dumps(baseline, indent=2))
        logger.info(f"Feature baseline saved → {self.baseline_path}")

    def check_drift(
        self,
        X_new: np.ndarray,
        feature_names: list[str],
        top_n: int = 10,
    ) -> dict:
        """
        Compare new data distribution to baseline using PSI.

        Returns a dict with PSI per feature and an overall recommendation.
        """
        if not self.baseline_path.exists():
            logger.warning("No baseline found. Run with --save-baseline first.")
            return {"status": "no_baseline", "psi_scores": {}}

        baseline = json.loads(self.baseline_path.read_text())

        psi_scores = {}
        for i, name in enumerate(feature_names[:top_n]):
            if name not in baseline:
                continue
            expected = np.array(baseline[name]["values"])
            actual   = X_new[:, i]
            psi_scores[name] = round(compute_psi(expected, actual), 4)

        # Sort by PSI descending
        psi_scores = dict(sorted(psi_scores.items(), key=lambda x: -x[1]))

        max_psi     = max(psi_scores.values()) if psi_scores else 0.0
        n_warning   = sum(1 for v in psi_scores.values() if v >= PSI_WARNING_THRESHOLD)
        n_critical  = sum(1 for v in psi_scores.values() if v >= PSI_RETRAIN_THRESHOLD)

        if max_psi >= PSI_RETRAIN_THRESHOLD:
            recommendation = "retrain"
            status = "drift_detected"
        elif max_psi >= PSI_WARNING_THRESHOLD:
            recommendation = "monitor"
            status = "mild_drift"
        else:
            recommendation = "ok"
            status = "stable"

        result = {
            "status":          status,
            "max_psi":         round(max_psi, 4),
            "n_warning_features":  n_warning,
            "n_critical_features": n_critical,
            "recommendation":  recommendation,
            "psi_scores":      psi_scores,
            "checked_at":      datetime.utcnow().isoformat() + "Z",
        }

        # Log summary
        logger.info(f"Drift check: status={status}  max_PSI={max_psi:.4f}  recommendation={recommendation}")
        for feat, psi in list(psi_scores.items())[:5]:
            flag = "⚠️ " if psi >= PSI_WARNING_THRESHOLD else "  "
            logger.info(f"  {flag}{feat:<40} PSI={psi:.4f}")

        # Save report
        out = REPORTS_DIR / "drift_report.json"
        out.write_text(json.dumps(result, indent=2))

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Model Versioning
# ──────────────────────────────────────────────────────────────────────────────

class ModelVersioner:
    """
    Manages versioned model artifacts.
    Each version = timestamped .pkl + metadata JSON.
    """

    @staticmethod
    def promote_to_champion(
        new_model_path: Path,
        new_metrics: dict,
        old_metrics: dict | None = None,
    ) -> bool:
        """
        Promote new model to champion if it beats or matches the current one.

        Returns True if promotion succeeded.
        """
        new_f1  = new_metrics.get("f1", 0)
        old_f1  = old_metrics.get("f1", 0) if old_metrics else 0

        if old_metrics is None or new_f1 >= old_f1:
            # Archive current champion before replacing
            current = Path("models/champion.pkl")
            if current.exists():
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                archive = VERSIONS_DIR / f"champion_{ts}.pkl"
                shutil.copy(current, archive)
                logger.info(f"Archived old champion → {archive}")

            shutil.copy(new_model_path, Path("models/champion.pkl"))
            logger.info(f"New champion promoted — F1: {old_f1:.4f} → {new_f1:.4f}")
            return True
        else:
            logger.info(
                f"New model did NOT beat champion — F1: {new_f1:.4f} < {old_f1:.4f}. Keeping current champion."
            )
            return False

    @staticmethod
    def list_versions() -> list[dict]:
        versions = []
        for p in sorted(VERSIONS_DIR.glob("champion_*.pkl")):
            meta_path = p.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            versions.append({"path": str(p), "metadata": meta})
        return versions


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

class RetrainingPipeline:

    def __init__(self):
        self.drift_monitor = DriftMonitor()
        self.versioner     = ModelVersioner()

    def check_drift_only(self, csv_path: str) -> dict:
        from src.preprocessing import ChurnPreprocessor
        prep = ChurnPreprocessor()
        _, X_test, _, _, feature_names = prep.run(csv_path)
        return self.drift_monitor.check_drift(X_test, feature_names)

    def run(self, csv_path: str, force: bool = False) -> None:
        from src.preprocessing import ChurnPreprocessor
        from src.train import ChurnTrainer
        from src.evaluate import ModelEvaluator

        logger.info("── MLOps Retraining Pipeline ─────────────────────────")

        # 1. Check drift
        prep = ChurnPreprocessor()
        _, X_test, _, y_test, feature_names = prep.run(csv_path)
        drift = self.drift_monitor.check_drift(X_test, feature_names)

        if not force and drift["recommendation"] == "ok":
            logger.info("No significant drift detected. Skipping retrain. Use --force to override.")
            return

        if drift["recommendation"] == "retrain" or force:
            logger.info("Drift threshold exceeded — initiating retrain...")

            # 2. Retrain
            trainer = ChurnTrainer(skip_ann=True)
            trainer.run(csv_path=csv_path)

            # 3. Compare
            old_meta_path = Path("models/champion_metadata.json")
            old_metrics   = None
            if old_meta_path.exists():
                old_meta    = json.loads(old_meta_path.read_text())
                old_metrics = old_meta.get("metrics", {})

            new_model_path = Path(f"models/{trainer.champion_name.replace(' ', '_').lower()}.pkl")
            new_metrics    = trainer.results.get(trainer.champion_name, {})

            # 4. Promote if better
            promoted = self.versioner.promote_to_champion(new_model_path, new_metrics, old_metrics)

            # 5. Save drift + retrain record
            record = {
                "retrained_at":  datetime.utcnow().isoformat() + "Z",
                "trigger":       "drift" if not force else "manual",
                "drift_status":  drift["status"],
                "max_psi":       drift["max_psi"],
                "promoted":      promoted,
                "new_metrics":   new_metrics,
                "old_metrics":   old_metrics,
            }
            (REPORTS_DIR / "last_retrain.json").write_text(json.dumps(record, indent=2))
            logger.info("── Retraining pipeline complete ──────────────────────")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@(lambda f: f)
def _parse_args():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChurnSense MLOps pipeline")
    parser.add_argument("--data",          default="data/telco_churn.csv")
    parser.add_argument("--check-drift",   action="store_true", help="Only check drift, no retrain")
    parser.add_argument("--retrain",       action="store_true", help="Run full retrain pipeline")
    parser.add_argument("--force",         action="store_true", help="Force retrain even if no drift")
    parser.add_argument("--list-versions", action="store_true", help="List archived model versions")
    args = parser.parse_args()

    pipeline = RetrainingPipeline()

    if args.list_versions:
        versions = ModelVersioner.list_versions()
        print(f"\n{len(versions)} archived versions:")
        for v in versions:
            print(f"  {v['path']}")

    elif args.check_drift:
        result = pipeline.check_drift_only(args.data)
        print(f"\nDrift status: {result['status']}  (max PSI={result['max_psi']:.4f})")
        print(f"Recommendation: {result['recommendation']}")

    elif args.retrain:
        pipeline.run(csv_path=args.data, force=args.force)

    else:
        parser.print_help()

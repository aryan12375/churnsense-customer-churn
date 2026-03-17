"""
src/preprocessing.py
────────────────────
Full preprocessing pipeline for the Telco Customer Churn dataset.

Steps:
  1. Load & validate raw CSV
  2. Clean: fix dtypes, impute missing values
  3. Feature engineering: tenure bins, charge ratios, service count
  4. Encode: one-hot for categoricals, StandardScaler for numerics
  5. Class imbalance: SMOTE on training split only
  6. Persist: preprocessed arrays + fitted transformers for inference

Usage:
    from src.preprocessing import ChurnPreprocessor
    prep = ChurnPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = prep.run("data/telco_churn.csv")
"""

import os
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE     = 0.20
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Churn"

# Features we know exist in the Telco dataset
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    # Engineered columns added later:
    "tenure_group",
]

ENGINEERED_NUMERIC = [
    "charge_per_tenure",   # MonthlyCharges / tenure (risk proxy)
    "service_count",       # total add-on services subscribed
    "is_high_value",       # TotalCharges > 75th percentile
]


class ChurnPreprocessor:
    """
    Stateful preprocessor. Fit on train, transform on test.
    Persists the fitted ColumnTransformer and scaler for inference.
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.smote = SMOTE(random_state=random_state, k_neighbors=5)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        csv_path: str,
        apply_smote: bool = True,
        save_artifacts: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Full preprocessing run.

        Returns
        -------
        X_train, X_test, y_train, y_test, feature_names
        """
        logger.info("── Starting preprocessing pipeline ──────────────────")

        df = self._load(csv_path)
        df = self._clean(df)
        df = self._engineer_features(df)

        X, y = self._split_xy(df)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, stratify=y
        )
        logger.info(f"Split → train={len(X_train_raw):,}  test={len(X_test_raw):,}")
        logger.info(f"Churn rate → train={y_train.mean():.2%}  test={y_test.mean():.2%}")

        X_train, X_test = self._fit_transform(X_train_raw, X_test_raw)

        if apply_smote:
            X_train, y_train = self._apply_smote(X_train, y_train)

        if save_artifacts:
            self._save_artifacts()

        logger.info("── Preprocessing complete ────────────────────────────")
        return X_train, X_test, y_train, y_test, self.feature_names

    def transform_single(self, record: dict) -> np.ndarray:
        """Transform a single customer record dict for live inference."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call run() first.")
        df = pd.DataFrame([record])
        df = self._clean(df)
        df = self._engineer_features(df)
        X, _ = self._split_xy(df, has_target=False)
        return self.pipeline.transform(X)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self, csv_path: str) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{path}'.\n"
                "Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
            )
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} cols from {path.name}")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop customerID — not predictive
        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)

        # TotalCharges arrives as object due to spaces for new customers
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Impute: TotalCharges NaN → MonthlyCharges × tenure (first bill)
        mask = df["TotalCharges"].isna()
        if mask.any():
            df.loc[mask, "TotalCharges"] = (
                df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"].clip(lower=1)
            )
            logger.info(f"Imputed {mask.sum()} missing TotalCharges values")

        # Target encoding: Yes → 1, No → 0
        if TARGET_COL in df.columns:
            df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

        # SeniorCitizen is already int (0/1), but keep it consistent
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Tenure groups — captures non-linear loyalty effect
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 60, 72],
            labels=["0-12", "13-24", "25-48", "49-60", "61+"],
        ).astype(str)

        # 2. Charge per tenure month — cost intensity signal
        df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"].clip(lower=1))

        # 3. Service count — breadth of engagement
        service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        for col in service_cols:
            if col in df.columns:
                df[col + "_bin"] = (df[col] == "Yes").astype(int)
        bin_cols = [c for c in df.columns if c.endswith("_bin")]
        df["service_count"] = df[bin_cols].sum(axis=1)
        df.drop(columns=bin_cols, inplace=True)

        # 4. High-value customer flag
        threshold = df["TotalCharges"].quantile(0.75)
        df["is_high_value"] = (df["TotalCharges"] >= threshold).astype(int).astype(str)

        logger.info("Engineered 4 new features: tenure_group, charge_per_tenure, service_count, is_high_value")
        return df

    def _split_xy(
        self, df: pd.DataFrame, has_target: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        if has_target and TARGET_COL in df.columns:
            return df.drop(columns=[TARGET_COL]), df[TARGET_COL]
        return df, None

    def _fit_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        # Determine actual feature sets from data
        numeric_cols = [c for c in NUMERIC_FEATURES + ENGINEERED_NUMERIC if c in X_train.columns]
        numeric_cols = [c for c in numeric_cols if X_train[c].dtype in ["float64", "int64", "float32"]]

        categorical_cols = [
            c for c in X_train.columns if c not in numeric_cols
        ]

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
        )

        self.pipeline = preprocessor

        X_train_enc = preprocessor.fit_transform(X_train)
        X_test_enc  = preprocessor.transform(X_test)

        # Reconstruct feature names
        num_names = numeric_cols
        cat_names = list(
            preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols)
        )
        self.feature_names = num_names + cat_names

        logger.info(
            f"Encoded → {X_train_enc.shape[1]} features "
            f"({len(num_names)} numeric, {len(cat_names)} categorical)"
        )
        return X_train_enc, X_test_enc

    def _apply_smote(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        before = y_train.sum()
        X_res, y_res = self.smote.fit_resample(X_train, y_train)
        after = y_res.sum()
        logger.info(
            f"SMOTE → churn samples {int(before):,} → {int(after):,}  "
            f"(total {len(y_res):,} training rows)"
        )
        return X_res, y_res

    def _save_artifacts(self) -> None:
        path = MODELS_DIR / "preprocessor.pkl"
        joblib.dump(
            {"pipeline": self.pipeline, "feature_names": self.feature_names},
            path,
        )
        logger.info(f"Saved preprocessor → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prep = ChurnPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = prep.run("data/telco_churn.csv")
    print(f"\nReady — X_train shape: {X_train.shape}, features: {len(feature_names)}")

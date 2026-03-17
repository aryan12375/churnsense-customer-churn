"""
src/ann_model.py
────────────────
PyTorch Artificial Neural Network for churn prediction.

Architecture:
  Input → [512 → BN → ReLU → Dropout(0.35)]
        → [256 → BN → ReLU → Dropout(0.30)]
        → [128 → BN → ReLU → Dropout(0.25)]
        → [64  → BN → ReLU → Dropout(0.20)]
        → [1   → Sigmoid]

Features:
  • Batch Normalisation for stable training on tabular data
  • Residual-style skip connection from layer 1 → layer 3
  • Early stopping (patience=15 epochs)
  • Class-weighted BCE loss to handle imbalance
  • Cosine annealing LR scheduler
  • Sklearn-compatible API (fit / predict / predict_proba)

Usage:
    from src.ann_model import ANNChurnModel
    ann = ANNChurnModel(input_dim=X_train.shape[1])
    ann.fit(X_train, y_train)
    y_proba = ann.predict_proba(X_test)[:, 1]
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Network definition
# ──────────────────────────────────────────────────────────────────────────────

class _ChurnNet(nn.Module):
    """
    4-layer MLP with batch norm, dropout, and one skip connection.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        # Skip connection: project block1 output to 128 dims
        self.skip_proj = nn.Linear(512, 128)

        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.block4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.head = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2) + self.skip_proj(h1)   # residual skip
        h4 = self.block4(h3)
        return torch.sigmoid(self.head(h4)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Sklearn-compatible wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ANNChurnModel:
    """
    Sklearn-compatible wrapper around _ChurnNet.

    Parameters
    ----------
    input_dim      : int   — number of input features
    epochs         : int   — maximum training epochs
    batch_size     : int
    lr             : float — initial learning rate
    patience       : int   — early stopping patience
    val_fraction   : float — fraction of training data held out for early stopping
    """

    def __init__(
        self,
        input_dim: int = 100,
        epochs: int = 200,
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 15,
        val_fraction: float = 0.10,
    ):
        self.input_dim    = input_dim
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.patience     = patience
        self.val_fraction = val_fraction
        self.model: Optional[_ChurnNet] = None
        self.history: dict = {"train_loss": [], "val_loss": []}

    # ── Sklearn interface ──────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANNChurnModel":
        logger.info(f"ANN training on {DEVICE} — {X.shape[0]:,} samples, {X.shape[1]} features")

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Train/val split for early stopping
        n_val    = max(1, int(len(X_t) * self.val_fraction))
        n_train  = len(X_t) - n_val
        perm     = torch.randperm(len(X_t))
        idx_tr, idx_va = perm[:n_train], perm[n_train:]

        train_loader = DataLoader(
            TensorDataset(X_t[idx_tr], y_t[idx_tr]),
            batch_size=self.batch_size, shuffle=True,
        )
        X_val, y_val = X_t[idx_va].to(DEVICE), y_t[idx_va].to(DEVICE)

        # Class-weighted loss
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32).to(DEVICE)
        criterion  = nn.BCELoss()  # sigmoid already applied in forward

        self.model = _ChurnNet(self.input_dim).to(DEVICE)
        optimizer  = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_val_loss = float("inf")
        patience_cnt  = 0
        best_weights  = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                preds = self.model(Xb)
                # Manually apply pos_weight in the loss
                loss  = -(pos_weight * yb * torch.log(preds + 1e-8) +
                          (1 - yb) * torch.log(1 - preds + 1e-8)).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val)
                val_loss  = -(pos_weight * y_val * torch.log(val_preds + 1e-8) +
                              (1 - y_val) * torch.log(1 - val_preds + 1e-8)).mean().item()

            self.history["train_loss"].append(epoch_loss / len(train_loader))
            self.history["val_loss"].append(val_loss)

            if epoch % 20 == 0:
                logger.info(f"  Epoch {epoch:3d}/{self.epochs} — train_loss={epoch_loss/len(train_loader):.4f}  val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                patience_cnt  = 0
                best_weights  = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        torch.save(self.model.state_dict(), MODELS_DIR / "ann_weights.pt")
        logger.info("ANN training complete — weights saved")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            proba_pos = self.model(X_t).cpu().numpy()
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "input_dim":    self.input_dim,
            "epochs":       self.epochs,
            "batch_size":   self.batch_size,
            "lr":           self.lr,
            "patience":     self.patience,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("ANNChurnModel is not fitted. Call .fit() first.")

    def plot_training_history(self) -> None:
        """Plot train vs val loss curve."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0F1117")
        ax.set_facecolor("#161B27")

        ax.plot(self.history["train_loss"], color="#6C63FF", linewidth=2, label="Train loss")
        ax.plot(self.history["val_loss"],   color="#00E5A0", linewidth=2, label="Val loss")
        ax.set_xlabel("Epoch", color="#808898")
        ax.set_ylabel("Loss",  color="#808898")
        ax.set_title("ANN Training History", color="#EAEAEA", fontweight="bold")
        ax.tick_params(colors="#808898")
        for spine in ax.spines.values():
            spine.set_color("#232838")
        legend = ax.legend(frameon=True)
        legend.get_frame().set_facecolor("#1C2130")
        for t in legend.get_texts():
            t.set_color("#EAEAEA")

        plt.tight_layout()
        out = Path("reports") / "ann_training_history.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Saved → {out}")

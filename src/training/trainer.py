"""
src/training/trainer.py
───────────────────────
Training loop, evaluation, and checkpointing for the ScreenMind MLP.

Public API:
    make_loaders(data, task, batch_size)  → train_loader, val_loader, test_loader
    make_criterion(task, y_train)         → loss function (handles class imbalance)
    train(model, ..., wandb_run=None)     → history dict  {train_loss, val_loss per epoch}
    evaluate_clf(model, loader)           → dict of classification metrics
    evaluate_reg(model, loader)           → dict of regression metrics

Loss functions:
    WeightedMSELoss  — MSE weighted by target magnitude (Milestone 5 Variant C)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── Dataset ──────────────────────────────────────────────────────────────────

class BRFSSDataset(Dataset):
    """Wraps numpy feature + target arrays as a PyTorch Dataset.

    A PyTorch Dataset must implement:
      __len__  → total number of examples
      __getitem__(i) → (features_tensor, label_tensor) for example i

    DataLoader uses these to pull mini-batches during training.

    Args:
        X: float32 array of shape (n_samples, n_features).
        y: float32 array of shape (n_samples,).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_loaders(
    data: dict,
    task: str,
    batch_size: int = 512,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from the processed data dict.

    Args:
        data:       Output of load_processed() — dict with X_* and y_* arrays.
        task:       'clf' (uses y_clf_*) or 'reg' (uses y_reg_*).
        batch_size: Number of examples per mini-batch. 512 is a good default
                    for this dataset size — large enough to use GPU efficiently,
                    small enough to introduce useful gradient noise.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    y_key = "y_clf" if task == "clf" else "y_reg"

    train_ds = BRFSSDataset(data["X_train"], data[f"{y_key}_train"])
    val_ds   = BRFSSDataset(data["X_val"],   data[f"{y_key}_val"])
    test_ds  = BRFSSDataset(data["X_test"],  data[f"{y_key}_test"])

    # shuffle=True only for training — val/test order doesn't matter
    # num_workers=0: safe on Windows (no multiprocessing issues)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


class WeightedMSELoss(nn.Module):
    """MSE loss weighted by the target value.

    Standard MSELoss treats every example equally, so a prediction error of
    5 days on a person who reported 0 bad days gets the same penalty as an
    error of 5 days on a person who reported 28.  For mental health risk, we
    care much more about accuracy at the high end of the scale.

    weight(y) = 1 + y / 30
        y=0  → weight 1.0   (zero-bad-days cases, weighted normally)
        y=15 → weight 1.5
        y=30 → weight 2.0   (worst cases penalised 2× harder)

    This encourages the model to fit the high-day tail rather than collapsing
    toward the dominant zero-inflated majority.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + target / 30.0
        return (weights * (pred - target).pow(2)).mean()


def make_criterion(task: str, y_train: np.ndarray) -> nn.Module:
    """Return the appropriate loss function for the task.

    Classification: BCEWithLogitsLoss with pos_weight to handle imbalance.

        pos_weight = n_negative / n_positive ≈ 6.6
        Effect: misclassifying a truly high-risk patient costs 6.6× as much
        as misclassifying a low-risk patient.  Without this, the model learns
        to always predict "low-risk" (which is 86.8% accurate but clinically useless).

    Regression: MSELoss — penalises large errors quadratically.
    Regression (weighted): WeightedMSELoss — up-weights errors on high-day cases.

    Args:
        task:    'clf', 'reg', or 'reg_weighted'.
        y_train: Training labels — used to compute pos_weight for 'clf'.
    """
    if task == "clf":
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task == "reg_weighted":
        return WeightedMSELoss()
    else:
        return nn.MSELoss()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    checkpoint_path: str | Path,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    device: str | None = None,
    wandb_run=None,
) -> dict:
    """Train the model with early stopping and checkpoint saving.

    Early stopping: if val_loss does not improve for `patience` consecutive
    epochs, training stops and the best checkpoint is retained.

    Checkpoint saving: model weights are saved to `checkpoint_path` every time
    val_loss improves.  Only the *best* weights are kept — not every epoch.

    Args:
        model:            The MLP instance.
        train_loader:     DataLoader for training data.
        val_loader:       DataLoader for validation data.
        criterion:        Loss function from make_criterion().
        checkpoint_path:  Where to save the best model weights (.pt file).
        lr:               Adam learning rate (default 1e-3).
        max_epochs:       Maximum number of full passes over training data.
        patience:         Early stopping patience in epochs.
        device:           'cuda', 'mps', or 'cpu'. Auto-detected if None.
        wandb_run:        Optional wandb Run object (from wandb.init()). When
                          provided, train_loss and val_loss are logged each epoch
                          so you can track curves live in the W&B dashboard.

    Returns:
        history: dict with 'train_loss' and 'val_loss' lists (one value per epoch).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Training on {device}  |  max_epochs={max_epochs}  |  patience={patience}")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Best':>6}")
    print("-" * 44)

    for epoch in range(1, max_epochs + 1):

        # ── Train phase ───────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch).squeeze(1)   # (batch,)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(X_batch)

        train_loss = running_loss / len(train_loader.dataset)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_running = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch).squeeze(1)
                val_running += criterion(output, y_batch).item() * len(X_batch)

        val_loss = val_running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # ── W&B logging (no-op if wandb_run is None) ──────────────────────────
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # ── Checkpoint + early stopping ───────────────────────────────────────
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        marker = " ✓" if improved else ""
        print(f"{epoch:>6}  {train_loss:>12.5f}  {val_loss:>12.5f}  {marker}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nBest val loss: {best_val_loss:.5f}  →  saved to {checkpoint_path}")
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def _get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (all_outputs, all_labels) as numpy arrays."""
    model.eval()
    outputs_list, labels_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch.to(device)).squeeze(1)
            outputs_list.append(out.cpu().numpy())
            labels_list.append(y_batch.numpy())

    return np.concatenate(outputs_list), np.concatenate(labels_list)


def evaluate_clf(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict:
    """Evaluate a classification model.

    Returns a dict with: accuracy, precision, recall, f1, roc_auc, n_samples.

    Args:
        model:     Trained MLP with task='clf'.
        loader:    DataLoader for the evaluation split.
        device:    Device string.
        threshold: Probability threshold for positive class (default 0.5).
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    logits, labels = _get_predictions(model, loader, device)
    probs = 1 / (1 + np.exp(-logits))   # sigmoid
    preds = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    return {
        "accuracy":  accuracy_score(labels_int, preds),
        "precision": precision_score(labels_int, preds, zero_division=0),
        "recall":    recall_score(labels_int, preds, zero_division=0),
        "f1":        f1_score(labels_int, preds, zero_division=0),
        "roc_auc":   roc_auc_score(labels_int, probs),
        "n_samples": len(labels),
    }


def evaluate_reg(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """Evaluate a regression model.

    Returns a dict with: mae, rmse, r2, n_samples.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    preds, labels = _get_predictions(model, loader, device)

    return {
        "mae":       mean_absolute_error(labels, preds),
        "rmse":      float(np.sqrt(mean_squared_error(labels, preds))),
        "r2":        r2_score(labels, preds),
        "n_samples": len(labels),
    }

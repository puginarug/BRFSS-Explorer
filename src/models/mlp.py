"""
src/models/mlp.py
─────────────────
Multi-Layer Perceptron for mental health risk prediction (Milestone 4).

Works for both tasks via the `task` argument:
  - task='clf'  → binary classification (high-risk yes/no)
  - task='reg'  → regression (predict MENTHLTH days 0–30)

The output is always a raw scalar (no sigmoid or ReLU on the final layer):
  - For 'clf': BCEWithLogitsLoss applies sigmoid internally — numerically stable
  - For 'reg': MSELoss expects a raw real number

Architecture (default):
    Input(15) → [Linear → BatchNorm → ReLU → Dropout] × n_blocks → Linear(1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Configurable Multi-Layer Perceptron.

    Args:
        input_dim:   Number of input features (15 after preprocessing).
        hidden_dims: List of hidden layer widths. Each entry creates one
                     block: Linear → BatchNorm1d → ReLU → Dropout.
        dropout:     Dropout probability applied after each hidden block.
        task:        'clf' (classification) or 'reg' (regression).
                     Stored as metadata; the architecture itself is the same.

    Example:
        model = MLP(input_dim=15, hidden_dims=[128, 64], dropout=0.3, task='clf')
        logits = model(x)          # shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        task: str = "clf",
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        if task not in ("clf", "reg"):
            raise ValueError(f"task must be 'clf' or 'reg', got '{task}'")

        self.task = task

        # ── Build hidden blocks ───────────────────────────────────────────────
        # Each block: Linear → BatchNorm1d → ReLU → Dropout
        #
        # Why BatchNorm?
        #   After each linear transform the activations can shift in scale.
        #   BatchNorm re-centres them to ~N(0,1) per mini-batch, which:
        #   1. Keeps gradients healthy throughout training
        #   2. Acts as mild regularisation
        #   3. Lets us use higher learning rates without diverging
        #
        # Why Dropout?
        #   Randomly zeros out `dropout` fraction of neurons each forward pass.
        #   Forces the network not to rely on any single neuron → learns more
        #   redundant, robust representations → reduces overfitting.

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            prev_dim = hidden_dim

        # ── Output layer ─────────────────────────────────────────────────────
        # Single neuron — no activation. Let the loss function handle the rest.
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape (batch_size, input_dim).

        Returns:
            Raw output tensor of shape (batch_size, 1).
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # richer than default
        n = self.count_parameters()
        return (
            f"MLP(task='{self.task}', params={n:,})\n"
            + super().__repr__()
        )

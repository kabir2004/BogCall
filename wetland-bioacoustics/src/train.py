"""Training orchestration for WetlandBioacousticsNet.

This module implements the full supervised learning loop: forward pass,
loss computation, backward pass, optimiser step, validation, and checkpoint
management.

Training configuration summary
--------------------------------
.. list-table::
   :header-rows: 1

   * - Component
     - Choice
     - Rationale
   * - Optimiser
     - AdamW
     - Decoupled weight decay avoids the L2 scaling issue of Adam+L2.
   * - Scheduler
     - CosineAnnealingLR
     - Smooth LR decay; avoids manual plateau-detection heuristics.
   * - Loss
     - BCEWithLogitsLoss
     - Numerically stable sigmoid+BCE fusion via log-sum-exp trick.
   * - Gradient clipping
     - max_norm=1.0
     - Prevents occasional large gradient spikes from disrupting training.
   * - Checkpoint gate
     - macro-F1 improvement
     - F1 balances precision and recall; better than loss for class-imbalanced
       multi-label problems.

Validation strategy
-------------------
Validation metrics are computed by accumulating **all** predictions and
targets across batches before calling the metric functions.  This is
statistically correct: macro-averaged metrics are not linear in the sample
count, so averaging per-batch results produces a biased estimate.

Checkpoint format
-----------------
.. code-block:: python

    {
        "epoch": int,
        "model_state_dict": dict,         # torch state dict
        "optimizer_state_dict": dict,     # torch state dict
        "best_f1": float,                 # best macro-F1 seen so far
        "metrics": {                      # metrics at the checkpoint epoch
            "train_loss": float,
            "val_loss": float,
            "macro_f1": float,
            "mAP": float,
        },
        "config": dict,                   # full Config.to_dict() snapshot
    }

The embedded ``config`` dict allows exact experiment reproduction from a
checkpoint file alone, without requiring the original YAML to be present.

Usage
-----
CLI::

    python -m src.train --config configs/default.yaml --seed 42

Programmatic::

    from src.config import load_config
    from src.train import Trainer

    cfg = load_config("configs/default.yaml")
    trainer = Trainer(cfg)
    trainer.fit()

Resume training from a checkpoint::

    trainer = Trainer(cfg)
    trainer.load_checkpoint("checkpoints/best.pt")
    trainer.fit(epochs=20)  # 20 additional epochs
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.dataset import create_dataloaders
from src.metrics import macro_f1, mean_average_precision
from src.model import WetlandBioacousticsNet
from src.utils import get_device, set_seed, setup_logging

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the complete training lifecycle for WetlandBioacousticsNet.

    Constructs the model, loss, optimiser, and scheduler from a ``Config``
    object; exposes ``train_epoch``, ``validate``, and ``fit`` as the primary
    interface; and automatically saves checkpoints whenever validation
    macro-F1 improves.

    Parameters
    ----------
    config : Config | None
        Full pipeline configuration.  Uses dataclass defaults if ``None``.
    train_loader : DataLoader | None
        Pre-built training DataLoader.  If ``None``, ``create_dataloaders``
        is called internally.  Pass explicit loaders for unit-testing or
        hyperparameter sweeps where the data setup is handled externally.
    val_loader : DataLoader | None
        Pre-built validation DataLoader.  Must be provided together with
        ``train_loader`` or omitted together.
    device : torch.device | None
        Compute device.  Auto-detected (CUDA → MPS → CPU) if ``None``.

    Attributes
    ----------
    cfg : Config
        The resolved configuration object.
    device : torch.device
        Device on which the model and tensors reside.
    model : WetlandBioacousticsNet
        The neural network, moved to ``device`` during ``__init__``.
    criterion : nn.BCEWithLogitsLoss
        Binary cross-entropy loss with numerically stable sigmoid fusion.
    optimizer : AdamW
        AdamW with decoupled weight decay.
    scheduler : CosineAnnealingLR
        Cosine learning-rate schedule from ``lr`` down to ``eta_min=1e-6``.
    train_loader : DataLoader
        Training data loader (with augmentation).
    val_loader : DataLoader
        Validation data loader (no augmentation).

    Notes
    -----
    **BCEWithLogitsLoss vs BCELoss:**
    The model outputs raw logits.  ``BCEWithLogitsLoss`` accepts logits and
    internally applies the numerically stable form
    ``max(x, 0) - x·y + log(1 + exp(-|x|))``, which avoids the NaN/Inf
    values that arise when a sigmoid-saturated output is passed to
    ``log(1 - p)`` in standard BCE.

    **CosineAnnealingLR configuration:**
    ``T_max=epochs`` means the schedule completes one full cosine half-cycle
    over the full training run, decaying from ``lr`` to ``eta_min=1e-6`` by
    the final epoch.  ``scheduler.step()`` is called once per epoch (not per
    batch) because the schedule is designed for epoch-level granularity.

    **Gradient clipping:**
    ``clip_grad_norm_`` with ``max_norm=1.0`` rescales the entire gradient
    vector if its L2 norm exceeds 1.0.  This does not bias the gradient
    direction — it only scales the magnitude — and prevents occasional large
    steps from destabilising the optimiser state for Adam's second moment.
    """

    def __init__(
        self,
        config: Config | None = None,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = config or Config()
        self.device = device or get_device()
        logger.info("Trainer using device: %s", self.device)

        # ── Model ────────────────────────────────────────────────────────────
        self.model = WetlandBioacousticsNet(config=self.cfg.model).to(self.device)
        logger.info(
            "Model parameters: %d", sum(p.numel() for p in self.model.parameters())
        )

        # ── Loss ─────────────────────────────────────────────────────────────
        # BCEWithLogitsLoss: accepts raw logits, applies numerically stable
        # sigmoid+BCE internally.  reduction="mean" averages over all
        # (sample, class) pairs in the batch.
        self.criterion = nn.BCEWithLogitsLoss()

        # ── Optimiser ────────────────────────────────────────────────────────
        # AdamW decouples weight decay from the gradient update, fixing the
        # L2 regularisation behaviour compared to Adam+L2.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        # ── LR scheduler ─────────────────────────────────────────────────────
        # One cosine half-cycle over the full training run: lr → eta_min.
        # step() is called once per epoch in fit().
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.training.epochs,
            eta_min=1e-6,
        )

        # ── DataLoaders ──────────────────────────────────────────────────────
        if train_loader is not None and val_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            # Let create_dataloaders handle the full setup from config paths
            self.train_loader, self.val_loader, _ = create_dataloaders(self.cfg)

        # ── Checkpointing state ──────────────────────────────────────────────
        self._checkpoint_dir = self.cfg.paths.checkpoint_path()
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._best_f1: float = 0.0  # tracks the best macro-F1 seen so far

    # ------------------------------------------------------------------
    # Training / validation primitives
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """Execute one full pass over the training DataLoader.

        Sets the model to ``train()`` mode (enabling Dropout and BatchNorm
        training statistics), iterates all batches, and returns the mean
        training loss.

        Returns
        -------
        float
            Arithmetic mean of per-batch BCEWithLogitsLoss values.

        Notes
        -----
        ``non_blocking=True`` on the ``.to(device)`` calls overlaps the
        host-to-device data transfer with the previous batch's backward pass
        when ``pin_memory=True`` is set on the DataLoader (which it is).

        Gradient clipping is applied **after** ``loss.backward()`` but
        **before** ``optimizer.step()``.  This is the correct order: clip the
        gradient, then update parameters using the clipped gradient.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for specs, labels in self.train_loader:
            # Transfer to device with non-blocking=True to overlap H2D copy
            # with previous batch's GPU compute (only effective with pin_memory)
            specs: Tensor = specs.to(self.device, non_blocking=True)
            labels: Tensor = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(specs)                            # (B, 16)
            loss = self.criterion(logits, labels)                 # scalar
            loss.backward()

            # Clip gradient L2 norm to max_norm=1.0 before the optimiser step
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> tuple[float, float, float]:
        """Run inference on the complete validation set and compute metrics.

        Sets the model to ``eval()`` mode (disabling Dropout and using
        BatchNorm running statistics), accumulates all predictions and targets
        across batches, then computes metrics on the full concatenated tensors.

        Returns
        -------
        tuple[float, float, float]
            ``(val_loss, macro_f1_score, map_score)`` where:

            - ``val_loss``: mean BCEWithLogitsLoss over all validation batches.
            - ``macro_f1_score``: macro-averaged F1 at the configured threshold.
            - ``map_score``: mean Average Precision (threshold-free).

        Notes
        -----
        **Why accumulate before computing metrics?**
        Macro-F1 and mAP are not decomposable over batches.  Computing them
        per-batch and averaging the scalars introduces a statistical bias
        proportional to the batch size and class imbalance.  Accumulating all
        predictions first and computing metrics once is O(N) extra memory but
        statistically correct.

        ``@torch.no_grad()`` suppresses gradient tape construction, halving
        the peak memory usage compared to using ``with torch.no_grad()``.
        """
        self.model.eval()
        all_probs: list[Tensor] = []
        all_targets: list[Tensor] = []
        total_loss = 0.0
        n_batches = 0

        for specs, labels in self.val_loader:
            specs = specs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(specs)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            # Accumulate sigmoid probabilities on CPU to avoid GPU OOM on large
            # validation sets (CPU has far more addressable memory than VRAM)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(labels.cpu())

        val_loss = total_loss / max(n_batches, 1)

        # Concatenate all batches into full-dataset tensors before computing metrics
        preds = torch.cat(all_probs, dim=0)       # (N_val, 16)
        targets = torch.cat(all_targets, dim=0)   # (N_val, 16)

        f1 = macro_f1(preds, targets, threshold=self.cfg.training.detection_threshold)
        map_score = mean_average_precision(preds, targets)

        return val_loss, f1, map_score

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        filename: str = "best.pt",
    ) -> None:
        """Serialise model and optimiser state to a checkpoint file.

        Parameters
        ----------
        epoch : int
            Epoch number at time of saving (1-indexed).
        metrics : dict[str, float]
            Metric values to embed in the checkpoint for provenance.
        filename : str
            Output filename relative to ``checkpoint_dir``.

        Notes
        -----
        The full ``Config`` dict is embedded alongside the model weights so
        that the checkpoint is self-contained: ``WetlandPredictor`` can
        reconstruct the model architecture and audio pipeline from the
        checkpoint alone, without a separate YAML file.
        """
        path = self._checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_f1": self._best_f1,
                "metrics": metrics,
                "config": self.cfg.to_dict(),  # embedded for reproducibility
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model and optimiser state from a checkpoint file.

        After loading, training can be resumed by calling ``fit()`` again.
        The scheduler state is **not** restored — a fresh cosine schedule
        starts from the current learning rate.  For exact resume, manually
        advance the scheduler to the correct epoch.

        Parameters
        ----------
        path : str | Path
            Path to a ``.pt`` checkpoint produced by ``_save_checkpoint``.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist at ``path``.
        RuntimeError
            If the checkpoint was saved from a different model architecture
            (mismatched state dict keys).
        """
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._best_f1 = ckpt.get("best_f1", 0.0)
        logger.info(
            "Loaded checkpoint from %s (epoch %d, best F1 %.4f)",
            path,
            ckpt.get("epoch", -1),
            self._best_f1,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self, epochs: int | None = None) -> None:
        """Execute the full training loop for the configured number of epochs.

        For each epoch:

        1. ``train_epoch()`` — forward/backward over all training batches.
        2. ``validate()`` — accumulate predictions, compute macro-F1 and mAP.
        3. ``scheduler.step()`` — advance the cosine learning rate.
        4. Print a one-line epoch summary to stdout.
        5. If macro-F1 improved, save a checkpoint to ``checkpoints/best.pt``.

        Parameters
        ----------
        epochs : int | None
            Number of epochs to run.  Falls back to ``config.training.epochs``
            if ``None``.  Pass an explicit value to run a partial training
            session (e.g., for debugging or after ``load_checkpoint``).

        Notes
        -----
        The epoch header and rows are printed to ``stdout`` (not the logger)
        so they appear in the terminal even when the log level is WARNING.
        The ``*`` flag in the last column marks epochs where a new best
        macro-F1 was achieved and a checkpoint was saved.
        """
        n_epochs = epochs or self.cfg.training.epochs
        header = (
            f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>8}  "
            f"{'Macro F1':>8}  {'mAP':>8}  {'Time':>6}  {'*':>2}"
        )
        print(header)
        print("-" * len(header))

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch()
            val_loss, f1, map_score = self.validate()

            # Step the cosine scheduler once per epoch (not per batch)
            self.scheduler.step()

            elapsed = time.time() - t0
            improved = f1 > self._best_f1
            flag = "*" if improved else ""

            print(
                f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
                f"{f1:>8.4f}  {map_score:>8.4f}  {elapsed:>5.1f}s  {flag:>2}"
            )

            if improved:
                self._best_f1 = f1
                self._save_checkpoint(
                    epoch,
                    metrics={
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "macro_f1": f1,
                        "mAP": map_score,
                    },
                    filename="best.pt",
                )

        logger.info("Training complete — best macro F1: %.4f", self._best_f1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train WetlandBioacousticsNet from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: parse arguments, configure, and run training."""
    args = _parse_args()
    setup_logging()
    set_seed(args.seed)

    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()

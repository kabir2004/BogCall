"""Multi-label evaluation metrics for wetland species detection.

All metric functions are implemented from scratch using PyTorch tensor
operations — no sklearn, scipy, or other external metric libraries are used.

Design rationale
----------------
**Why macro averaging?**
    Macro averaging computes the metric independently per species and then
    averages, treating each species equally regardless of class frequency.
    This is the correct choice for conservation monitoring, where a rare
    species (e.g., Marsh Deer with few training recordings) is at least as
    important to detect correctly as a common one (e.g., Cicada chorus).
    Micro averaging would let common classes dominate, effectively hiding
    poor performance on rare species.

**Why compute metrics on the full validation set, not per-batch?**
    Macro-averaged metrics are not linear in the sample count.  Computing the
    macro-F1 as the average of per-batch macro-F1 values gives a biased
    estimate when batches are imbalanced (which they always are with 16
    classes and a small dataset).  ``Trainer.validate()`` accumulates all
    predictions and targets and passes them here in a single call.

**Why not threshold-based metrics for mAP?**
    Mean Average Precision (mAP) is threshold-free.  It measures ranking
    quality — how well the model orders the validation set by probability for
    each species.  This is more informative than F1 at a fixed threshold and
    allows post-training threshold tuning without re-evaluating the model.

**Handling classes with no positive examples**
    When a class has zero positive examples in the evaluation set, Average
    Precision is mathematically undefined (denominator = 0).  Such classes are
    silently excluded from the mAP average.  This follows the VOC Challenge
    convention and prevents a single empty class from pulling the mAP to zero.

Input contract
--------------
All public functions expect:

- ``preds``: sigmoid probabilities in ``[0, 1]``, shape ``(N, C)``.
  Not logits.  Apply ``torch.sigmoid`` before calling these functions.
- ``targets``: binary multi-hot ground truth, shape ``(N, C)``,
  dtype float32 or int (automatically broadcast by PyTorch).

where ``N`` = number of samples and ``C`` = number of species (16).
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from src.config import SPECIES_LIST

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _threshold_preds(preds: Tensor, threshold: float) -> Tensor:
    """Binarise probability predictions at a decision threshold.

    Parameters
    ----------
    preds : Tensor
        Sigmoid probabilities, shape ``(N, C)``, values in ``[0, 1]``.
    threshold : float
        Decision boundary.  Values ``>= threshold`` are mapped to 1.

    Returns
    -------
    Tensor
        Binary prediction tensor, shape ``(N, C)``, dtype float32.
        Float32 (not bool) so it can participate in arithmetic directly.
    """
    return (preds >= threshold).float()


def _per_class_tp_fp_fn(
    preds: Tensor, targets: Tensor, threshold: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-class confusion matrix primitives in vectorised form.

    Binarises predictions at ``threshold`` and counts true positives,
    false positives, and false negatives for all ``C`` classes simultaneously.

    Parameters
    ----------
    preds : Tensor
        Sigmoid probabilities, shape ``(N, C)``.
    targets : Tensor
        Binary ground truth, shape ``(N, C)``.
    threshold : float
        Decision threshold for binarising ``preds``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(tp, fp, fn)`` — each a 1-D tensor of shape ``(C,)`` containing
        per-class counts as float32.

    Notes
    -----
    The three quantities are derived from the element-wise product of the
    binary prediction matrix and the target matrix:

    - TP: predicted 1 AND target 1  →  ``binary × target``
    - FP: predicted 1 AND target 0  →  ``binary × (1 - target)``
    - FN: predicted 0 AND target 1  →  ``(1 - binary) × target``

    Summing along dim=0 (samples) yields the per-class count.
    """
    binary = _threshold_preds(preds, threshold)           # (N, C)
    tp = (binary * targets).sum(dim=0)                    # (C,)
    fp = (binary * (1.0 - targets)).sum(dim=0)            # (C,)
    fn = ((1.0 - binary) * targets).sum(dim=0)            # (C,)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def macro_precision(preds: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """Compute macro-averaged precision across all 16 species.

    For each species ``c``:

    .. code-block:: text

        precision_c = TP_c / (TP_c + FP_c)   if (TP_c + FP_c) > 0  else 0

    Macro precision = mean(precision_c for c in 0..15).

    Parameters
    ----------
    preds : Tensor
        Sigmoid probabilities, shape ``(N, C)``, values in ``[0, 1]``.
    targets : Tensor
        Binary multi-hot ground truth, shape ``(N, C)``.
    threshold : float
        Decision threshold for binarising ``preds``.  Default 0.5.

    Returns
    -------
    float
        Macro precision in ``[0, 1]``.

    Notes
    -----
    Classes with zero predicted positives (TP + FP = 0) contribute 0 to the
    macro average rather than being excluded.  This penalises the model for
    never predicting a species, which is the desired behaviour in a detection
    setting.
    """
    tp, fp, _ = _per_class_tp_fp_fn(preds, targets, threshold)
    denom = tp + fp
    # Use torch.where to zero-fill classes with no predicted positives
    per_class = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return float(per_class.mean())


def macro_recall(preds: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """Compute macro-averaged recall across all 16 species.

    For each species ``c``:

    .. code-block:: text

        recall_c = TP_c / (TP_c + FN_c)   if (TP_c + FN_c) > 0  else 0

    Macro recall = mean(recall_c for c in 0..15).

    Parameters
    ----------
    preds : Tensor
        Sigmoid probabilities, shape ``(N, C)``, values in ``[0, 1]``.
    targets : Tensor
        Binary multi-hot ground truth, shape ``(N, C)``.
    threshold : float
        Decision threshold for binarising ``preds``.  Default 0.5.

    Returns
    -------
    float
        Macro recall in ``[0, 1]``.

    Notes
    -----
    A class with zero true positives in the evaluation set (TP + FN = 0)
    has undefined recall.  Such classes contribute 0 rather than being
    excluded, which is conservative and correct for monitoring purposes.
    """
    tp, _, fn = _per_class_tp_fp_fn(preds, targets, threshold)
    denom = tp + fn
    per_class = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return float(per_class.mean())


def macro_f1(preds: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """Compute macro-averaged F1 score across all 16 species.

    F1 is the harmonic mean of precision and recall.  The per-class form:

    .. code-block:: text

        F1_c = 2·TP_c / (2·TP_c + FP_c + FN_c)   if denom > 0  else 0

    Macro F1 = mean(F1_c for c in 0..15).

    This is the **primary training metric** used to select the best checkpoint.

    Parameters
    ----------
    preds : Tensor
        Sigmoid probabilities, shape ``(N, C)``, values in ``[0, 1]``.
    targets : Tensor
        Binary multi-hot ground truth, shape ``(N, C)``.
    threshold : float
        Decision threshold.  Default 0.5; adjust post-training on a held-out
        set by calling this function over a grid of candidate thresholds.

    Returns
    -------
    float
        Macro F1 in ``[0, 1]``.

    Notes
    -----
    The formulation ``2·TP / (2·TP + FP + FN)`` is mathematically equivalent
    to ``2·P·R / (P + R)`` but computed in a single pass from the confusion
    matrix counts, avoiding a division by zero when P or R is individually 0.
    """
    tp, fp, fn = _per_class_tp_fp_fn(preds, targets, threshold)
    denom = 2.0 * tp + fp + fn
    per_class = torch.where(denom > 0, 2.0 * tp / denom, torch.zeros_like(tp))
    return float(per_class.mean())


def mean_average_precision(preds: Tensor, targets: Tensor) -> float:
    """Compute mean Average Precision (mAP) across all species.

    For each species ``c``:

    1. Sort all ``N`` samples by descending predicted probability.
    2. Walk down the ranked list, computing ``Precision@k`` at each
       position ``k`` where the ground truth label is positive.
    3. ``AP_c = (1 / n_pos_c) · Σ_k [Precision@k · indicator(target_k == 1)]``

    mAP = mean(AP_c) averaged over species with at least one positive example.

    Parameters
    ----------
    preds : Tensor
        Sigmoid probabilities, shape ``(N, C)``, values in ``[0, 1]``.
    targets : Tensor
        Binary multi-hot ground truth, shape ``(N, C)``.

    Returns
    -------
    float
        Mean Average Precision in ``[0, 1]``.  Returns 0.0 if no class has
        any positive examples.

    Notes
    -----
    **Threshold-free:** unlike F1, AP does not require a decision threshold.
    It rewards models that place positive examples higher in the ranking,
    capturing quality across the entire precision-recall curve.

    **Classes with no positives:** AP is undefined when ``n_pos == 0``
    (the normalising denominator is zero).  Such classes are excluded from
    the mean following the VOC Challenge convention.  This is different from
    macro-F1, which includes them as 0.

    **Implementation:** ``cumsum`` + ``arange`` computes running TP counts
    and rank positions in O(N) per class without any Python-level loops
    over samples.
    """
    n_classes = preds.shape[1]
    ap_scores: list[float] = []

    for c in range(n_classes):
        pred_c = preds[:, c]      # (N,)  probabilities for species c
        target_c = targets[:, c]  # (N,)  binary ground truth for species c

        n_pos = int(target_c.sum().item())
        if n_pos == 0:
            # AP is undefined for classes with no positive examples — skip
            continue

        # Sort samples by descending predicted probability for this class
        sorted_indices = torch.argsort(pred_c, descending=True)
        sorted_targets = target_c[sorted_indices]  # (N,) ground truth in rank order

        # Cumulative TP count at each rank position
        cumulative_tp = torch.cumsum(sorted_targets, dim=0)           # (N,)

        # Precision@k = cumulative_tp[k] / (k + 1)
        ranks = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
        precision_at_k = cumulative_tp / ranks                        # (N,)

        # AP = average of Precision@k values at positive retrieval positions
        # The indicator function selects only positions where target == 1
        ap = float((precision_at_k * sorted_targets).sum() / n_pos)
        ap_scores.append(ap)

    if not ap_scores:
        return 0.0
    return float(sum(ap_scores) / len(ap_scores))


def per_species_report(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Generate a per-species breakdown of precision, recall, F1, and support.

    Useful for post-training analysis and for identifying which species are
    hardest to detect (candidates for targeted data collection or threshold
    adjustment).

    Parameters
    ----------
    preds : Tensor
        Sigmoid probabilities, shape ``(N, C)``, values in ``[0, 1]``.
    targets : Tensor
        Binary multi-hot ground truth, shape ``(N, C)``.
    threshold : float
        Decision threshold for binarising ``preds``.  Default 0.5.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict mapping each species display name (from ``SPECIES_LIST``)
        to a sub-dict with keys:

        - ``"precision"``: TP / (TP + FP), rounded to 4 decimal places.
        - ``"recall"``:    TP / (TP + FN), rounded to 4 decimal places.
        - ``"f1"``:        2·TP / (2·TP + FP + FN), rounded to 4 decimal places.
        - ``"support"``:   total number of positive examples in ``targets``.

    Notes
    -----
    ``"support"`` is computed from ``targets.sum(dim=0)`` (positive labels),
    not from ``preds.sum(dim=0)`` (positive predictions).  It reflects the
    empirical class frequency in the evaluation set, useful for interpreting
    whether a high-F1 score is backed by enough examples to be statistically
    reliable.

    When a denominator is zero (no predictions or no positives for a class),
    the corresponding metric is set to 0.0 rather than raising an error.

    Examples
    --------
        >>> report = per_species_report(probs, targets)
        >>> report["Jabiru (Jabiru mycteria)"]
        {'precision': 0.8571, 'recall': 0.6667, 'f1': 0.7500, 'support': 12.0}
    """
    tp, fp, fn = _per_class_tp_fp_fn(preds, targets, threshold)
    support = targets.sum(dim=0)  # (C,)  total positives per class in eval set

    report: dict[str, dict[str, float]] = {}
    for idx, species in enumerate(SPECIES_LIST):
        tp_i = float(tp[idx])
        fp_i = float(fp[idx])
        fn_i = float(fn[idx])

        prec_denom = tp_i + fp_i
        rec_denom = tp_i + fn_i
        f1_denom = 2.0 * tp_i + fp_i + fn_i

        prec = tp_i / prec_denom if prec_denom > 0 else 0.0
        rec = tp_i / rec_denom if rec_denom > 0 else 0.0
        f1 = 2.0 * tp_i / f1_denom if f1_denom > 0 else 0.0

        report[species] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": float(support[idx]),
        }

    return report

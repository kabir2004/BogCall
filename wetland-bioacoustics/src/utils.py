"""Utility helpers for reproducibility, device selection, and logging.

This module provides thin wrappers around standard library and PyTorch APIs
that are used in multiple places across the pipeline.  Nothing here is
domain-specific — these helpers would be appropriate in any PyTorch project.

Module contents
---------------
``set_seed``
    Seed every source of randomness in the process so that experiments are
    exactly reproducible from a given seed value.

``get_device``
    Auto-detect the best available compute device (CUDA → MPS → CPU) and
    return a ``torch.device`` object.

``setup_logging``
    Configure the root ``logging`` logger with a consistent format, routing
    output to the console and optionally to a log file.

``count_parameters``
    Count trainable parameters in a ``nn.Module`` — a quick sanity-check
    after model construction.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import torch


def set_seed(seed: int = 42) -> None:
    """Seed all sources of randomness for fully reproducible experiments.

    Sets seeds for Python's ``random``, NumPy (if available), PyTorch CPU
    RNG, and all CUDA device RNGs.  Also disables cuDNN's non-deterministic
    algorithm selection and benchmarking mode.

    Parameters
    ----------
    seed : int
        Seed value.  Any non-negative integer is valid; 42 is the default
        by convention.

    Notes
    -----
    **``PYTHONHASHSEED``:** Python 3.3+ randomises the hash seed of strings
    and bytes by default (hash randomisation).  Setting this environment
    variable to a fixed value makes dictionary iteration and set ordering
    deterministic, which affects any code that relies on Python dict/set
    traversal order (e.g., ``csv.DictReader`` field ordering on Python 3.6
    could be affected in edge cases).

    **``cudnn.deterministic = True``:** Forces cuDNN to use deterministic
    (non-random) algorithms.  Some cuDNN kernels have non-deterministic
    behaviour (e.g., atomics in backward passes) that cannot be seeded.
    Setting this flag forces cuDNN to use slower but fully deterministic
    alternatives.

    **``cudnn.benchmark = False``:** Disables cuDNN's auto-tuner, which
    profiles different kernel implementations and caches the fastest one for
    the current input size.  The auto-tuner itself is non-deterministic and
    can change the selected kernel between runs, breaking reproducibility.
    Disabling it also avoids a slow first-batch warm-up on each run.

    **Performance trade-off:** Deterministic mode can be 10–20% slower on
    some GPU architectures.  Disable it for production runs where speed
    matters more than bit-exact reproducibility.
    """
    random.seed(seed)

    # Fix Python's hash randomisation so dict/set ordering is consistent
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Seed NumPy if it is available (it is an optional dependency)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # Seed PyTorch's CPU and CUDA RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # seeds all GPUs, not just the current one

    # Enforce deterministic cuDNN kernels at the cost of some throughput
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Auto-detect and return the best available compute device.

    Priority order: CUDA → MPS (Apple Silicon) → CPU.

    Parameters
    ----------
    prefer_cuda : bool
        If ``True`` (default) and a CUDA-capable GPU is available, return
        ``torch.device("cuda")``.  Set ``False`` to force CPU even on a
        CUDA machine (useful for debugging memory issues).

    Returns
    -------
    torch.device
        The selected device.

    Notes
    -----
    **MPS (Metal Performance Shaders):** Apple Silicon GPUs are exposed via
    the ``torch.backends.mps`` backend since PyTorch 1.12.  MPS is
    significantly faster than CPU for this model's convolution-heavy
    workload, making it a useful fallback on MacBook Pro / Mac Studio
    development machines.

    **Multi-GPU:** This function always returns the default CUDA device
    (``cuda:0``).  For multi-GPU training, wrap the model in
    ``torch.nn.DataParallel`` or ``DistributedDataParallel`` after calling
    ``get_device()``.

    Examples
    --------
        >>> device = get_device()
        >>> model.to(device)
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure the root logger with a standardised format.

    Sets up a ``StreamHandler`` that writes to ``stderr`` and, optionally,
    a ``FileHandler`` that appends to a log file.  Idempotent if called
    multiple times with the same arguments (Python's ``basicConfig`` is a
    no-op if handlers are already configured).

    Parameters
    ----------
    level : int
        Minimum log level.  Messages below this level are silently discarded.
        Common values: ``logging.DEBUG`` (10), ``logging.INFO`` (20),
        ``logging.WARNING`` (30).
    log_file : str | Path | None
        If provided, also write all log messages to this file path.  The
        parent directory is created automatically if it does not exist.

    Notes
    -----
    **Log format:** ``YYYY-MM-DD HH:MM:SS | LEVEL    | module.name | message``

    The fixed-width ``LEVEL`` field (``%-8s``) ensures columns align in the
    terminal even when mixing INFO and WARNING messages.

    **``basicConfig`` idempotency:** ``logging.basicConfig`` only takes effect
    if the root logger has no handlers.  If third-party libraries (e.g., an
    IPython kernel) have already configured the root logger before this
    function is called, the format will not be changed.  In that case,
    configure logging explicitly before importing any library.

    Examples
    --------
        >>> setup_logging(level=logging.DEBUG, log_file="logs/train.log")
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        Any PyTorch module.

    Returns
    -------
    int
        Sum of ``param.numel()`` for all parameters where
        ``param.requires_grad`` is ``True``.

    Notes
    -----
    Non-trainable parameters (e.g., frozen BatchNorm running stats, embedding
    layers with ``requires_grad=False``) are excluded.  This count is what
    the optimiser updates; it is what matters for memory budgeting.

    Examples
    --------
        >>> model = WetlandBioacousticsNet()
        >>> count_parameters(model)
        3145728   # approximate; exact value depends on base_channels
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

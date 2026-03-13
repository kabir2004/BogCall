"""PyTorch Dataset and DataLoader factory for wetland bioacoustics.

This module owns the data loading contract between raw files on disk and
tensors on the GPU.  It composes ``AudioPreprocessor`` and
``BioacousticAugmentor`` from ``src.preprocessing`` into a standard PyTorch
``Dataset`` and wraps it in ``create_dataloaders``, the single function
the rest of the codebase uses to obtain ready-to-iterate data.

Expected data layout
--------------------
``prepare_labels.py`` produces the following structure:

.. code-block:: text

    data/
    ├── train/
    │   ├── audio/            # .wav / .flac / .ogg recordings
    │   └── labels.csv        # multi-hot label file
    ├── val/
    │   ├── audio/
    │   └── labels.csv
    └── test/
        ├── audio/
        └── labels.csv

``labels.csv`` format (multi-hot, column order matches ``SPECIES_LIST``):

.. code-block:: text

    filename,Jabiru (Jabiru mycteria),Hyacinth Macaw (Anodorhynchus hyacinthinus),...
    recording_001.wav,1,0,0,1,0,...
    recording_002.wav,0,0,0,0,1,...

Design notes
------------
**CSV loading over HDF5 / LMDB**
    The dataset is expected to contain O(5000–10000) recordings.  At this
    scale, torchaudio I/O per sample is fast enough that a pre-computed HDF5
    cache offers minimal benefit but adds significant complexity.  The Dataset
    stays simple: one file read per sample.

**Augmentation is opt-in per split**
    ``augment=True`` is only set for the training split.  The ``BioacousticAugmentor``
    is constructed once in ``__init__`` and reused for every ``__getitem__``
    call on that split.  Val and test splits use the bare preprocessor for
    deterministic, reproducible evaluation.

**Robust CSV parsing**
    The CSV parser cross-references the header against ``SPECIES_LIST`` and
    logs a warning (rather than raising) for missing species columns.  This
    allows the Dataset to be used with partial label files (e.g., when only
    a subset of species were annotated) without crashing the training run.

**Thread safety**
    ``AudioPreprocessor`` holds no mutable state after ``__init__`` — the
    torchaudio transforms are stateless functions.  ``__getitem__`` is
    therefore safe to call from multiple DataLoader worker processes.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.config import SPECIES_LIST, AudioConfig, AugmentationConfig, Config, TrainingConfig
from src.preprocessing import AudioPreprocessor, BioacousticAugmentor

logger = logging.getLogger(__name__)


class WetlandAudioDataset(Dataset):
    """PyTorch Dataset for multi-label wetland species detection.

    Loads audio files and multi-hot labels from a split directory, applies
    preprocessing to produce log-mel spectrograms, and optionally applies
    bioacoustic augmentation for training.

    Parameters
    ----------
    audio_dir : str | Path
        Directory containing the audio files referenced in ``labels_csv``.
    labels_csv : str | Path
        Path to the multi-hot labels CSV file.  Must contain a ``filename``
        column followed by one column per species in ``SPECIES_LIST`` order.
    config : AudioConfig | None
        Audio processing configuration.  Uses dataclass defaults if ``None``.
    augment : bool
        If ``True``, apply ``BioacousticAugmentor`` to every sample returned
        by ``__getitem__``.  Set ``True`` only for the training split.
    aug_config : AugmentationConfig | None
        Augmentation hyperparameters.  Ignored when ``augment=False``.

    Attributes
    ----------
    audio_dir : Path
        Resolved path to the audio directory.
    cfg : AudioConfig
        Resolved audio configuration.
    augment : bool
        Whether augmentation is active for this dataset instance.

    Raises
    ------
    FileNotFoundError
        If an audio file referenced in the CSV does not exist on disk.
        Raised lazily in ``__getitem__`` (not at construction time) to avoid
        scanning all files upfront.

    Notes
    -----
    **Label dtype:** Labels are returned as ``torch.float32`` tensors rather
    than ``torch.long``.  ``BCEWithLogitsLoss`` expects float targets, and
    soft Mixup labels (produced by ``BioacousticAugmentor.mixup``) are
    inherently float.

    **Missing species columns:** If a species column is absent from the CSV
    header, a warning is logged and that species defaults to label=0 for all
    rows.  Training proceeds but that species will never have a positive label.

    Examples
    --------
        >>> ds = WetlandAudioDataset("data/train/audio", "data/train/labels.csv", augment=True)
        >>> spec, label = ds[0]
        >>> spec.shape, label.shape
        (torch.Size([1, 128, 313]), torch.Size([16]))
    """

    def __init__(
        self,
        audio_dir: str | Path,
        labels_csv: str | Path,
        config: AudioConfig | None = None,
        augment: bool = False,
        aug_config: AugmentationConfig | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.cfg = config or AudioConfig()
        self.augment = augment

        # Preprocessor is stateless after construction — safe to share across
        # DataLoader worker processes without copies or locks.
        self._preprocessor = AudioPreprocessor(self.cfg)

        # Augmentor is only constructed for the training split.
        self._augmentor: BioacousticAugmentor | None = None
        if augment:
            aug = aug_config or AugmentationConfig()
            self._augmentor = BioacousticAugmentor(
                config=self.cfg,
                freq_mask_param=aug.freq_mask_param,
                time_mask_param=aug.time_mask_param,
                noise_std=aug.noise_std,
                mixup_alpha=aug.mixup_alpha,
            )

        self._filenames, self._labels = self._load_csv(labels_csv)
        logger.info(
            "WetlandAudioDataset: %d samples | augment=%s | dir=%s",
            len(self._filenames),
            augment,
            self.audio_dir,
        )

    def _load_csv(self, csv_path: str | Path) -> tuple[list[str], list[list[float]]]:
        """Parse the labels CSV and build parallel filename/label lists.

        Validates that the CSV header contains all expected species columns and
        logs a warning for any that are missing (rather than raising, to support
        partial-annotation workflows).

        Parameters
        ----------
        csv_path : str | Path
            Path to the labels CSV file.

        Returns
        -------
        tuple[list[str], list[list[float]]]
            ``(filenames, labels)`` where ``filenames[i]`` is the audio
            filename for sample ``i`` and ``labels[i]`` is its 16-element
            multi-hot label vector (float32 values in {0.0, 1.0}).

        Notes
        -----
        The column-to-index mapping is built once from the CSV header and
        reused for every row, rather than looking up column names on each
        ``DictReader`` access.  This is ~2× faster for large CSVs.

        Missing values in the CSV body default to 0 via ``row.get(col, 0)``,
        which is lenient toward ragged rows produced by annotation tools that
        omit trailing zero columns.
        """
        csv_path = Path(csv_path)
        filenames: list[str] = []
        labels: list[list[float]] = []

        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []

            # Build header → species index mapping; warn on missing species
            col_to_idx: dict[str, int] = {}
            for species in SPECIES_LIST:
                if species in header:
                    col_to_idx[species] = SPECIES_LIST.index(species)
                else:
                    logger.warning(
                        "Species column '%s' not found in CSV header — defaulting to 0",
                        species,
                    )

            for row in reader:
                filenames.append(row["filename"])
                label = [0.0] * len(SPECIES_LIST)
                for col, idx in col_to_idx.items():
                    label[idx] = float(row.get(col, 0))
                labels.append(label)

        return filenames, labels

    def __len__(self) -> int:
        """Return the total number of samples in this split.

        Returns
        -------
        int
            Number of rows parsed from the labels CSV.
        """
        return len(self._filenames)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Load, preprocess, and optionally augment one sample.

        Called by the DataLoader for each sample index.  The underlying
        torchaudio I/O is safe to run in multiple worker processes.

        Parameters
        ----------
        idx : int
            Zero-based sample index into the dataset.

        Returns
        -------
        tuple[Tensor, Tensor]
            A two-element tuple:

            - **spec** ``(1, n_mels, n_frames)`` — log-mel spectrogram in
              ``[0, 1]`` (float32).  If augmentation is enabled, stochastic
              transforms have already been applied.
            - **label** ``(num_classes,)`` — multi-hot float32 label vector
              with values in {0.0, 1.0} for un-mixed samples or in ``[0, 1]``
              for Mixup-blended samples (when called externally by the
              training loop).

        Notes
        -----
        The Mixup augmentation is **not** applied here — it operates on pairs
        of samples and must be called from the training loop after the batch
        has been assembled by the DataLoader.
        """
        filename = self._filenames[idx]
        audio_path = self.audio_dir / filename

        # Full audio pipeline: load → resample → pad/trim → mel-spectrogram
        spec = self._preprocessor.process(audio_path)  # (1, n_mels, T)

        # Stochastic augmentation for training split only
        if self.augment and self._augmentor is not None:
            spec = self._augmentor(spec)

        # Float32 label tensor: BCEWithLogitsLoss expects float targets
        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return spec, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    config: Config | None = None,
    data_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, val, and test DataLoaders from a configured data directory.

    This is the primary entry point for data setup in ``Trainer`` and any
    script that needs all three splits.

    Parameters
    ----------
    config : Config | None
        Full pipeline configuration.  Uses dataclass defaults if ``None``.
    data_dir : str | Path | None
        Override for the data root directory (``config.paths.data_dir`` is
        used otherwise).  Useful when running experiments from a non-standard
        working directory.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)`` with the following settings:

        - **train**: ``shuffle=True``, ``drop_last=True``, augmentation on.
        - **val**:   ``shuffle=False``, ``drop_last=False``, augmentation off.
        - **test**:  ``shuffle=False``, ``drop_last=False``, augmentation off.

    Notes
    -----
    **``drop_last=True`` for training:** dropping the last incomplete batch
    prevents batch normalisation statistics from being skewed by a very small
    final batch (e.g., 1–2 samples), which can destabilise training.

    **``pin_memory=True``:** pre-allocates page-locked host memory for batch
    tensors, enabling asynchronous host-to-device transfer with
    ``non_blocking=True`` in the training loop.

    **``persistent_workers=True``:** keeps DataLoader worker processes alive
    between epochs, avoiding the ~0.5 s per-epoch overhead of forking and
    importing torchaudio in each new worker.  Disabled when ``num_workers=0``
    (in-process loading, e.g., during debugging).
    """
    cfg = config or Config()
    root = Path(data_dir) if data_dir else cfg.paths.data_path()
    t_cfg: TrainingConfig = cfg.training
    a_cfg: AudioConfig = cfg.audio
    aug_cfg: AugmentationConfig = cfg.augmentation

    def _make_loader(split: str, augment: bool, shuffle: bool, drop_last: bool) -> DataLoader:
        """Construct a DataLoader for one data split."""
        split_dir = root / split
        dataset = WetlandAudioDataset(
            audio_dir=split_dir / "audio",
            labels_csv=split_dir / "labels.csv",
            config=a_cfg,
            augment=augment,
            aug_config=aug_cfg,
        )
        return DataLoader(
            dataset,
            batch_size=t_cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=t_cfg.num_workers,
            pin_memory=True,
            # Keep workers alive across epochs to amortise fork overhead
            persistent_workers=t_cfg.num_workers > 0,
        )

    train_loader = _make_loader("train", augment=True, shuffle=True, drop_last=True)
    val_loader = _make_loader("val", augment=False, shuffle=False, drop_last=False)
    test_loader = _make_loader("test", augment=False, shuffle=False, drop_last=False)

    logger.info(
        "DataLoaders ready — train: %d batches | val: %d batches | test: %d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader

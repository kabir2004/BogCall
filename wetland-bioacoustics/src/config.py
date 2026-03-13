"""Configuration module for the wetland bioacoustics species detector.

This module is the **single source of truth** for every tuneable value in the
pipeline.  All other modules import their constants from here — nothing is
hard-coded elsewhere.

Design
------
Configuration is structured as a hierarchy of Python dataclasses (``AudioConfig``,
``ModelConfig``, ``TrainingConfig``, ``AugmentationConfig``, ``PathsConfig``)
collected under a top-level ``Config`` container.  The hierarchy mirrors the
sections of ``configs/default.yaml``, so YAML keys map 1-to-1 to dataclass
fields.  ``load_config()`` deserialises the YAML and applies any programmatic
overrides via a deep-merge strategy, returning a fully-typed ``Config`` object.

Usage
-----
Load from YAML (typical):

    >>> from src.config import load_config
    >>> cfg = load_config("configs/default.yaml")
    >>> cfg.audio.sample_rate
    32000

Override individual values at runtime (e.g., from a sweep script):

    >>> cfg = load_config(
    ...     "configs/default.yaml",
    ...     overrides={"training": {"lr": 3e-4, "batch_size": 64}},
    ... )

Use defaults without a YAML file (useful in tests):

    >>> from src.config import Config
    >>> cfg = Config()  # all defaults

Access the species registry:

    >>> from src.config import SPECIES_LIST, SPECIES_TO_IDX
    >>> SPECIES_TO_IDX["Jabiru (Jabiru mycteria)"]
    0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Species registry — 16 target species
# ---------------------------------------------------------------------------
# Order is significant: it defines the column order in labels.csv and the
# index of each logit in the model output vector.  Never reorder without
# rebuilding the dataset labels and retraining the model.

SPECIES_LIST: list[str] = [
    # ── Birds (8) ───────────────────────────────────────────────────────────
    # Pantanal flagship species; vocally distinctive and ecologically important
    "Jabiru (Jabiru mycteria)",
    "Hyacinth Macaw (Anodorhynchus hyacinthinus)",
    "Rufescent Tiger-Heron (Tigrisoma lineatum)",
    "Bare-faced Curassow (Crax fasciolata)",
    "Chestnut-bellied Guan (Penelope ochrogaster)",
    "Great Potoo (Nyctibius grandis)",
    "Ringed Kingfisher (Megaceryle torquata)",
    "Screaming Piha (Lipaugus vociferans)",
    # ── Amphibians (3) ──────────────────────────────────────────────────────
    # Chorus species; tend to produce dense overlapping calls at night
    "Cane Toad (Rhinella marina)",
    "Boana Treefrog (Boana raniceps)",
    "Leptodactylus fuscus",
    # ── Mammals (3) ─────────────────────────────────────────────────────────
    # Low-frequency callers; require fmin=50 Hz to capture fundamental
    "Black Howler Monkey (Alouatta caraya)",
    "Giant Otter (Pteronura brasiliensis)",
    "Marsh Deer (Blastocerus dichotomus)",
    # ── Reptiles (1) ────────────────────────────────────────────────────────
    "Yacare Caiman (Caiman yacare)",
    # ── Insects (1) ─────────────────────────────────────────────────────────
    # Broadband chorus; aggregated at genus level because species-level
    # acoustic separation of Cicadidae in the Pantanal is not reliably possible
    "Cicada chorus (Cicadidae spp.)",
]

NUM_SPECIES: int = len(SPECIES_LIST)  # 16 — used throughout as a compile-time constant

# Bidirectional index lookups, built once at import time
SPECIES_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(SPECIES_LIST)}
IDX_TO_SPECIES: dict[int, str] = {idx: name for idx, name in enumerate(SPECIES_LIST)}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig:
    """Audio processing parameters shared by preprocessing, dataset, and inference.

    All values are chosen specifically for the target species vocal range in
    Brazilian wetland environments.  Changing any value invalidates existing
    spectrograms and requires a full re-run of the preprocessing pipeline.

    Parameters
    ----------
    sample_rate : int
        Target sample rate in Hz.  32 kHz captures the full vocal range of all
        16 target species, including cicada harmonics up to ~15 kHz, without
        the storage cost of 44.1 kHz or 48 kHz.
    clip_duration : int
        Duration of each training clip in seconds.  5 s provides enough
        temporal context for phrase-level bird calls while keeping GPU memory
        manageable at batch size 32.
    n_mels : int
        Number of mel frequency bins.  128 bins gives sufficient frequency
        resolution to separate species with adjacent dominant frequencies
        (e.g., Great Potoo ~200–800 Hz vs. Screaming Piha ~3–4 kHz).
    n_fft : int
        FFT window size in samples.  1024 samples = ~32 ms at 32 kHz,
        balancing frequency resolution (Δf = 32000/1024 ≈ 31 Hz) against
        time resolution.
    hop_length : int
        Step size between successive STFT frames in samples.  512 = 50%
        overlap, producing ~16 ms temporal resolution.  Combined with
        clip_duration=5 s this gives ≈313 frames per spectrogram.
    fmin : float
        Minimum frequency for the mel filterbank in Hz.  50 Hz is chosen to
        capture the fundamental frequency of caiman bellows (~50–80 Hz) and
        howler monkey roars (~80–400 Hz) that sit below the standard 80 Hz
        floor used in speech processing.
    fmax : float
        Maximum frequency for the mel filterbank in Hz.  15 kHz captures
        upper harmonics of bird song and cicada calls while staying well below
        the Nyquist frequency of 16 kHz at 32 kHz sample rate.
    top_db : float
        Dynamic range ceiling for AmplitudeToDB normalisation.  80 dB clips
        the quietest mel energy bands, suppressing the digital noise floor of
        field recorders without discarding meaningful low-amplitude calls.

    Attributes
    ----------
    n_samples : int
        Total number of waveform samples in one clip (read-only property).
    n_frames : int
        Number of time frames in the resulting mel-spectrogram (read-only property).

    Notes
    -----
    ``n_frames`` is computed as ``ceil(n_samples / hop_length) + 1``, which
    matches the torchaudio ``MelSpectrogram`` output for ``center=True``
    (the default).  The model uses ``AdaptiveAvgPool2d`` in its head, so it
    can handle small deviations in ``n_frames`` at inference time.
    """

    sample_rate: int = 32000
    clip_duration: int = 5
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    fmin: float = 50.0
    fmax: float = 15000.0
    top_db: float = 80.0

    @property
    def n_samples(self) -> int:
        """Total number of samples in one clip (``sample_rate * clip_duration``)."""
        return self.sample_rate * self.clip_duration

    @property
    def n_frames(self) -> int:
        """Approximate number of time frames in one mel-spectrogram.

        Returns
        -------
        int
            ``ceil(n_samples / hop_length) + 1``, matching torchaudio centre-padded STFT.
        """
        import math

        return math.ceil(self.n_samples / self.hop_length) + 1


@dataclass
class ModelConfig:
    """Model architecture parameters for ``WetlandBioacousticsNet``.

    Parameters
    ----------
    base_channels : int
        Number of output channels from the stem convolution.  The three
        residual stages expand the channel count multiplicatively:
        base → 2×base → 4×base → 8×base (default: 32→64→128→256).
        Increasing ``base_channels`` grows parameter count roughly as its
        square; doubling it approximately quadruples total parameters.
    dropout : float
        Dropout probability applied at both positions in the classifier head.
        0.3 is chosen as a modest regulariser; the SE blocks provide implicit
        regularisation by suppressing irrelevant frequency channels.

    Notes
    -----
    The final feature dimension feeding the classifier is always
    ``base_channels * 8``.  With the default of 32 this gives 256 features,
    sufficient to represent 16 species with diversity to spare.
    """

    base_channels: int = 32
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop.

    Parameters
    ----------
    epochs : int
        Number of complete passes over the training set.  50 epochs with
        CosineAnnealingLR provides a warm decay to the minimum learning rate
        by the final epoch.
    batch_size : int
        Number of samples per gradient-update step.  32 fits comfortably in
        8 GB GPU VRAM for the default architecture.
    lr : float
        Peak learning rate for AdamW.  1e-3 is the standard AdamW starting
        point for training-from-scratch CNN models.
    weight_decay : float
        L2 regularisation coefficient.  1e-4 is a conservative value that
        prevents overfitting without restricting learning speed.
    num_workers : int
        Number of background processes for the DataLoader.  Set to 0 to
        disable multiprocessing (useful for debugging).
    detection_threshold : float
        Sigmoid probability cutoff used when converting model outputs to
        binary present/absent decisions.  0.5 is the neutral threshold;
        lower values increase recall at the cost of precision.

    Notes
    -----
    ``detection_threshold`` is used during validation to compute macro-F1 but
    does not affect training loss (BCEWithLogitsLoss is threshold-free).
    Adjust post-training by operating on the saved probability outputs rather
    than by retraining.
    """

    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    detection_threshold: float = 0.5


@dataclass
class AugmentationConfig:
    """Parameters for bioacoustic data augmentation.

    Each parameter controls one augmentation that is applied stochastically
    during training only.  See ``BioacousticAugmentor`` in
    ``src.preprocessing`` for the full implementation and ecological rationale.

    Parameters
    ----------
    freq_mask_param : int
        Maximum width (in mel bins) of a SpecAugment frequency mask.
        20 bins ≈ 15% of the 128-bin axis — wide enough to challenge the
        model without hiding most of the useful signal.
    time_mask_param : int
        Maximum width (in frames) of a SpecAugment time mask.
        40 frames ≈ 640 ms at the default hop length — enough to cover a
        typical inter-call gap.
    noise_std : float
        Standard deviation of additive Gaussian noise (applied to normalised
        [0, 1] spectrograms).  0.005 is sub-perceptual to the human ear but
        meaningful as a regulariser.
    mixup_alpha : float
        Concentration parameter for the Beta(α, α) distribution used by
        Mixup.  α=0.4 produces a distribution peaked near 0 and 1 with a
        moderate probability of blending ratios near 0.5, giving soft labels
        without completely destroying individual sample identity.
    """

    freq_mask_param: int = 20
    time_mask_param: int = 40
    noise_std: float = 0.005
    mixup_alpha: float = 0.4


@dataclass
class PathsConfig:
    """Filesystem path configuration.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``train/``, ``val/``, and ``test/``
        subdirectories produced by ``scripts/prepare_labels.py``.
    checkpoint_dir : str
        Directory where ``Trainer`` writes ``.pt`` checkpoint files.
        Created automatically if it does not exist.
    log_dir : str
        Directory for log files when ``setup_logging`` is called with a
        ``log_file`` argument.  Created automatically if needed.
    """

    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def data_path(self) -> Path:
        """Return ``data_dir`` as a resolved ``Path`` object."""
        return Path(self.data_dir)

    def checkpoint_path(self) -> Path:
        """Return ``checkpoint_dir`` as a resolved ``Path`` object."""
        return Path(self.checkpoint_dir)

    def log_path(self) -> Path:
        """Return ``log_dir`` as a resolved ``Path`` object."""
        return Path(self.log_dir)


@dataclass
class Config:
    """Top-level configuration container.

    Aggregates all sub-configs into a single object that is passed through the
    pipeline.  Prefer passing ``Config`` to individual sub-configs so that
    future additions to any section do not require updating every call site.

    Parameters
    ----------
    audio : AudioConfig
        Audio processing configuration.
    model : ModelConfig
        Model architecture configuration.
    training : TrainingConfig
        Training hyperparameter configuration.
    augmentation : AugmentationConfig
        Data augmentation configuration.
    paths : PathsConfig
        Filesystem path configuration.

    Notes
    -----
    ``to_dict()`` is used when persisting the config inside a checkpoint file,
    enabling exact experiment reproduction from a saved ``.pt`` file without
    requiring the original YAML to be present at inference time.

    Examples
    --------
    Load from YAML:

        >>> cfg = load_config("configs/default.yaml")

    Override at runtime without a YAML file:

        >>> cfg = Config()
        >>> cfg.training.lr = 3e-4  # direct mutation is fine within one script
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict:
        """Serialise the full configuration to a plain nested dictionary.

        Returns
        -------
        dict
            Nested dict matching the structure of ``configs/default.yaml``.
            Suitable for JSON/YAML serialisation and checkpoint embedding.
        """
        import dataclasses

        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------


def load_config(yaml_path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load and validate configuration from a YAML file with optional overrides.

    Parameters
    ----------
    yaml_path : str | Path | None
        Path to a YAML configuration file.  If ``None``, dataclass defaults
        are used for all values.
    overrides : dict | None
        Nested dictionary of override values merged on top of the YAML before
        constructing the dataclasses.  Supports partial sub-trees, e.g.::

            overrides={"training": {"lr": 3e-4}}

        Deep-merges into the YAML rather than replacing the entire section.

    Returns
    -------
    Config
        Fully-typed, validated configuration object.

    Raises
    ------
    FileNotFoundError
        If ``yaml_path`` is specified but the file does not exist.

    Notes
    -----
    The ``hasattr`` guard in each dataclass constructor call silently drops
    unknown keys from the YAML, so future YAML additions do not break older
    code.  The inverse (code field not present in YAML) falls back to the
    dataclass default.

    Examples
    --------
    Typical experiment load:

        >>> cfg = load_config("configs/default.yaml")

    Learning-rate sweep without modifying the YAML:

        >>> cfg = load_config(
        ...     "configs/default.yaml",
        ...     overrides={"training": {"lr": 5e-4}},
        ... )

    Reconstruct config from a checkpoint dict:

        >>> ckpt = torch.load("checkpoints/best.pt")
        >>> cfg = load_config(overrides=ckpt["config"])
    """
    raw: dict = {}

    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with yaml_path.open("r") as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("Loaded config from %s", yaml_path)

    if overrides:
        _deep_merge(raw, overrides)

    # Extract each section, defaulting to an empty dict so missing sections
    # fall back to dataclass defaults without KeyError.
    audio_raw = raw.get("audio", {})
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    augmentation_raw = raw.get("augmentation", {})
    paths_raw = raw.get("paths", {})

    return Config(
        audio=AudioConfig(**{k: v for k, v in audio_raw.items() if hasattr(AudioConfig, k)}),
        model=ModelConfig(**{k: v for k, v in model_raw.items() if hasattr(ModelConfig, k)}),
        training=TrainingConfig(
            **{k: v for k, v in training_raw.items() if hasattr(TrainingConfig, k)}
        ),
        augmentation=AugmentationConfig(
            **{k: v for k, v in augmentation_raw.items() if hasattr(AugmentationConfig, k)}
        ),
        paths=PathsConfig(**{k: v for k, v in paths_raw.items() if hasattr(PathsConfig, k)}),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place.

    Nested dicts are merged at each level; scalar values (including lists) are
    replaced wholesale.  This preserves unrelated YAML keys when only a partial
    override is supplied.

    Parameters
    ----------
    base : dict
        The dict to mutate (typically the raw YAML contents).
    override : dict
        Override values to layer on top of *base*.
    """
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

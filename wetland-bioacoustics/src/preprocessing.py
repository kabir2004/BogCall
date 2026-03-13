"""Audio loading, mel-spectrogram extraction, and bioacoustic augmentation.

This module converts raw audio files into the normalised log-mel spectrograms
expected by ``WetlandBioacousticsNet``, and applies training-time augmentations
designed specifically for passive acoustic monitoring data.

Module responsibilities
-----------------------
``AudioPreprocessor``
    Stateful object that wraps ``torchaudio`` transforms.  Created once per
    process (e.g., inside the Dataset) and reused for every sample to avoid
    re-allocating the ``MelSpectrogram`` filter bank on each call.

``BioacousticAugmentor``
    Stochastic augmentation pipeline.  Applies a random subset of transforms
    to a pre-computed spectrogram.  Mixup is intentionally a ``@staticmethod``
    because it requires two samples and must be orchestrated by the training
    loop, not the Dataset.

Pipeline for a single training sample
--------------------------------------
::

    audio file (.wav / .flac / .ogg)
          │
    torchaudio.load()          → (C, T_raw) at original sample rate
          │
    mean channels              → (1, T_raw)  mono
          │
    T.Resample()               → (1, T_raw') at 32 kHz  (if needed)
          │
    pad_or_trim()              → (1, 160000) exactly 5 seconds
          │
    T.MelSpectrogram()         → (1, 128, T)  power mel-spectrogram
          │
    T.AmplitudeToDB()          → (1, 128, T)  log scale, clipped at top_db
          │
    min-max normalise          → (1, 128, T)  ∈ [0, 1]
          │
    BioacousticAugmentor       → (1, 128, T)  ∈ [0, 1]  (training only)
          │
    WetlandBioacousticsNet     → (16,)  logits

Notes on padding strategy
--------------------------
Short recordings are **repeat-padded** (tiled), not zero-padded.  Zero-padding
inserts a hard silent segment into the spectrogram.  The model learns to
associate this silence signature with low-confidence predictions and
generalises poorly when species calls happen to fall near a clip boundary —
a common occurrence in the field.  Repeat-tiling preserves the ambient noise
floor statistics of the original recording throughout the clip.

Notes on min-max normalisation
--------------------------------
Global min-max across the full spectrogram (rather than per-frequency-band
standardisation) is used because:

1. The absolute dynamic range of the spectrogram carries ecological information
   (strong vs. weak vocalisers relative to ambient noise).
2. It maps cleanly to ``[0, 1]`` without clipping, compatible with the
   ``[0, 1]``-bounded augmentation operations.
3. Per-band z-score normalisation would require computing running statistics
   over the training set, complicating the preprocessing pipeline.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from torch import Tensor

from src.config import AudioConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AudioPreprocessor
# ---------------------------------------------------------------------------


class AudioPreprocessor:
    """Convert raw audio files into normalised log-mel spectrograms.

    This class is the single entry point for all audio-to-tensor conversion.
    It wraps the torchaudio transform pipeline and exposes a clean interface
    used by both the Dataset (for training) and the WetlandPredictor (for
    inference).

    The ``MelSpectrogram`` and ``AmplitudeToDB`` transforms are constructed
    once in ``__init__`` rather than on every call, which avoids the overhead
    of reallocating the mel filterbank for each sample.

    Parameters
    ----------
    config : AudioConfig | None
        Audio processing configuration.  Uses dataclass defaults if ``None``.

    Attributes
    ----------
    cfg : AudioConfig
        Resolved audio configuration.
    _mel_transform : torchaudio.transforms.MelSpectrogram
        Stateful transform: STFT → power spectrogram → mel filterbank.
        ``power=2.0`` computes power (amplitude²) spectrograms, which are
        then converted to dB scale by ``_amplitude_to_db``.
    _amplitude_to_db : torchaudio.transforms.AmplitudeToDB
        Converts linear power to decibels and clips the dynamic range to
        ``top_db`` dB below the peak, suppressing the noise floor.

    Examples
    --------
        >>> preprocessor = AudioPreprocessor()
        >>> spec = preprocessor.process("recording.wav")
        >>> spec.shape
        torch.Size([1, 128, 313])
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.cfg = config or AudioConfig()

        # MelSpectrogram: combines STFT, power, and mel filterbank in one call.
        # power=2.0 → power spectrogram (not amplitude); AmplitudeToDB handles
        # the log conversion correctly for power inputs via stype="power".
        self._mel_transform = T.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            power=2.0,
        )

        # AmplitudeToDB with top_db clips the log-scale floor to top_db dB
        # below the loudest mel bin in each spectrogram.  This suppresses the
        # silent portions of the dynamic range without distorting the signal.
        self._amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=self.cfg.top_db)

    def load_audio(self, path: str | Path) -> Tensor:
        """Load an audio file and return a mono waveform resampled to the target rate.

        Supports all formats that torchaudio can decode on the current system
        (typically .wav, .flac, .ogg, and .mp3 with the appropriate backend).

        Parameters
        ----------
        path : str | Path
            Path to the audio file.

        Returns
        -------
        Tensor
            Mono waveform of shape ``(1, T)`` at ``config.sample_rate`` Hz,
            dtype ``float32``.

        Notes
        -----
        Stereo-to-mono conversion averages across all input channels
        (``mean(dim=0)``).  This is preferred over selecting a single channel
        because it preserves the signal energy from both channels and is
        consistent regardless of which channel carries the target vocalisation.

        Resampling uses the default ``sinc_interp_hann`` kernel from
        torchaudio, which provides good anti-aliasing when downsampling.
        """
        path = Path(path)
        waveform, sr = torchaudio.load(str(path))  # (C, T) at original sample rate

        # Multi-channel → mono: average across channels to preserve energy
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

        # Resample to target rate if the source differs
        if sr != self.cfg.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.cfg.sample_rate)
            waveform = resampler(waveform)

        return waveform  # (1, T)

    def pad_or_trim(self, waveform: Tensor) -> Tensor:
        """Enforce exactly ``sample_rate × clip_duration`` samples.

        Short waveforms are repeat-padded (tiled) rather than zero-padded.
        Long waveforms are front-truncated.

        Parameters
        ----------
        waveform : Tensor
            Mono waveform of shape ``(1, T)``.

        Returns
        -------
        Tensor
            Waveform of shape ``(1, n_samples)`` where
            ``n_samples = sample_rate × clip_duration``.

        Notes
        -----
        **Repeat-padding rationale:** Zero-padding inserts a hard silent
        segment into the spectrogram.  The model learns to associate this
        silence signature with reduced detection confidence.  This causes it to
        under-predict for clips where a vocalisation happens to fall near the
        end of the clip, which is frequent in field recordings.  Tiling the
        waveform preserves the recording's ambient noise statistics throughout
        the padded region, which is a much weaker spurious signal.

        For very short waveforms (e.g., 0.1 s), multiple tiles are stacked.
        The ``math.ceil`` + ``repeat`` + slice pattern handles this
        efficiently without any Python-level loops.
        """
        target = self.cfg.n_samples  # e.g., 32000 × 5 = 160000
        n = waveform.shape[-1]

        if n == target:
            return waveform
        if n > target:
            # Truncate from the start; the beginning of a recording may
            # contain handling noise before the sensor stabilises.
            return waveform[..., :target]

        # Repeat-pad: tile enough times, then trim the excess
        repeats = math.ceil(target / n)
        waveform = waveform.repeat(1, repeats)  # (1, n * repeats)
        return waveform[..., :target]           # (1, target)

    def to_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        """Convert a waveform to a normalised log-mel spectrogram.

        Pipeline: MelSpectrogram → AmplitudeToDB → min-max normalise to [0, 1].

        Parameters
        ----------
        waveform : Tensor
            Mono waveform of shape ``(1, T)`` at ``config.sample_rate`` Hz.

        Returns
        -------
        Tensor
            Log-mel spectrogram of shape ``(1, n_mels, n_frames)`` with values
            in ``[0, 1]`` (float32).

        Notes
        -----
        **Min-max normalisation** is computed globally across the full
        spectrogram (scalar min/max, not per-frequency-band statistics).
        This is intentional — see the module-level docstring for rationale.

        When ``max_val == min_val`` (e.g., a constant or silent waveform),
        the result is set to all zeros rather than dividing by zero.  The model
        is expected to output near-zero probabilities for a blank spectrogram.
        """
        mel = self._mel_transform(waveform)       # (1, n_mels, T)  linear power
        log_mel = self._amplitude_to_db(mel)      # (1, n_mels, T)  decibels

        # Global min-max normalisation → [0, 1]
        min_val = log_mel.min()
        max_val = log_mel.max()
        if max_val > min_val:
            log_mel = (log_mel - min_val) / (max_val - min_val)
        else:
            # Degenerate case: constant waveform — return blank spectrogram
            log_mel = torch.zeros_like(log_mel)

        return log_mel  # (1, n_mels, n_frames)

    def process(self, path: str | Path) -> Tensor:
        """Full preprocessing pipeline for a single audio clip.

        Convenience method composing ``load_audio`` → ``pad_or_trim`` →
        ``to_mel_spectrogram`` in the correct order.

        Parameters
        ----------
        path : str | Path
            Path to a .wav / .flac / .ogg audio file.

        Returns
        -------
        Tensor
            Log-mel spectrogram of shape ``(1, n_mels, n_frames)`` in
            ``[0, 1]``, ready for the model input layer.
        """
        waveform = self.load_audio(path)
        waveform = self.pad_or_trim(waveform)
        return self.to_mel_spectrogram(waveform)

    def process_segments(self, path: str | Path, overlap: float = 0.5) -> list[Tensor]:
        """Segment a long field recording and return per-segment mel-spectrograms.

        Implements a sliding-window approach where the window length equals
        ``clip_duration`` seconds and advances by ``hop = clip_duration × (1 - overlap)``
        seconds per step.  The final window is padded with ``pad_or_trim`` if
        the recording does not divide evenly.

        Parameters
        ----------
        path : str | Path
            Path to a long audio file (any duration ≥ 1 sample).
        overlap : float
            Fractional overlap between consecutive windows.  Must be in
            ``[0, 1)``.  With ``overlap=0.5`` and ``clip_duration=5``, the
            window advances 2.5 seconds per step, giving approximately
            ``(total_duration / 2.5) - 1`` segments for long recordings.

        Returns
        -------
        list[Tensor]
            Ordered list of log-mel spectrograms, each of shape
            ``(1, n_mels, n_frames)`` in ``[0, 1]``.  Never empty — a
            recording shorter than ``clip_duration`` returns exactly one
            padded segment.

        Notes
        -----
        The loop terminates on the segment whose *start* exceeds
        ``total_samples - 1``, but the final segment (where ``end >=
        total_samples``) is still processed because ``pad_or_trim`` handles
        the trailing underrun.  This ensures the last few seconds of any
        recording are always included.

        In ``WetlandPredictor``, per-segment probabilities are subsequently
        max-pooled across all segments to produce a single recording-level
        score.
        """
        waveform = self.load_audio(path)  # (1, T_total)
        total_samples = waveform.shape[-1]
        clip_len = self.cfg.n_samples
        hop = int(clip_len * (1.0 - overlap))
        hop = max(hop, 1)  # guard against overlap=1.0 or numerical edge cases

        segments: list[Tensor] = []
        start = 0
        while start < total_samples:
            end = start + clip_len
            chunk = waveform[..., start:end]     # (1, clip_len) or shorter at tail
            chunk = self.pad_or_trim(chunk)      # always (1, clip_len)
            segments.append(self.to_mel_spectrogram(chunk))
            if end >= total_samples:
                break  # processed the final (possibly padded) segment
            start += hop

        return segments


# ---------------------------------------------------------------------------
# BioacousticAugmentor
# ---------------------------------------------------------------------------


class BioacousticAugmentor:
    """Stochastic augmentation pipeline for mel-spectrograms.

    Each augmentation is applied independently with its own probability during
    a ``__call__`` invocation.  The pipeline is designed for passive acoustic
    monitoring data where:

    - Recording conditions vary (wind, rain, sensor orientation).
    - Species vocalisations can occur at any point within a clip.
    - Multiple species produce overlapping broadband signals.

    Augmentations are applied **only during training**.  The Dataset
    instantiates this class only when ``augment=True``.  Mixup is a
    ``@staticmethod`` because it requires two samples and cannot be applied
    within ``__getitem__`` — it must be orchestrated by the training loop
    after batch assembly.

    Parameters
    ----------
    config : AudioConfig | None
        Audio configuration (currently unused but stored for potential future
        frequency-range-aware masking).
    freq_mask_param : int
        Maximum width of a frequency mask in mel bins.  ``F ~ Uniform(0, F_max)``
        where ``F_max = freq_mask_param`` per SpecAugment (Park et al., 2019).
    time_mask_param : int
        Maximum width of a time mask in frames.
    noise_std : float
        Standard deviation of Gaussian noise added to the normalised [0,1]
        spectrogram.  Values above ~0.02 start to audibly distort the signal;
        0.005 is sub-perceptual but still a meaningful regulariser.
    mixup_alpha : float
        Beta distribution concentration for Mixup.  ``alpha=0.4`` yields a
        distribution peaked near 0 and 1 with moderate probability mass around
        0.5, producing soft labels without completely erasing sample identity.

    Attributes
    ----------
    cfg : AudioConfig
        Audio configuration.
    freq_mask_param, time_mask_param, noise_std, mixup_alpha : float
        Stored augmentation parameters (see Parameters above).

    References
    ----------
    Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation
        Method for Automatic Speech Recognition. Interspeech.

    Zhang, H., et al. (2018). Mixup: Beyond Empirical Risk Minimization. ICLR.

    Examples
    --------
        >>> aug = BioacousticAugmentor()
        >>> spec = torch.rand(1, 128, 313)
        >>> augmented = aug(spec)         # (1, 128, 313), stochastically modified
        >>> augmented.min(), augmented.max()
        (tensor(0.), tensor(1.))
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        noise_std: float = 0.005,
        mixup_alpha: float = 0.4,
    ) -> None:
        self.cfg = config or AudioConfig()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha

    def __call__(self, spec: Tensor) -> Tensor:
        """Apply a random subset of augmentations to a mel-spectrogram.

        Each augmentation is evaluated with its own independent Bernoulli draw,
        so multiple transforms can fire on the same sample.  The output is
        clamped to ``[0, 1]`` regardless of which transforms applied.

        Parameters
        ----------
        spec : Tensor
            Input spectrogram of shape ``(1, n_mels, T)`` with values in
            ``[0, 1]``.

        Returns
        -------
        Tensor
            Augmented spectrogram of shape ``(1, n_mels, T)`` clamped to
            ``[0, 1]``.

        Notes
        -----
        ``spec.clone()`` is called first to prevent in-place mutations from
        modifying the original tensor (important when the Dataset caches
        spectrograms in memory).  The clamp at the end is a safety net for
        Gaussian noise, which can push values slightly outside ``[0, 1]``.
        """
        spec = spec.clone()  # avoid mutating the cached original

        if torch.rand(1).item() < 0.5:
            spec = self._freq_mask(spec)

        if torch.rand(1).item() < 0.5:
            spec = self._time_mask(spec)

        if torch.rand(1).item() < 0.3:
            spec = self._gaussian_noise(spec)

        # Time-shift is implemented at spectrogram level (circular roll on the
        # time axis) rather than at waveform level.  This avoids an extra FFT
        # and is a reasonable approximation because the phase offset introduced
        # by the roll is irrelevant to the log-mel representation.
        if torch.rand(1).item() < 0.3:
            spec = self._time_shift(spec)

        return spec.clamp(0.0, 1.0)

    def _freq_mask(self, spec: Tensor) -> Tensor:
        """Apply SpecAugment frequency masking.

        Zeros a random contiguous band of ``f`` mel bins starting at ``f0``,
        where ``f ~ Uniform(0, freq_mask_param)`` and
        ``f0 ~ Uniform(0, n_mels - f)``.

        Parameters
        ----------
        spec : Tensor
            Shape ``(1, n_mels, T)``.

        Returns
        -------
        Tensor
            Spectrogram with one masked frequency band (same shape).

        Notes
        -----
        Simulates partial frequency occlusion caused by wind noise, rain, or
        a malfunctioning recorder frequency response.  Forcing the model to
        predict species presence from a band-incomplete spectrogram improves
        robustness to sensor variability.
        """
        n_mels = spec.shape[1]
        # Sample mask width uniformly; clamp f0 so the mask stays in-bounds
        f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
        f0 = torch.randint(0, max(n_mels - int(f), 1), (1,)).item()
        spec[:, int(f0): int(f0) + int(f), :] = 0.0
        return spec

    def _time_mask(self, spec: Tensor) -> Tensor:
        """Apply SpecAugment time masking.

        Zeros a random contiguous block of ``t`` time frames starting at
        ``t0``, where ``t ~ Uniform(0, time_mask_param)`` and
        ``t0 ~ Uniform(0, T - t)``.

        Parameters
        ----------
        spec : Tensor
            Shape ``(1, n_mels, T)``.

        Returns
        -------
        Tensor
            Spectrogram with one masked time block (same shape).

        Notes
        -----
        Simulates transient interruptions such as a large bird flying past
        the sensor, water splashing, or a momentary recording dropout.
        Masking up to 40 frames ≈ 640 ms, which covers a typical inter-call
        gap without concealing an entire vocalisation phrase.
        """
        n_frames = spec.shape[2]
        t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
        t0 = torch.randint(0, max(n_frames - int(t), 1), (1,)).item()
        spec[:, :, int(t0): int(t0) + int(t)] = 0.0
        return spec

    def _gaussian_noise(self, spec: Tensor) -> Tensor:
        """Add i.i.d. Gaussian noise to every pixel of the spectrogram.

        Parameters
        ----------
        spec : Tensor
            Shape ``(1, n_mels, T)`` in ``[0, 1]``.

        Returns
        -------
        Tensor
            Noisy spectrogram (same shape).  May momentarily exceed ``[0, 1]``
            before the final clamp in ``__call__``.

        Notes
        -----
        Simulates recorder noise floor variation across deployment sites and
        seasons.  Standard deviation of 0.005 on the normalised ``[0, 1]``
        scale is sub-perceptual but statistically meaningful as a regulariser,
        reducing the model's sensitivity to exact pixel intensity values.
        """
        noise = torch.randn_like(spec) * self.noise_std
        return spec + noise

    def _time_shift(self, spec: Tensor) -> Tensor:
        """Circularly shift the spectrogram along the time axis.

        Shift magnitude is sampled uniformly from ``[-max_shift, +max_shift]``
        where ``max_shift = int(n_frames × 0.10)`` (10% of clip length).

        Parameters
        ----------
        spec : Tensor
            Shape ``(1, n_mels, T)``.

        Returns
        -------
        Tensor
            Circularly shifted spectrogram (same shape).

        Notes
        -----
        In field recordings, species vocalisations rarely start at a clip
        boundary — they begin at an arbitrary point within the 5-second window.
        Circular time shift augments the model's invariance to call onset
        position.  Circular (rather than linear) shifting avoids introducing a
        zero-padded gap, which would be an artificial signal boundary.
        """
        n_frames = spec.shape[2]
        max_shift = max(int(n_frames * 0.10), 1)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(spec, shifts=int(shift), dims=2)

    @staticmethod
    def mixup(
        spec_a: Tensor,
        labels_a: Tensor,
        spec_b: Tensor,
        labels_b: Tensor,
        alpha: float = 0.4,
    ) -> tuple[Tensor, Tensor]:
        """Blend two spectrograms and their label vectors via Mixup.

        Samples a mixing coefficient ``λ ~ Beta(alpha, alpha)`` and linearly
        interpolates both the spectrogram and the label vector:

        .. code-block:: text

            mixed_spec   = λ · spec_a   + (1−λ) · spec_b
            mixed_labels = λ · labels_a + (1−λ) · labels_b

        This is a ``@staticmethod`` because it operates on a pair of samples
        and must be called from the training loop after batch construction.

        Parameters
        ----------
        spec_a : Tensor
            First spectrogram, shape ``(1, n_mels, T)`` in ``[0, 1]``.
        labels_a : Tensor
            First multi-hot label vector, shape ``(num_classes,)``.
        spec_b : Tensor
            Second spectrogram, shape ``(1, n_mels, T)`` in ``[0, 1]``.
        labels_b : Tensor
            Second multi-hot label vector, shape ``(num_classes,)``.
        alpha : float
            Beta distribution concentration parameter.  Larger values produce
            mixing coefficients closer to 0.5; smaller values (→0) produce
            near-identity mixtures.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(mixed_spec, mixed_labels)`` where ``mixed_spec`` is clamped to
            ``[0, 1]`` and ``mixed_labels`` contains soft float labels in
            ``[0, 1]`` rather than binary integers.

        Notes
        -----
        The mixed labels are **soft** (real-valued), not binary.  Pass them
        directly to ``BCEWithLogitsLoss``, which accepts any value in
        ``[0, 1]`` as a target.  Do not threshold before the loss.

        Mixup reduces overconfident predictions and improves calibration,
        which is especially important for rare species with few positive
        examples in the training set.
        """
        # λ from Beta(alpha, alpha) — symmetric distribution centred at 0.5
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
        mixed_spec = lam * spec_a + (1.0 - lam) * spec_b
        mixed_labels = lam * labels_a.float() + (1.0 - lam) * labels_b.float()
        return mixed_spec.clamp(0.0, 1.0), mixed_labels

"""Inference engine for WetlandBioacousticsNet.

Provides ``WetlandPredictor``, the production-facing interface for running
trained model checkpoints against new audio files.  Handles single clips,
batches, and long field recordings via a sliding-window strategy.

Architecture of a prediction call
----------------------------------
::

    audio file
        │
    AudioPreprocessor.process()        → (1, 128, 313)  mel-spectrogram
        │
    _spec_to_probs()
        ├─ unsqueeze(0)                → (1, 1, 128, 313)  batch dim
        ├─ model.predict_proba()       → (1, 16)  sigmoid probabilities
        └─ squeeze(0).cpu()            → (16,)
        │
    _probs_to_result()                 → list[{species, probability, detected}]
        │  sorted by descending probability
        ▼
    PredictionResult (16 dicts)

Long recording strategy
-----------------------
Field recorders capture hours of continuous audio.  ``predict_long_recording``
applies a sliding 5-second window with configurable overlap:

1. ``AudioPreprocessor.process_segments`` slices the waveform and produces
   one spectrogram per segment.
2. Each segment is scored independently by ``_spec_to_probs``.
3. Species probabilities are **max-pooled** across segments.

Max-pooling is the correct aggregation for presence/absence monitoring: if a
species vocalises in any 5-second window, its peak probability is the final
score.  Mean-pooling would dilute a strong 5-second detection across an entire
hour of background noise.

CLI usage
---------
::

    # Single 5-second clip
    python -m src.inference recording.wav --checkpoint checkpoints/best.pt

    # Long field recording with sliding window
    python -m src.inference recording.wav --long --overlap 0.5 \\
        --checkpoint checkpoints/best.pt --threshold 0.5 --output results.json

Type aliases
------------
``PredictionResult``
    ``list[dict[str, object]]`` — one dict per species, sorted by descending
    probability.  Each dict has keys:

    - ``"species"`` (str): species display name from ``SPECIES_LIST``.
    - ``"probability"`` (float): sigmoid score in ``[0, 1]``, rounded to 4 dp.
    - ``"detected"`` (bool): ``True`` if ``probability >= threshold``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import Tensor

from src.config import SPECIES_LIST, Config, load_config
from src.model import WetlandBioacousticsNet
from src.preprocessing import AudioPreprocessor
from src.utils import get_device, setup_logging

logger = logging.getLogger(__name__)

# Structured return type for a single-file or aggregated prediction
PredictionResult = list[dict[str, object]]


class WetlandPredictor:
    """Production inference engine wrapping a trained WetlandBioacousticsNet.

    Loads a checkpoint produced by ``Trainer``, reconstructs the model
    architecture and audio preprocessing pipeline from the embedded config,
    and exposes clean prediction methods for different input scenarios.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a ``.pt`` checkpoint file produced by ``Trainer._save_checkpoint``.
        The checkpoint must contain ``"model_state_dict"`` and ``"config"`` keys.
    config : Config | None
        Configuration override.  If ``None``, the config embedded in the
        checkpoint is used.  Providing an explicit config is useful when
        loading weights from a checkpoint that was saved with a different
        config than the current deployment environment.
    device : torch.device | None
        Compute device.  Auto-detected (CUDA → MPS → CPU) if ``None``.
    threshold : float
        Probability cutoff for the binary ``"detected"`` flag.  Does not
        affect the ``"probability"`` values, only the boolean flag.

    Attributes
    ----------
    threshold : float
        Detection probability threshold (see Parameters).
    device : torch.device
        Compute device in use.
    cfg : Config
        Resolved configuration (from checkpoint or explicit override).
    model : WetlandBioacousticsNet
        Loaded model in ``eval()`` mode.

    Notes
    -----
    **Config resolution order:**

    1. Explicit ``config`` argument (highest priority).
    2. ``"config"`` dict embedded in the checkpoint (typical case).
    3. ``Config()`` dataclass defaults (fallback for old checkpoints).

    This ordering means a deployed predictor exactly reproduces the audio
    preprocessing parameters used during training without requiring the
    original YAML file to be present at the deployment site.

    **Thread safety:** ``WetlandPredictor`` is safe to call from multiple
    threads as long as each thread uses a separate instance.  The model
    parameters are read-only after loading and ``AudioPreprocessor`` is
    stateless, but ``torch.no_grad()`` context and device tensors are not
    shared between threads.

    Examples
    --------
        >>> predictor = WetlandPredictor("checkpoints/best.pt", threshold=0.4)
        >>> results = predictor.predict("field_clip.wav")
        >>> [r for r in results if r["detected"]]
        [{'species': 'Black Howler Monkey ...', 'probability': 0.812, 'detected': True}]
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Config | None = None,
        device: torch.device | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.device = device or get_device()

        # ── Load checkpoint ──────────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        logger.info(
            "Loaded checkpoint — epoch %d, best F1 %.4f",
            ckpt.get("epoch", -1),
            ckpt.get("best_f1", float("nan")),
        )
        # Print to stdout (not logger) so it appears regardless of log level
        print(
            f"[WetlandPredictor] Checkpoint: {checkpoint_path} | "
            f"epoch={ckpt.get('epoch', '?')} | best_F1={ckpt.get('best_f1', '?'):.4f}"
        )

        # ── Resolve config (checkpoint → explicit arg → defaults) ─────────
        if config is None:
            ckpt_cfg_dict = ckpt.get("config", {})
            if ckpt_cfg_dict:
                # Reconstruct the typed Config from the embedded flat dict
                config = load_config(overrides=ckpt_cfg_dict)
            else:
                config = Config()  # fallback for legacy checkpoints without config
        self.cfg = config

        # ── Build model ──────────────────────────────────────────────────────
        self.model = WetlandBioacousticsNet(config=self.cfg.model)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()  # disable Dropout; use BN running statistics

        # ── Audio preprocessor ───────────────────────────────────────────────
        # Must use the same AudioConfig that was active during training
        # so that mel filterbank parameters match the trained weights.
        self._preprocessor = AudioPreprocessor(self.cfg.audio)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spec_to_probs(self, spec: Tensor) -> Tensor:
        """Score a single mel-spectrogram and return per-species probabilities.

        Parameters
        ----------
        spec : Tensor
            Log-mel spectrogram of shape ``(1, n_mels, T)`` in ``[0, 1]``.

        Returns
        -------
        Tensor
            Per-species sigmoid probabilities, shape ``(num_classes,)``,
            on CPU in ``[0, 1]``.

        Notes
        -----
        The spectrogram is unsqueezed to add a batch dimension before passing
        to the model, then the batch dimension is squeezed back out.
        Moving to CPU before returning avoids accumulating tensors on the GPU
        when scoring many segments in ``predict_long_recording``.
        """
        # Add batch dim: (1, n_mels, T) → (1, 1, n_mels, T)
        batch = spec.unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(batch)  # (1, num_classes)
        return probs.squeeze(0).cpu()                # (num_classes,)

    def _probs_to_result(self, probs: Tensor) -> PredictionResult:
        """Convert a probability vector to a sorted list of result dicts.

        Parameters
        ----------
        probs : Tensor
            Per-species probabilities, shape ``(num_classes,)``, in ``[0, 1]``.

        Returns
        -------
        PredictionResult
            16 dicts sorted by descending probability.  Each dict contains:
            ``"species"`` (str), ``"probability"`` (float, 4 dp), and
            ``"detected"`` (bool).

        Notes
        -----
        Results are sorted by probability so the highest-confidence detections
        appear first in both console output and JSON export, regardless of the
        position of each species in ``SPECIES_LIST``.
        """
        results = [
            {
                "species": SPECIES_LIST[i],
                "probability": round(float(probs[i]), 4),
                "detected": bool(float(probs[i]) >= self.threshold),
            }
            for i in range(len(SPECIES_LIST))
        ]
        # Sort descending by probability; ties broken by SPECIES_LIST order
        results.sort(key=lambda d: d["probability"], reverse=True)  # type: ignore[return-value]
        return results

    # ------------------------------------------------------------------
    # Public prediction methods
    # ------------------------------------------------------------------

    def predict(self, audio_path: str | Path) -> PredictionResult:
        """Predict species probabilities for a single audio clip.

        The clip is loaded, resampled if necessary, padded or trimmed to
        exactly ``clip_duration`` seconds, and converted to a mel-spectrogram
        before scoring.

        Parameters
        ----------
        audio_path : str | Path
            Path to a ``.wav``, ``.flac``, or ``.ogg`` audio file.
            Files longer than ``clip_duration`` are front-truncated.
            For long recordings, use :meth:`predict_long_recording` instead.

        Returns
        -------
        PredictionResult
            16 species dicts sorted by descending probability.
        """
        spec = self._preprocessor.process(audio_path)
        probs = self._spec_to_probs(spec)
        return self._probs_to_result(probs)

    def predict_long_recording(
        self, audio_path: str | Path, overlap: float = 0.5
    ) -> PredictionResult:
        """Predict species presence across a long field recording.

        Applies a sliding 5-second window over the full recording, scores each
        segment independently, then max-pools probabilities across all segments
        to produce a single recording-level detection score per species.

        Parameters
        ----------
        audio_path : str | Path
            Path to a ``.wav`` / ``.flac`` / ``.ogg`` file of any duration.
        overlap : float
            Fractional overlap between consecutive windows.
            Must be in ``[0, 1)``.  ``overlap=0.5`` means the window advances
            2.5 seconds per step; ``overlap=0.0`` gives non-overlapping windows.

        Returns
        -------
        PredictionResult
            16 species dicts sorted by descending probability, where each
            probability is the **maximum** across all segments.

        Notes
        -----
        **Why max-pool?**
        For presence/absence monitoring, the ecologically relevant question is:
        "Was this species present at any point during this recording?"
        Max-pooling answers that question: if a species vocalised clearly in
        one 5-second segment, its peak score is the final score.  Mean-pooling
        would dilute a strong detection across silent background segments,
        causing systematic underestimation for species that call briefly.

        If the recording file is empty or produces no segments, a zero-probability
        result is returned for all species rather than raising.
        """
        segments = self._preprocessor.process_segments(audio_path, overlap=overlap)
        logger.info(
            "predict_long_recording: %d segments from '%s'", len(segments), audio_path
        )

        if not segments:
            # Edge case: empty file or unsupported format
            return self._probs_to_result(torch.zeros(len(SPECIES_LIST)))

        # Score every segment; stack into (n_segments, num_classes) for pooling
        all_probs = torch.stack(
            [self._spec_to_probs(seg) for seg in segments], dim=0
        )

        # Max-pool across the segment axis: (n_segments, C) → (C,)
        max_probs, _ = all_probs.max(dim=0)
        return self._probs_to_result(max_probs)

    def predict_batch(self, audio_paths: list[str | Path]) -> list[PredictionResult]:
        """Predict species probabilities for a list of audio files.

        Processes each file independently in sequence.  For GPU-accelerated
        batch processing, consider calling the model directly with a stacked
        tensor of pre-computed spectrograms.

        Parameters
        ----------
        audio_paths : list[str | Path]
            List of paths to audio files.  All files are treated as
            individual 5-second clips (not long recordings).

        Returns
        -------
        list[PredictionResult]
            One ``PredictionResult`` per input file, in the same order as
            ``audio_paths``.
        """
        return [self.predict(path) for path in audio_paths]

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_results(results: PredictionResult, title: str = "Detection Results") -> None:
        """Print a formatted console report with an ASCII probability bar chart.

        Detected species (above threshold) are marked with ``✓``.  The bar
        chart uses Unicode block characters for visual clarity in terminals
        that support UTF-8.

        Parameters
        ----------
        results : PredictionResult
            Sorted list of 16 species dicts (output of any ``predict*`` method).
        title : str
            Header text displayed above the chart.
        """
        bar_width = 30  # characters for the 0–1 probability bar
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

        detected_count = sum(1 for r in results if r["detected"])
        for r in results:
            prob = float(r["probability"])  # type: ignore[arg-type]
            filled = int(prob * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            marker = " ✓" if r["detected"] else "  "
            print(f"{marker} {r['species']:<45} [{bar}] {prob:.3f}")

        print(f"\n  {detected_count}/{len(results)} species detected above threshold")
        print(f"{'=' * 60}\n")

    @staticmethod
    def to_json(results: PredictionResult, path: str | Path) -> None:
        """Export prediction results to a JSON file.

        The output JSON is a list of 16 dicts, each containing ``"species"``,
        ``"probability"``, and ``"detected"`` keys, sorted by descending
        probability (same order as the console output).

        Parameters
        ----------
        results : PredictionResult
            Sorted list of 16 species dicts.
        path : str | Path
            Destination file path.  Parent directories are created if they
            do not exist.

        Notes
        -----
        ``indent=2`` produces human-readable output and is used by default.
        For high-volume batch export where file size matters, pass the results
        to ``json.dumps`` directly without indentation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Results saved → %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WetlandBioacousticsNet inference on an audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=str, help="Path to the input audio file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config file.  If omitted, the config embedded "
            "in the checkpoint is used."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the 'detected' flag.",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help=(
            "Treat the input as a long field recording and use sliding-window "
            "inference with max-pooling across segments."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding window overlap fraction used with --long.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to export results as a JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: parse arguments, load model, and run inference."""
    setup_logging()
    args = _parse_args()

    # Resolve config: explicit YAML > checkpoint-embedded > defaults
    config: Config | None = None
    if args.config:
        config = load_config(args.config)

    predictor = WetlandPredictor(
        checkpoint_path=args.checkpoint,
        config=config,
        threshold=args.threshold,
    )

    if args.long:
        results = predictor.predict_long_recording(args.audio, overlap=args.overlap)
    else:
        results = predictor.predict(args.audio)

    predictor.print_results(results, title=f"Results: {Path(args.audio).name}")

    if args.output:
        predictor.to_json(results, args.output)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

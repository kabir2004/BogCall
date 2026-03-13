"""End-to-end smoke tests for WetlandPredictor."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torchaudio

from src.config import NUM_SPECIES, SPECIES_LIST, Config
from src.model import WetlandBioacousticsNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_dummy_wav(path: Path, duration_s: float, sr: int = 32000) -> None:
    """Save a random-noise WAV file for use in tests."""
    n = int(sr * duration_s)
    waveform = torch.randn(1, n)
    torchaudio.save(str(path), waveform, sr)


def _make_checkpoint(path: Path, config: Config) -> None:
    """Create a minimal checkpoint file from an untrained model."""
    model = WetlandBioacousticsNet(config=config.model)
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "best_f1": 0.0,
            "metrics": {},
            "config": config.to_dict(),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> Config:
    return Config()


@pytest.fixture()
def checkpoint_path(tmp_path: Path, config: Config) -> Path:
    ckpt = tmp_path / "dummy_checkpoint.pt"
    _make_checkpoint(ckpt, config)
    return ckpt


@pytest.fixture()
def audio_5s(tmp_path: Path) -> Path:
    p = tmp_path / "clip_5s.wav"
    _save_dummy_wav(p, duration_s=5.0)
    return p


@pytest.fixture()
def audio_30s(tmp_path: Path) -> Path:
    p = tmp_path / "recording_30s.wav"
    _save_dummy_wav(p, duration_s=30.0)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictSingleFile:
    def test_returns_list_of_16_dicts(
        self, checkpoint_path: Path, config: Config, audio_5s: Path
    ) -> None:
        """predict() must return exactly 16 species result dicts."""
        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict(audio_5s)
        assert len(results) == NUM_SPECIES

    def test_result_keys(
        self, checkpoint_path: Path, config: Config, audio_5s: Path
    ) -> None:
        """Each result dict must have 'species', 'probability', 'detected'."""
        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict(audio_5s)
        for r in results:
            assert "species" in r
            assert "probability" in r
            assert "detected" in r

    def test_probability_in_unit_interval(
        self, checkpoint_path: Path, config: Config, audio_5s: Path
    ) -> None:
        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict(audio_5s)
        for r in results:
            p = float(r["probability"])  # type: ignore[arg-type]
            assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"

    def test_all_species_present(
        self, checkpoint_path: Path, config: Config, audio_5s: Path
    ) -> None:
        """All 16 species must appear in the results."""
        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict(audio_5s)
        returned_species = {r["species"] for r in results}
        for species in SPECIES_LIST:
            assert species in returned_species, f"Missing species: {species}"

    def test_results_sorted_by_probability(
        self, checkpoint_path: Path, config: Config, audio_5s: Path
    ) -> None:
        """Results must be sorted in descending order of probability."""
        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict(audio_5s)
        probs = [float(r["probability"]) for r in results]  # type: ignore[arg-type]
        assert probs == sorted(probs, reverse=True)


class TestPredictLongRecording:
    def test_returns_16_dicts(
        self, checkpoint_path: Path, config: Config, audio_30s: Path
    ) -> None:
        """predict_long_recording on a 30s file should still return 16 dicts."""
        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict_long_recording(audio_30s, overlap=0.5)
        assert len(results) == NUM_SPECIES

    def test_probabilities_valid(
        self, checkpoint_path: Path, config: Config, audio_30s: Path
    ) -> None:
        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict_long_recording(audio_30s, overlap=0.5)
        for r in results:
            p = float(r["probability"])  # type: ignore[arg-type]
            assert 0.0 <= p <= 1.0


class TestPredictBatch:
    def test_batch_returns_correct_number(
        self, checkpoint_path: Path, config: Config, audio_5s: Path, audio_30s: Path
    ) -> None:
        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict_batch([audio_5s, audio_30s])
        assert len(results) == 2
        for r in results:
            assert len(r) == NUM_SPECIES


class TestToJson:
    def test_json_export(
        self, checkpoint_path: Path, config: Config, audio_5s: Path, tmp_path: Path
    ) -> None:
        import json

        from src.inference import WetlandPredictor

        predictor = WetlandPredictor(checkpoint_path, config=config)
        results = predictor.predict(audio_5s)
        out = tmp_path / "results.json"
        WetlandPredictor.to_json(results, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert len(loaded) == NUM_SPECIES

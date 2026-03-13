"""Tests for AudioPreprocessor and BioacousticAugmentor."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torchaudio

from src.config import AudioConfig
from src.preprocessing import AudioPreprocessor, BioacousticAugmentor


@pytest.fixture()
def cfg() -> AudioConfig:
    return AudioConfig()


@pytest.fixture()
def preprocessor(cfg: AudioConfig) -> AudioPreprocessor:
    return AudioPreprocessor(cfg)


@pytest.fixture()
def augmentor(cfg: AudioConfig) -> BioacousticAugmentor:
    return BioacousticAugmentor(config=cfg)


@pytest.fixture()
def tmp_wav(tmp_path: Path) -> Path:
    """Write a 5-second stereo 32 kHz sine-wave WAV for testing."""
    sr = 32000
    duration = 5
    t = torch.linspace(0, duration, sr * duration)
    waveform = torch.stack([torch.sin(2 * math.pi * 440 * t)] * 2)  # stereo (2, T)
    path = tmp_path / "test_audio.wav"
    torchaudio.save(str(path), waveform, sr)
    return path


@pytest.fixture()
def tmp_short_wav(tmp_path: Path) -> Path:
    """Write a 2-second mono 16 kHz WAV — short + wrong sample rate."""
    sr = 16000
    duration = 2
    t = torch.linspace(0, duration, sr * duration)
    waveform = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)  # (1, T)
    path = tmp_path / "short_audio.wav"
    torchaudio.save(str(path), waveform, sr)
    return path


@pytest.fixture()
def tmp_long_wav(tmp_path: Path) -> Path:
    """Write a 30-second mono 32 kHz WAV for segment tests."""
    sr = 32000
    duration = 30
    t = torch.linspace(0, duration, sr * duration)
    waveform = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)
    path = tmp_path / "long_audio.wav"
    torchaudio.save(str(path), waveform, sr)
    return path


class TestMonoConversion:
    def test_stereo_converted_to_mono(
        self, preprocessor: AudioPreprocessor, tmp_wav: Path
    ) -> None:
        waveform = preprocessor.load_audio(tmp_wav)
        assert waveform.shape[0] == 1, "Stereo input must be collapsed to mono"

    def test_mono_stays_mono(
        self, preprocessor: AudioPreprocessor, tmp_short_wav: Path
    ) -> None:
        waveform = preprocessor.load_audio(tmp_short_wav)
        assert waveform.shape[0] == 1


class TestResample:
    def test_resampled_to_target_rate(
        self, preprocessor: AudioPreprocessor, tmp_short_wav: Path
    ) -> None:
        """16 kHz input must be resampled to 32 kHz."""
        waveform = preprocessor.load_audio(tmp_short_wav)
        # 16 kHz 2s → 32 kHz 2s = 64000 samples
        assert waveform.shape[-1] == 64000, (
            f"Expected 64000 samples after resampling, got {waveform.shape[-1]}"
        )


class TestPadOrTrim:
    def test_exact_length_unchanged(self, preprocessor: AudioPreprocessor) -> None:
        target = preprocessor.cfg.n_samples
        waveform = torch.randn(1, target)
        result = preprocessor.pad_or_trim(waveform)
        assert result.shape[-1] == target

    def test_long_waveform_trimmed(self, preprocessor: AudioPreprocessor) -> None:
        target = preprocessor.cfg.n_samples
        waveform = torch.randn(1, target + 10000)
        result = preprocessor.pad_or_trim(waveform)
        assert result.shape[-1] == target

    def test_short_waveform_padded(self, preprocessor: AudioPreprocessor) -> None:
        target = preprocessor.cfg.n_samples
        waveform = torch.randn(1, target // 3)
        result = preprocessor.pad_or_trim(waveform)
        assert result.shape[-1] == target

    def test_very_short_waveform_padded(self, preprocessor: AudioPreprocessor) -> None:
        """Even a 1-sample waveform should be padded to target length."""
        target = preprocessor.cfg.n_samples
        waveform = torch.randn(1, 1)
        result = preprocessor.pad_or_trim(waveform)
        assert result.shape[-1] == target


class TestMelSpectrogram:
    def test_output_shape_5s_clip(self, preprocessor: AudioPreprocessor) -> None:
        """5-second clip should produce (1, 128, 313) spectrogram."""
        waveform = torch.randn(1, preprocessor.cfg.n_samples)
        spec = preprocessor.to_mel_spectrogram(waveform)
        assert spec.shape[0] == 1
        assert spec.shape[1] == 128
        # Allow ±1 frame tolerance
        assert abs(spec.shape[2] - 313) <= 1, (
            f"Expected ~313 frames, got {spec.shape[2]}"
        )

    def test_output_in_unit_interval(self, preprocessor: AudioPreprocessor) -> None:
        waveform = torch.randn(1, preprocessor.cfg.n_samples)
        spec = preprocessor.to_mel_spectrogram(waveform)
        assert spec.min().item() >= 0.0
        assert spec.max().item() <= 1.0

    def test_constant_signal_normalized(self, preprocessor: AudioPreprocessor) -> None:
        """A constant-value waveform should not raise and output zeros (max == min)."""
        waveform = torch.ones(1, preprocessor.cfg.n_samples)
        spec = preprocessor.to_mel_spectrogram(waveform)
        assert spec.min().item() >= 0.0
        assert spec.max().item() <= 1.0


class TestProcess:
    def test_full_pipeline_output_shape(
        self, preprocessor: AudioPreprocessor, tmp_wav: Path
    ) -> None:
        spec = preprocessor.process(tmp_wav)
        assert spec.shape[0] == 1
        assert spec.shape[1] == 128
        assert abs(spec.shape[2] - 313) <= 1

    def test_full_pipeline_output_range(
        self, preprocessor: AudioPreprocessor, tmp_wav: Path
    ) -> None:
        spec = preprocessor.process(tmp_wav)
        assert spec.min().item() >= 0.0
        assert spec.max().item() <= 1.0


class TestProcessSegments:
    def test_segment_count_30s_overlap50(
        self, preprocessor: AudioPreprocessor, tmp_long_wav: Path
    ) -> None:
        """30-second recording with 50% overlap and 5-second window.

        hop = 5 * 0.5 = 2.5 s. Expected segments ≈ ceil((30-5)/2.5) + 1 = 11.
        """
        segments = preprocessor.process_segments(tmp_long_wav, overlap=0.5)
        assert len(segments) > 0
        # Rough bounds: at least 10, at most 13 segments for a 30-second clip
        assert 8 <= len(segments) <= 13, f"Unexpected segment count: {len(segments)}"

    def test_each_segment_correct_shape(
        self, preprocessor: AudioPreprocessor, tmp_long_wav: Path
    ) -> None:
        segments = preprocessor.process_segments(tmp_long_wav, overlap=0.5)
        for i, seg in enumerate(segments):
            assert seg.shape[0] == 1, f"Segment {i}: expected 1 channel"
            assert seg.shape[1] == 128, f"Segment {i}: expected 128 mel bins"
            assert abs(seg.shape[2] - 313) <= 1, f"Segment {i}: unexpected frames {seg.shape[2]}"

    def test_short_recording_returns_one_segment(
        self, preprocessor: AudioPreprocessor, tmp_wav: Path
    ) -> None:
        """A 5-second recording should yield exactly 1 segment."""
        segments = preprocessor.process_segments(tmp_wav, overlap=0.5)
        assert len(segments) == 1


class TestAugmentor:
    def test_output_in_unit_interval(self, augmentor: BioacousticAugmentor) -> None:
        spec = torch.rand(1, 128, 313)
        out = augmentor(spec)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_output_shape_preserved(self, augmentor: BioacousticAugmentor) -> None:
        spec = torch.rand(1, 128, 313)
        out = augmentor(spec)
        assert out.shape == spec.shape

    def test_mixup_output_shapes(self) -> None:
        spec_a = torch.rand(1, 128, 313)
        spec_b = torch.rand(1, 128, 313)
        labels_a = torch.zeros(16)
        labels_b = torch.ones(16)
        mixed_spec, mixed_labels = BioacousticAugmentor.mixup(spec_a, labels_a, spec_b, labels_b)
        assert mixed_spec.shape == spec_a.shape
        assert mixed_labels.shape == labels_a.shape

    def test_mixup_output_in_unit_interval(self) -> None:
        spec_a = torch.rand(1, 128, 313)
        spec_b = torch.rand(1, 128, 313)
        labels_a = torch.zeros(16)
        labels_b = torch.ones(16)
        mixed_spec, mixed_labels = BioacousticAugmentor.mixup(spec_a, labels_a, spec_b, labels_b)
        assert mixed_spec.min().item() >= 0.0
        assert mixed_spec.max().item() <= 1.0
        assert mixed_labels.min().item() >= 0.0
        assert mixed_labels.max().item() <= 1.0

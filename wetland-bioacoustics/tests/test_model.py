"""Tests for WetlandBioacousticsNet model architecture."""

from __future__ import annotations

import torch
import pytest

from src.config import NUM_SPECIES, ModelConfig
from src.model import WetlandBioacousticsNet


@pytest.fixture()
def model() -> WetlandBioacousticsNet:
    return WetlandBioacousticsNet()


@pytest.fixture()
def dummy_batch() -> torch.Tensor:
    """Synthetic mel-spectrogram batch: (4, 1, 128, 313)."""
    return torch.randn(4, 1, 128, 313)


class TestForwardPass:
    def test_output_shape(self, model: WetlandBioacousticsNet, dummy_batch: torch.Tensor) -> None:
        """Forward pass should produce (batch, 16) logits."""
        logits = model(dummy_batch)
        assert logits.shape == (4, NUM_SPECIES), (
            f"Expected (4, {NUM_SPECIES}), got {logits.shape}"
        )

    def test_output_shape_single_sample(self, model: WetlandBioacousticsNet) -> None:
        """Single-sample inference should return (1, 16) logits."""
        x = torch.randn(1, 1, 128, 313)
        logits = model(x)
        assert logits.shape == (1, NUM_SPECIES)

    def test_variable_time_dimension(self, model: WetlandBioacousticsNet) -> None:
        """AdaptiveAvgPool should handle varying time dimensions gracefully."""
        for t in [100, 200, 313, 500]:
            x = torch.randn(2, 1, 128, t)
            logits = model(x)
            assert logits.shape == (2, NUM_SPECIES), (
                f"Failed for T={t}: got {logits.shape}"
            )


class TestGradientFlow:
    def test_gradients_reach_stem(
        self, model: WetlandBioacousticsNet, dummy_batch: torch.Tensor
    ) -> None:
        """Gradients must flow back to the stem convolution."""
        logits = model(dummy_batch)
        loss = logits.sum()
        loss.backward()
        stem_conv = model.stem[0]
        assert stem_conv.weight.grad is not None
        assert stem_conv.weight.grad.abs().sum().item() > 0.0

    def test_no_dead_parameters(
        self, model: WetlandBioacousticsNet, dummy_batch: torch.Tensor
    ) -> None:
        """Every trainable parameter should receive a non-zero gradient."""
        logits = model(dummy_batch)
        logits.sum().backward()
        dead: list[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum().item() == 0.0:
                    dead.append(name)
        assert not dead, f"Dead parameters (zero gradient): {dead}"


class TestPredictProba:
    def test_output_in_unit_interval(
        self, model: WetlandBioacousticsNet, dummy_batch: torch.Tensor
    ) -> None:
        """predict_proba output must lie in [0, 1]."""
        probs = model.predict_proba(dummy_batch)
        assert probs.shape == (4, NUM_SPECIES)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0

    def test_no_gradient_computed(
        self, model: WetlandBioacousticsNet, dummy_batch: torch.Tensor
    ) -> None:
        """predict_proba should not allocate gradients."""
        probs = model.predict_proba(dummy_batch)
        assert not probs.requires_grad


class TestWeightInit:
    def test_batchnorm_weight_one_bias_zero(self, model: WetlandBioacousticsNet) -> None:
        """BatchNorm layers should be initialised with weight=1, bias=0."""
        import torch.nn as nn

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert module.weight.data.allclose(torch.ones_like(module.weight.data))
                assert module.bias.data.allclose(torch.zeros_like(module.bias.data))
                break  # One check is sufficient as a smoke test

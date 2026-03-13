"""WetlandBioacousticsNet — custom CNN for multi-label species detection.

This module defines the complete neural network architecture used for passive
acoustic monitoring of 16 target species in Brazilian wetland environments.

Architecture overview
---------------------
The network is a custom residual CNN designed specifically for log-mel
spectrograms.  It is intentionally **not** a pretrained ImageNet backbone —
mel-spectrograms are single-channel, log-scaled, and semantically structured
along the frequency axis in a domain-specific way that is incompatible with
RGB photographic inductive biases.

::

    Input  (B, 1, 128, T)
      │
      ▼  Stem: Conv7×7/2 → BN → ReLU → MaxPool/2          → (B, 32,  32, T/4)
      ▼  Stage 1: Conv3×3/2 + 2×ResidualBlock(64)          → (B, 64,  16, T/8)
      ▼  Stage 2: Conv3×3/2 + 2×ResidualBlock(128)         → (B, 128,  8, T/16)
      ▼  Stage 3: Conv3×3/2 + 2×ResidualBlock(256)         → (B, 256,  4, T/32)
      ▼  AdaptiveAvgPool2d(1) → Flatten                    → (B, 256)
      ▼  Dropout → Linear(256, 128) → ReLU → Dropout       → (B, 128)
      ▼  Linear(128, 16)                                    → (B, 16)  raw logits

Key design choices
------------------
**Pre-activation residual blocks (He et al., 2016 v2)**
    BN→ReLU→Conv ordering (vs post-activation) improves gradient flow through
    the identity path and empirically trains more stably for smaller datasets.

**Squeeze-and-Excitation (SE) recalibration (Hu et al., 2018)**
    Each mel frequency bin corresponds to a specific acoustic band.  SE learns
    to up-weight bands containing species-characteristic vocalisations (e.g.,
    300–800 Hz for howler monkey, >10 kHz for cicada harmonics) and suppress
    broadband noise, without additional supervision beyond the multi-hot labels.

**Multi-label sigmoid output, not softmax**
    Species co-occur: a Pantanal dawn recording can simultaneously contain
    howler monkeys, kingfishers, caimans, and cicadas.  Softmax enforces
    mutual exclusivity which is ecologically incorrect.  Each of the 16 output
    logits is an independent binary classification problem.

**AdaptiveAvgPool2d in the head**
    Allows the model to accept variable time-axis lengths at inference time,
    which is important when processing recordings segmented with slightly
    different overlap fractions.

Usage
-----
Training (logits for BCEWithLogitsLoss):

    >>> model = WetlandBioacousticsNet()
    >>> logits = model(specs)          # (B, 16)  raw logits

Inference (calibrated probabilities):

    >>> probs = model.predict_proba(specs)   # (B, 16)  ∈ [0, 1]

References
----------
He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Identity Mappings in Deep Residual Networks. ECCV.

Hu, J., Shen, L., & Sun, G. (2018).
    Squeeze-and-Excitation Networks. CVPR.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from src.config import NUM_SPECIES, ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block
# ---------------------------------------------------------------------------


class SqueezeExcitation(nn.Module):
    """Channel-wise attention gate via global pooling and a two-layer bottleneck.

    The SE block learns a per-channel scalar weight in [0, 1] that gates how
    much each channel contributes to the next layer.  For mel-spectrograms,
    each feature-map channel is a learned combination of mel frequency bands,
    so SE functions as a soft frequency-band selector trained end-to-end.

    Mathematical form:

    .. code-block:: text

        s = σ( W₂ · ReLU( W₁ · GAP(x) ) )   # shape: (B, C)
        out = x · s.unsqueeze(-1).unsqueeze(-1)  # broadcast over H×W

    Parameters
    ----------
    channels : int
        Number of input (and output) channels C.
    reduction : int
        Bottleneck ratio.  Hidden dimensionality = max(channels // reduction, 1).
        The default of 4 gives a modest bottleneck that learns inter-channel
        relationships without adding significant parameter count.

    Attributes
    ----------
    pool : nn.AdaptiveAvgPool2d
        Global average pool squeezing spatial dims to (B, C, 1, 1).
    fc1 : nn.Linear
        First fully-connected layer: C → C // reduction.
    relu : nn.ReLU
        Non-linearity between the two FC layers.
    fc2 : nn.Linear
        Second fully-connected layer: C // reduction → C.
    sigmoid : nn.Sigmoid
        Squashes recalibration weights to (0, 1).

    Notes
    -----
    Bias terms are kept in both linear layers (unlike the Conv layers)
    because the bottleneck dimensionality is small and biases matter there.
    No BatchNorm is applied inside SE — this follows the original paper and
    avoids statistics coupling between the SE path and the main branch.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Compute and apply per-channel attention weights.

        Parameters
        ----------
        x : Tensor
            Feature map of shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Channel-recalibrated tensor of the same shape ``(B, C, H, W)``.
        """
        b, c, _, _ = x.shape

        # Squeeze: collapse spatial dims to a channel descriptor vector
        scale = self.pool(x).view(b, c)            # (B, C)

        # Excitation: two-layer bottleneck learns channel interdependencies
        scale = self.relu(self.fc1(scale))          # (B, C // reduction)
        scale = self.sigmoid(self.fc2(scale))       # (B, C)  ∈ (0, 1)

        # Scale: broadcast recalibration weights over H and W
        return x * scale.view(b, c, 1, 1)


# ---------------------------------------------------------------------------
# Residual block (pre-activation) with SE
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Pre-activation residual block with Squeeze-and-Excitation recalibration.

    Implements the identity-mapping variant from He et al. (2016v2):

    .. code-block:: text

        x ──► BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3 → SE ──► + ──► out
        │                                                            │
        └────────────────────────────────────────────────────────────┘
              identity (no projection; channels and spatial size unchanged)

    Pre-activation ordering (BN→ReLU before Conv, vs. Conv→BN→ReLU) means
    the gradient flows through the identity path without passing through any
    normalisation or non-linearity, which improves training stability at the
    depth used here.

    Parameters
    ----------
    channels : int
        Number of input and output channels.  Channel count is preserved
        through the block — spatial downsampling occurs only in the stage
        transition convolution, not inside blocks.
    se_reduction : int
        SE reduction ratio forwarded to :class:`SqueezeExcitation`.

    Attributes
    ----------
    bn1, bn2 : nn.BatchNorm2d
        Pre-activation batch normalisation layers.
    relu1, relu2 : nn.ReLU
        In-place ReLU activations (saves memory with no gradient impact).
    conv1, conv2 : nn.Conv2d
        3×3 convolutions with ``padding=1`` to preserve spatial dimensions.
        ``bias=False`` because the following BN has a learnable bias term.
    se : SqueezeExcitation
        Channel recalibration applied after the second convolution, before
        adding the residual.
    """

    def __init__(self, channels: int, se_reduction: int = 4) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.se = SqueezeExcitation(channels, reduction=se_reduction)

    def forward(self, x: Tensor) -> Tensor:
        """Compute residual transformation and add identity skip connection.

        Parameters
        ----------
        x : Tensor
            Input feature map of shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Output feature map of shape ``(B, C, H, W)``.
            Spatial resolution and channel count are unchanged.
        """
        residual = x  # save identity for skip connection

        # Pre-activation residual path
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))

        # SE recalibration applied to the residual branch output (not to the sum)
        out = self.se(out)

        # Identity skip connection: no projection needed — channels unchanged
        return out + residual


# ---------------------------------------------------------------------------
# Residual stage builder
# ---------------------------------------------------------------------------


def _make_stage(in_ch: int, out_ch: int, n_blocks: int) -> nn.Sequential:
    """Build one residual stage: a strided transition conv followed by residual blocks.

    The transition convolution doubles the channel count and halves both spatial
    dimensions (stride=2).  The subsequent ``ResidualBlock`` layers refine
    features at the new resolution without further downsampling.

    Parameters
    ----------
    in_ch : int
        Number of input channels entering this stage.
    out_ch : int
        Number of output channels leaving this stage (typically 2× in_ch).
    n_blocks : int
        Number of :class:`ResidualBlock` layers to append after the
        transition convolution.

    Returns
    -------
    nn.Sequential
        The complete stage as a single sequential module so it can be
        registered on ``WetlandBioacousticsNet`` as a named sub-module.

    Notes
    -----
    The transition conv uses ``stride=2`` rather than a max-pool to perform
    the downsampling, so the network learns the optimal spatial summarisation
    for this data domain rather than always taking the maximum activation.
    """
    layers: list[nn.Module] = [
        # Transition: increase channels, halve spatial resolution
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    # Depth-wise feature refinement at the new resolution
    for _ in range(n_blocks):
        layers.append(ResidualBlock(out_ch))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------


class WetlandBioacousticsNet(nn.Module):
    """Custom CNN for multi-label species detection from log-mel spectrograms.

    Accepts a batch of single-channel mel-spectrograms and outputs 16 raw
    logits (one per target species).  Sigmoid is deferred to inference time
    via :meth:`predict_proba` so that training uses the numerically stable
    ``BCEWithLogitsLoss``.

    Parameters
    ----------
    config : ModelConfig | None
        Architecture configuration.  Uses dataclass defaults if ``None``.
    num_classes : int
        Number of output neurons.  Defaults to ``NUM_SPECIES`` (16).
        Override only when adapting the model to a different species set.

    Attributes
    ----------
    stem : nn.Sequential
        Conv7×7 (stride=2) → BN → ReLU → MaxPool2d(2).
        Reduces ``(B, 1, 128, T)`` → ``(B, 32, 32, T//4)`` in one step using
        a large receptive field to aggregate low-level acoustic texture.
    stage1, stage2, stage3 : nn.Sequential
        Residual stages with channel progression 32→64→128→256.
        Each stage halves spatial resolution via stride-2 transition conv.
    pool : nn.AdaptiveAvgPool2d
        Collapses variable-length time axis to a fixed 256-dim descriptor.
        Enables inference on recordings of any length without re-training.
    classifier : nn.Sequential
        ``Dropout → Linear(256,128) → ReLU → Dropout → Linear(128,16)``.
        The intermediate 128-dim layer compresses the spatial descriptor
        before projecting to the 16-class output space.

    Notes
    -----
    **Parameter count** (default config, base_channels=32):

    - Stem: ≈ 14k params
    - Stage 1 (64ch, 2 blocks + SE): ≈ 148k params
    - Stage 2 (128ch, 2 blocks + SE): ≈ 591k params
    - Stage 3 (256ch, 2 blocks + SE): ≈ 2.36M params
    - Classifier head: ≈ 35k params
    - **Total: ≈ 3.1M trainable parameters**

    **Weight initialisation** follows these conventions:

    - ``Conv2d``: Kaiming normal, ``fan_out``, ``relu`` — correct for
      ReLU-activated networks; avoids vanishing/exploding gradients at init.
    - ``BatchNorm2d``: weight=1, bias=0 — identity transform at init, letting
      the network learn scale/shift from data.
    - ``Linear``: Xavier uniform — appropriate for layers not followed by ReLU
      (the final classifier output) and consistent for those that are.

    Examples
    --------
    Training forward pass:

        >>> model = WetlandBioacousticsNet()
        >>> specs = torch.randn(8, 1, 128, 313)  # batch of 5-second clips
        >>> logits = model(specs)                # (8, 16)
        >>> loss = nn.BCEWithLogitsLoss()(logits, labels)

    Inference:

        >>> probs = model.predict_proba(specs)   # (8, 16), no grad
        >>> detected = probs > 0.5               # (8, 16) bool mask
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        num_classes: int = NUM_SPECIES,
    ) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        base = cfg.base_channels  # 32 by default

        # ── Stem ────────────────────────────────────────────────────────────
        # Large 7×7 kernel captures low-level acoustic texture (harmonics,
        # transient onsets) across a wide receptive field in the first layer.
        # Stride-2 conv + stride-2 MaxPool gives 4× spatial reduction upfront,
        # shrinking memory before the residual stages.
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Residual stages ──────────────────────────────────────────────────
        # Channel counts double at each stage; spatial resolution halves.
        # 2 blocks per stage is the minimum for meaningful feature composition
        # while staying compact enough to train on ~5000 audio clips.
        self.stage1 = _make_stage(base, base * 2, n_blocks=2)       # 32 → 64
        self.stage2 = _make_stage(base * 2, base * 4, n_blocks=2)   # 64 → 128
        self.stage3 = _make_stage(base * 4, base * 8, n_blocks=2)   # 128 → 256

        final_ch = base * 8  # 256 channels entering the classifier head

        # ── Classifier head ──────────────────────────────────────────────────
        # AdaptiveAvgPool collapses (H, W) → (1, 1) regardless of input time
        # length, making the head resolution-agnostic for long-recording
        # inference.  Two dropout layers combat overfitting in the dense layers.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(final_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()
        logger.info(
            "WetlandBioacousticsNet initialised — %d trainable parameters",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def _init_weights(self) -> None:
        """Apply principled weight initialisation to all submodules.

        Iterates ``self.modules()`` and applies:

        - ``Conv2d``: Kaiming normal (fan_out, relu).
        - ``BatchNorm2d``: weight=1, bias=0 (identity at init).
        - ``Linear``: Xavier uniform, bias=0.

        Notes
        -----
        Called once from ``__init__``.  ``nn.init.kaiming_normal_`` with
        ``mode='fan_out'`` is preferred for deep convolutional networks as it
        accounts for the number of output connections (relevant during the
        forward pass) rather than input connections.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Compute raw (pre-sigmoid) logits for a batch of mel-spectrograms.

        Parameters
        ----------
        x : Tensor
            Batch of log-mel spectrograms, shape ``(B, 1, n_mels, T)``,
            values normalised to ``[0, 1]`` by ``AudioPreprocessor``.

        Returns
        -------
        Tensor
            Raw logits of shape ``(B, num_classes)`` with unrestricted range.
            Pass to ``BCEWithLogitsLoss`` during training; apply ``sigmoid``
            at inference via :meth:`predict_proba`.

        Notes
        -----
        Raw logits are returned (not probabilities) to allow the use of
        ``BCEWithLogitsLoss``, which fuses the sigmoid and BCE into one
        numerically stable operation via the log-sum-exp trick.  Applying
        ``sigmoid`` before the loss would introduce floating-point cancellation
        for extreme logit values.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)  # (B, 256)
        return self.classifier(x)    # (B, num_classes)

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Compute calibrated sigmoid probabilities for a batch of mel-spectrograms.

        Wraps :meth:`forward` with ``torch.no_grad()`` and applies sigmoid,
        producing values in ``[0, 1]`` suitable for thresholding and reporting.

        Parameters
        ----------
        x : Tensor
            Batch of log-mel spectrograms, shape ``(B, 1, n_mels, T)``,
            values normalised to ``[0, 1]``.

        Returns
        -------
        Tensor
            Per-species detection probabilities, shape ``(B, num_classes)``,
            values in ``[0, 1]``.  The ``requires_grad`` flag is always
            ``False`` because this method is decorated with ``@torch.no_grad()``.

        Notes
        -----
        The ``@torch.no_grad()`` decorator suppresses gradient computation and
        intermediate activation storage, halving peak memory usage compared to
        calling ``torch.sigmoid(model(x))`` inside a training loop.  Always
        use this method at inference time rather than calling ``forward``
        directly.
        """
        return torch.sigmoid(self.forward(x))

<div align="center">

# 🌿 Wetland Bioacoustics Species Detector

**Production-grade multi-label CNN for passive acoustic monitoring of Brazilian wetland ecosystems**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchaudio](https://img.shields.io/badge/torchaudio-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/audio/)
[![ONNX](https://img.shields.io/badge/ONNX-Export-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)
[![pytest](https://img.shields.io/badge/pytest-36%20passing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)
[![Ruff](https://img.shields.io/badge/Ruff-Linted-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![YAML](https://img.shields.io/badge/Config-YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white)](https://yaml.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](wetland-bioacoustics/LICENSE)

---

*Given a raw audio file recorded in the Pantanal, Amazon várzea, or Cerrado, output a calibrated probability score for each of 16 target species — birds, amphibians, mammals, reptiles, and insects — simultaneously present in the recording.*

</div>

---

## Overview

Passive acoustic monitoring (PAM) generates terabytes of continuous audio from remote sensor arrays deployed across threatened wetland biomes. Manual review is infeasible at scale. This project provides a complete ML pipeline — from raw field recording to per-species detection probability — designed to run on embedded edge hardware after ONNX export.

The core model, `WetlandBioacousticsNet`, is a **custom CNN trained from scratch** on log-mel spectrograms. It is a **multi-label classifier** (sigmoid output, not softmax) because multiple species vocalize simultaneously in real recordings. No pretrained weights. No transfer learning. No external ML frameworks beyond PyTorch.

**Key design constraints:**
- Zero sklearn, librosa, or soundfile in library code — pure `torch` + `torchaudio` I/O and DSP
- ONNX-exportable for deployment on low-power field sensors
- Config-driven via YAML with CLI override support — every experiment is fully reproducible
- All metrics implemented from scratch — mAP, macro-F1, per-species precision/recall

---

## Target Species (16)

| Taxon | Species | Scientific Name |
|-------|---------|-----------------|
| 🐦 Bird | Jabiru | *Jabiru mycteria* |
| 🐦 Bird | Hyacinth Macaw | *Anodorhynchus hyacinthinus* |
| 🐦 Bird | Rufescent Tiger-Heron | *Tigrisoma lineatum* |
| 🐦 Bird | Bare-faced Curassow | *Crax fasciolata* |
| 🐦 Bird | Chestnut-bellied Guan | *Penelope ochrogaster* |
| 🐦 Bird | Great Potoo | *Nyctibius grandis* |
| 🐦 Bird | Ringed Kingfisher | *Megaceryle torquata* |
| 🐦 Bird | Screaming Piha | *Lipaugus vociferans* |
| 🐸 Amphibian | Cane Toad | *Rhinella marina* |
| 🐸 Amphibian | Boana Treefrog | *Boana raniceps* |
| 🐸 Amphibian | Leptodactylus | *Leptodactylus fuscus* |
| 🐒 Mammal | Black Howler Monkey | *Alouatta caraya* |
| 🦦 Mammal | Giant Otter | *Pteronura brasiliensis* |
| 🦌 Mammal | Marsh Deer | *Blastocerus dichotomus* |
| 🐊 Reptile | Yacare Caiman | *Caiman yacare* |
| 🦟 Insect | Cicada chorus | *Cicadidae spp.* |

---

## Architecture

`WetlandBioacousticsNet` processes a single-channel log-mel spectrogram `(1, 128, 313)` through four stages:

```
Input (1, 128, 313)
        │
   ┌────▼────────────────────────────────────────────┐
   │  STEM                                            │
   │  Conv2d(1→32, 7×7, stride=2) → BN → ReLU        │
   │  MaxPool(2×2, stride=2)                          │
   └────┬────────────────────────────────────────────┘
        │ (32, 32, 78)
   ┌────▼────────────────────────────────────────────┐
   │  STAGE 1  — 32 → 64 channels, stride-2 conv     │
   │  2× ResidualBlock(64) with SE recalibration      │
   └────┬────────────────────────────────────────────┘
        │ (64, 16, 39)
   ┌────▼────────────────────────────────────────────┐
   │  STAGE 2  — 64 → 128 channels, stride-2 conv    │
   │  2× ResidualBlock(128) with SE recalibration     │
   └────┬────────────────────────────────────────────┘
        │ (128, 8, 20)
   ┌────▼────────────────────────────────────────────┐
   │  STAGE 3  — 128 → 256 channels, stride-2 conv   │
   │  2× ResidualBlock(256) with SE recalibration     │
   └────┬────────────────────────────────────────────┘
        │ (256, 4, 10)
   ┌────▼────────────────────────────────────────────┐
   │  HEAD                                            │
   │  AdaptiveAvgPool2d(1) → Flatten                  │
   │  Dropout(0.3) → Linear(256, 128) → ReLU          │
   │  Dropout(0.3) → Linear(128, 16)                  │
   └────┬────────────────────────────────────────────┘
        │
   Raw logits (16,)  ──train──▶  BCEWithLogitsLoss
                     ──infer──▶  Sigmoid → probabilities
```

### ResidualBlock (Pre-activation + Squeeze-and-Excitation)

```
x ──► BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3 → SE ──► + ──► out
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**Why SE blocks?** Each mel frequency bin maps to a specific acoustic band. SE learns to upweight bands where target species vocalize (e.g., 300–3000 Hz for howler monkey resonance, >10 kHz for cicada harmonics) and suppress irrelevant bands — achieving this without additional supervision beyond the multi-hot species labels.

**Weight initialisation:**
- `Conv2d` → Kaiming normal (`fan_out`, `relu`)
- `BatchNorm2d` → weight=1, bias=0
- `Linear` → Xavier uniform, bias=0

---

## Project Structure

```
wetland-bioacoustics/
├── src/
│   ├── config.py            # Dataclass configs + YAML loader (single source of truth)
│   ├── model.py             # WetlandBioacousticsNet, ResidualBlock, SqueezeExcitation
│   ├── preprocessing.py     # AudioPreprocessor, BioacousticAugmentor
│   ├── dataset.py           # WetlandAudioDataset, create_dataloaders()
│   ├── train.py             # Trainer: AdamW + CosineAnnealingLR + checkpointing
│   ├── inference.py         # WetlandPredictor: single / batch / sliding-window
│   ├── metrics.py           # macro-F1, mAP, per-species report (no sklearn)
│   └── utils.py             # set_seed, get_device, setup_logging
├── scripts/
│   ├── download_data.py     # Xeno-canto API v2 crawler → 32 kHz WAV
│   ├── prepare_labels.py    # Multi-hot labels + stratified 70/15/15 split
│   └── export_onnx.py       # ONNX export, output verification, latency benchmark
├── tests/
│   ├── test_model.py        # Forward shape, gradient flow, init, predict_proba
│   ├── test_preprocessing.py # Mono/resample/pad/mel/segment/augment
│   └── test_inference.py    # End-to-end: single, long, batch, JSON export
├── configs/
│   └── default.yaml         # Canonical hyperparameter config
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

## Setup

**Requirements:** Python 3.10+, PyTorch 2.1+, torchaudio 2.1+

```bash
git clone https://github.com/kabir2004/BogCall.git
cd BogCall/wetland-bioacoustics
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.1.0
torchaudio>=2.1.0
pyyaml>=6.0
pytest>=7.4.0
ruff>=0.1.0
numpy>=1.24.0
```

> **Note:** torchaudio requires a system I/O backend for WAV I/O. On macOS without ffmpeg, install `soundfile` (`pip install soundfile`) to provide this backend. It is only used by torchaudio internally — no `soundfile` imports appear anywhere in library code.

---

## Data Pipeline

```bash
# Step 1 — Download quality-A recordings from Xeno-canto (Brazil, ~300/species)
python scripts/download_data.py \
    --output data/raw \
    --max-per-species 300

# Step 2 — Build multi-hot labels + stratified train/val/test splits (70/15/15)
python scripts/prepare_labels.py \
    --metadata data/raw/metadata.json \
    --audio-root data/raw \
    --output data
```

`download_data.py` queries the [Xeno-canto API v2](https://xeno-canto.org/api/2/recordings), enforces 1 req/sec rate limiting, converts MP3 → mono 32 kHz WAV via torchaudio, and resumes interrupted downloads by checking for existing files.

`prepare_labels.py` reads the `also` field on each recording to assign multi-hot secondary species labels (a Jabiru recording that also contains a Ringed Kingfisher call gets both bits set), then performs stratified splits so each species is proportionally represented across all three splits.

**Expected data layout after preparation:**

```
data/
├── train/
│   ├── audio/          # .wav files
│   └── labels.csv      # multi-hot: filename, Species A, Species B, ...
├── val/
│   ├── audio/
│   └── labels.csv
└── test/
    ├── audio/
    └── labels.csv
```

---

## Training

```bash
make train
# or with explicit config
python -m src.train --config configs/default.yaml
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | weight_decay=1e-4 |
| Learning rate | 1e-3 | Initial |
| Scheduler | CosineAnnealingLR | T_max=epochs, η_min=1e-6 |
| Loss | BCEWithLogitsLoss | numerically stable sigmoid+BCE |
| Gradient clipping | max_norm=1.0 | clips before optimizer step |
| Batch size | 32 | |
| Epochs | 50 | |
| Detection threshold | 0.5 | for F1 computation during training |

### Augmentation Pipeline (training only)

| Augmentation | Probability | Parameters | Rationale |
|---|---|---|---|
| SpecAugment — frequency mask | 0.5 | up to 20 mel bins | Simulates partial frequency occlusion by wind/rain |
| SpecAugment — time mask | 0.5 | up to 40 frames | Simulates transient interruptions (flight, splashing) |
| Gaussian noise | 0.3 | σ=0.005 | Recorder noise floor variation |
| Circular time shift | 0.3 | ±10% | Species calls rarely align to clip boundaries |
| Mixup | per-batch | α=0.4 (Beta dist.) | Reduces overconfidence, improves calibration |

### Training Output

```
 Epoch  Train Loss  Val Loss  Macro F1       mAP    Time   *
--------------------------------------------------------------
     1      0.6831    0.6712    0.1234    0.0891    12.3s  *
     2      0.6201    0.6043    0.1891    0.1320     9.8s
     3      0.5984    0.5701    0.2310    0.1745    10.1s  *
```

`*` = new best macro-F1. Checkpoint saved to `checkpoints/best.pt` on improvement.

**Checkpoint format:**
```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "best_f1": float,
    "metrics": {"train_loss", "val_loss", "macro_f1", "mAP"},
    "config": dict,   # full config snapshot for reproducibility
}
```

---

## Inference

### CLI

```bash
# Single 5-second clip
make predict FILE=recording.wav

# Long field recording — sliding window (50% overlap), max-pool across segments
python -m src.inference recording.wav \
    --checkpoint checkpoints/best.pt \
    --long \
    --overlap 0.5 \
    --threshold 0.5 \
    --output results.json
```

**Console output:**

```
============================================================
  Results: recording.wav
============================================================
 ✓ Black Howler Monkey (Alouatta caraya)         [██████████████████████░░░░░░░░] 0.741
 ✓ Screaming Piha (Lipaugus vociferans)          [█████████████████░░░░░░░░░░░░░] 0.567
   Jabiru (Jabiru mycteria)                      [████████░░░░░░░░░░░░░░░░░░░░░░] 0.281
   Cicada chorus (Cicadidae spp.)                [████░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.142
   ...

  2/16 species detected above threshold
============================================================
```

### Programmatic API

```python
from src.inference import WetlandPredictor

predictor = WetlandPredictor(
    checkpoint_path="checkpoints/best.pt",
    threshold=0.5,
)

# Single 5-second clip
results = predictor.predict("clip.wav")

# Long field recording — segments internally, max-pools probabilities
results = predictor.predict_long_recording("field_recording_2h.wav", overlap=0.5)

# Batch inference
results_list = predictor.predict_batch(["a.wav", "b.wav", "c.wav"])

# Formatted console output
WetlandPredictor.print_results(results, title="Site A — Dawn Chorus")

# JSON export
WetlandPredictor.to_json(results, "results/site_a.json")
```

**`predict_long_recording` strategy:** The recording is segmented with a sliding 5-second window at `overlap` fractional overlap. Each segment is independently scored. Final probability per species = `max()` across all segments — if a species appears in any segment, its peak probability is reported. This is the correct aggregation for presence/absence monitoring.

---

## Metrics

All metrics are implemented from scratch in `src/metrics.py`. No sklearn.

| Metric | Function | Description |
|--------|----------|-------------|
| Macro Precision | `macro_precision(preds, targets, threshold)` | Mean per-species precision |
| Macro Recall | `macro_recall(preds, targets, threshold)` | Mean per-species recall |
| Macro F1 | `macro_f1(preds, targets, threshold)` | Mean per-species F1 (primary training signal) |
| mAP | `mean_average_precision(preds, targets)` | Mean per-species average precision (threshold-free) |
| Per-species report | `per_species_report(preds, targets)` | `{species → {precision, recall, f1, support}}` |

**Validation is computed on the full validation set** (not averaged per-batch) — this is statistically correct for macro metrics where class imbalance matters.

---

## Audio Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 32 kHz | Bioacoustics standard; captures harmonics up to 15 kHz |
| Clip duration | 5 s | Sufficient context for most vocalisation types |
| n_mels | 128 | High enough resolution to separate species in adjacent bands |
| n_fft | 1024 | ~32 ms window; balances freq/time resolution |
| hop_length | 512 | 50% overlap; ~16 ms frame step |
| f_min | 50 Hz | Captures infrasonic caiman rumbles and howler resonance |
| f_max | 15 kHz | Captures upper harmonics of bird song and cicadas |
| top_db | 80 dB | Dynamic range ceiling; suppresses digital noise floor |

Spectrogram output shape: `(1, 128, 313)` for a 5-second 32 kHz clip.

**Padding strategy:** Short clips are repeat-padded (tiled), not zero-padded. Zero-padding creates a hard spectral discontinuity at the splice point that the model learns as a spurious feature. Tiling preserves the spectral statistics of the original recording.

---

## ONNX Export

```bash
make export
# or
python scripts/export_onnx.py \
    --checkpoint checkpoints/best.pt \
    --output checkpoints/model.onnx \
    --opset 17
```

The export script:
1. Loads the checkpoint and exports with dynamic batch and time axes
2. Verifies ONNX output matches PyTorch output within `atol=1e-5`
3. Reports model file size and mean single-sample inference latency

Suitable for deployment via ONNX Runtime on ARM/x86 edge hardware without a PyTorch dependency.

---

## Testing

```bash
make test
# or
pytest tests/ -v
```

**36 tests, 0 failures.** All tests use synthetically generated audio (`torch.randn`) — no real recordings required to run the test suite.

| Test module | Coverage |
|-------------|----------|
| `test_model.py` | Forward shape `(B, 16)`, gradient flow to all parameters, `predict_proba` ∈ [0,1], no gradient in inference mode, BatchNorm init, variable time-axis inputs |
| `test_preprocessing.py` | Stereo→mono, 16kHz→32kHz resample, pad (short/long/exact/1-sample), mel shape `(1,128,313)`, unit-interval output, segment count for 30s recording, augmentor bounds, Mixup shapes |
| `test_inference.py` | 16 result dicts, correct keys, probabilities ∈ [0,1], all species present, sorted by probability, long recording, batch, JSON export |

---

## Configuration Reference

`configs/default.yaml` — the canonical experiment configuration:

```yaml
audio:
  sample_rate: 32000
  clip_duration: 5
  n_mels: 128
  n_fft: 1024
  hop_length: 512
  fmin: 50
  fmax: 15000
  top_db: 80.0

model:
  base_channels: 32    # stem output channels; stages progress 32→64→128→256
  dropout: 0.3

training:
  epochs: 50
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  num_workers: 4
  detection_threshold: 0.5

augmentation:
  freq_mask_param: 20
  time_mask_param: 40
  noise_std: 0.005
  mixup_alpha: 0.4

paths:
  data_dir: data
  checkpoint_dir: checkpoints
  log_dir: logs
```

Override any value at runtime:

```python
from src.config import load_config

cfg = load_config(
    "configs/default.yaml",
    overrides={"training": {"lr": 3e-4, "batch_size": 64}},
)
```

---

## Makefile

```bash
make train      # python -m src.train
make test       # pytest tests/ -v
make lint       # ruff check src/ tests/
make predict    # FILE=recording.wav — single-file inference
make download   # scripts/download_data.py
make export     # scripts/export_onnx.py
```

---

## Design Decisions

**Why a custom CNN rather than a pretrained ResNet?**
Pretrained ImageNet weights encode RGB photographic statistics (3 channels, natural textures). Mel spectrograms are single-channel, log-scaled, and semantically structured along the frequency axis in a domain-specific way. Training from scratch on bioacoustic data lets the network learn features optimal for this domain without fighting the inductive biases baked into ImageNet pretraining.

**Why multi-label sigmoid and not softmax?**
Species co-occur. A recording from a Pantanal marsh at dawn routinely contains overlapping calls from howler monkeys, kingfishers, caimans, and cicadas. Softmax enforces a mutual-exclusivity constraint that is ecologically wrong. BCEWithLogitsLoss treats each species as an independent binary variable.

**Why repeat-pad rather than zero-pad?**
Zero-padding creates a sharp silent segment in the spectrogram. The model learns to associate the silence with reduced probability mass and generalises poorly to recordings where calls happen to fall near a clip boundary. Repeat-padding maintains the ambient noise floor statistics throughout the clip.

**Why max-pool across segments for long recordings?**
In presence/absence monitoring, the relevant question is whether a species was detected at any point during the recording. Max-pooling preserves the peak signal. Mean-pooling would dilute a strong 5-second detection buried in 55 seconds of background noise.

---

## License

MIT

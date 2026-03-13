"""Export a trained WetlandBioacousticsNet checkpoint to ONNX.

Usage::

    python scripts/export_onnx.py \\
        --checkpoint checkpoints/best.pt \\
        --output checkpoints/model.onnx

Verifies that the ONNX model output matches PyTorch output within tolerance,
then prints model size and an estimated single-sample inference time.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config
from src.model import WetlandBioacousticsNet
from src.utils import get_device, setup_logging

logger = logging.getLogger(__name__)


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    config: Config | None = None,
    opset: int = 17,
    atol: float = 1e-5,
) -> None:
    """Load a checkpoint and export to ONNX format.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the PyTorch checkpoint (.pt).
    output_path : str | Path
        Destination path for the ONNX file (.onnx).
    config : Config | None
        Configuration object. Uses checkpoint config if None.
    opset : int
        ONNX opset version.
    atol : float
        Absolute tolerance for PyTorch vs ONNX output verification.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    logger.info(
        "Loaded checkpoint — epoch %d, best F1 %.4f",
        ckpt.get("epoch", -1),
        ckpt.get("best_f1", float("nan")),
    )

    if config is None:
        ckpt_cfg = ckpt.get("config", {})
        config = load_config(overrides=ckpt_cfg) if ckpt_cfg else Config()

    model = WetlandBioacousticsNet(config=config.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Dummy input: single 5-second mel-spectrogram
    dummy = torch.randn(1, 1, config.audio.n_mels, config.audio.n_frames, device=device)

    print(f"Exporting to ONNX (opset {opset})...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["mel_spectrogram"],
            output_names=["logits"],
            dynamic_axes={
                "mel_spectrogram": {0: "batch_size", 3: "time_frames"},
                "logits": {0: "batch_size"},
            },
        )
    print(f"ONNX model saved → {output_path}")

    # Verify ONNX output matches PyTorch
    _verify_onnx(model, dummy, output_path, atol=atol)

    # Print stats
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")
    _benchmark_inference(model, dummy, device)


def _verify_onnx(
    model: WetlandBioacousticsNet,
    dummy: torch.Tensor,
    onnx_path: Path,
    atol: float,
) -> None:
    """Verify ONNX outputs match PyTorch outputs within tolerance.

    Parameters
    ----------
    model : WetlandBioacousticsNet
        PyTorch model (in eval mode).
    dummy : torch.Tensor
        Dummy input tensor.
    onnx_path : Path
        Path to the exported ONNX file.
    atol : float
        Absolute tolerance for comparison.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping ONNX verification.")
        print("  Install with: pip install onnxruntime")
        return

    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path))
    onnx_out = sess.run(["logits"], {"mel_spectrogram": dummy.cpu().numpy()})[0]

    import numpy as np

    max_diff = float(np.abs(pt_out - onnx_out).max())
    if max_diff <= atol:
        print(f"ONNX verification passed. Max output difference: {max_diff:.2e}")
    else:
        raise RuntimeError(
            f"ONNX verification FAILED. Max output difference {max_diff:.2e} exceeds "
            f"tolerance {atol:.2e}"
        )


def _benchmark_inference(
    model: WetlandBioacousticsNet,
    dummy: torch.Tensor,
    device: torch.device,
    n_runs: int = 50,
) -> None:
    """Measure and print mean single-sample inference latency.

    Parameters
    ----------
    model : WetlandBioacousticsNet
        PyTorch model (in eval mode).
    dummy : torch.Tensor
        Dummy input tensor (batch size 1).
    device : torch.device
        Compute device.
    n_runs : int
        Number of forward passes to average.
    """
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000
    print(f"Mean inference latency: {elapsed:.2f} ms / sample ({device})")


def main() -> None:
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(description="Export WetlandBioacousticsNet to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for PyTorch/ONNX output comparison",
    )
    args = parser.parse_args()

    setup_logging()
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
        atol=args.atol,
    )


if __name__ == "__main__":
    main()

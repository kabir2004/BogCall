"""Convert Xeno-canto metadata to multi-hot labels.csv and train/val/test splits.

Usage::

    python scripts/prepare_labels.py \\
        --metadata data/raw/metadata.json \\
        --audio-root data/raw \\
        --output data

Reads metadata produced by download_data.py, builds multi-hot label vectors
(primary species = 1; secondary species in the "also" field = 1), and splits
recordings into train/val/test (70/15/15) with stratification by species.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SPECIES_LIST, SPECIES_TO_IDX
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def _build_label_vector(
    primary_label: str,
    also_field: str | None,
) -> list[int]:
    """Build a 16-dimensional multi-hot label vector.

    Parameters
    ----------
    primary_label : str
        The species label stored in ``_species_label`` by download_data.py.
    also_field : str | None
        Xeno-canto "also" string listing secondary species (comma-separated
        common or scientific names).

    Returns
    -------
    list[int]
        Binary label vector of length ``len(SPECIES_LIST)``.
    """
    vec = [0] * len(SPECIES_LIST)

    if primary_label in SPECIES_TO_IDX:
        vec[SPECIES_TO_IDX[primary_label]] = 1

    if also_field:
        for fragment in also_field.split(","):
            fragment = fragment.strip().lower()
            for species in SPECIES_LIST:
                if fragment and fragment in species.lower():
                    vec[SPECIES_TO_IDX[species]] = 1

    return vec


def _stratified_split(
    records: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train / val / test with species-level stratification.

    Stratification is performed per primary species so each split contains a
    proportional sample of each species class.

    Parameters
    ----------
    records : list[dict]
        List of record dicts with ``_species_label`` key.
    train_frac : float
        Fraction allocated to training (e.g. 0.70).
    val_frac : float
        Fraction allocated to validation (e.g. 0.15). Test = remainder.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[list[dict], list[dict], list[dict]]
        (train_records, val_records, test_records)
    """
    rng = random.Random(seed)

    by_species: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_species[rec["_species_label"]].append(rec)

    train, val, test = [], [], []
    for species_recs in by_species.values():
        rng.shuffle(species_recs)
        n = len(species_recs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(species_recs[:n_train])
        val.extend(species_recs[n_train: n_train + n_val])
        test.extend(species_recs[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _write_split(
    records: list[dict],
    audio_root: Path,
    out_dir: Path,
) -> None:
    """Copy audio files and write labels.csv for one split.

    Parameters
    ----------
    records : list[dict]
        Records for this split.
    audio_root : Path
        Root directory where raw audio files live.
    out_dir : Path
        Output split directory (e.g. ``data/train``).
    """
    audio_out = out_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"

    header = ["filename"] + SPECIES_LIST
    rows: list[list[str]] = []

    for rec in records:
        src = Path(rec.get("_local_path", ""))
        if not src.exists():
            logger.warning("Audio file not found, skipping: %s", src)
            continue

        dst = audio_out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

        vec = _build_label_vector(rec["_species_label"], rec.get("also"))
        rows.append([src.name] + [str(v) for v in vec])

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), csv_path)


def _print_summary(
    train: list[dict], val: list[dict], test: list[dict]
) -> None:
    """Print split statistics to stdout."""
    total = len(train) + len(val) + len(test)
    print(f"\n{'Split':<8} {'Count':>7}  {'%':>6}")
    print("-" * 25)
    for name, recs in [("train", train), ("val", val), ("test", test)]:
        pct = 100.0 * len(recs) / max(total, 1)
        print(f"{name:<8} {len(recs):>7}  {pct:>5.1f}%")
    print(f"{'total':<8} {total:>7}")

    # Per-species counts in train
    print(f"\n{'Species':<50} {'Train':>6}  {'Val':>6}  {'Test':>6}")
    print("-" * 68)
    from collections import Counter

    def _count(recs: list[dict]) -> Counter:
        return Counter(r["_species_label"] for r in recs)

    tc, vc, ec = _count(train), _count(val), _count(test)
    for sp in SPECIES_LIST:
        print(f"{sp:<50} {tc.get(sp, 0):>6}  {vc.get(sp, 0):>6}  {ec.get(sp, 0):>6}")


def main() -> None:
    """CLI entry point for label preparation."""
    parser = argparse.ArgumentParser(description="Prepare multi-hot labels from Xeno-canto data")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/raw/metadata.json",
        help="Path to metadata.json produced by download_data.py",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default="data/raw",
        help="Root directory containing downloaded audio files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output data root (creates train/val/test subdirectories)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        logger.error("Metadata file not found: %s", meta_path)
        sys.exit(1)

    with meta_path.open() as fh:
        records: list[dict] = json.load(fh)
    logger.info("Loaded %d metadata records.", len(records))

    train, val, test = _stratified_split(records, seed=args.seed)
    _print_summary(train, val, test)

    out_root = Path(args.output)
    audio_root = Path(args.audio_root)
    for split_name, split_recs in [("train", train), ("val", val), ("test", test)]:
        _write_split(split_recs, audio_root, out_root / split_name)

    print(f"\nData splits written to {out_root}/{{train,val,test}}/")


if __name__ == "__main__":
    main()

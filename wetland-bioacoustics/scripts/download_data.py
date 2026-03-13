"""Download bioacoustic recordings from Xeno-canto for target species.

Usage::

    python scripts/download_data.py --output data/raw --max-per-species 300

Queries Xeno-canto API v2 for each target species + country:brazil + quality:A,
downloads .mp3 files, converts them to mono 32 kHz .wav via torchaudio, and
skips files that have already been downloaded (resume support).

Rate-limiting: 1 API request per second, 0.5 s between file downloads.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

# Add repo root to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SPECIES_LIST, AudioConfig
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# Scientific name extracted from parentheses, e.g. "Jabiru (Jabiru mycteria)" → "Jabiru mycteria"
_XENO_CANTO_API = "https://xeno-canto.org/api/2/recordings"


def _scientific_name(species_label: str) -> str:
    """Extract scientific name from display label.

    Parameters
    ----------
    species_label : str
        Label like ``"Jabiru (Jabiru mycteria)"``.

    Returns
    -------
    str
        Scientific name, e.g. ``"Jabiru mycteria"``.
    """
    if "(" in species_label and ")" in species_label:
        return species_label.split("(", 1)[1].rstrip(")")
    return species_label


def _query_xeno_canto(species_label: str, page: int = 1) -> dict:
    """Query Xeno-canto API for recordings of a species in Brazil.

    Parameters
    ----------
    species_label : str
        Species display label (scientific name extracted internally).
    page : int
        API page number (1-indexed).

    Returns
    -------
    dict
        Parsed JSON response from the API.
    """
    sci_name = _scientific_name(species_label)
    query = f'"{sci_name}" cnt:brazil q:A'
    url = f"{_XENO_CANTO_API}?query={urllib.request.quote(query)}&page={page}"
    logger.debug("GET %s", url)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_recording(
    recording: dict,
    out_dir: Path,
    target_sr: int = 32000,
) -> Path | None:
    """Download one Xeno-canto recording, convert to mono WAV at target_sr.

    Parameters
    ----------
    recording : dict
        Xeno-canto recording metadata dict.
    out_dir : Path
        Directory to save the .wav file.
    target_sr : int
        Target sample rate in Hz.

    Returns
    -------
    Path | None
        Path to the saved .wav file, or None if the download failed.
    """
    xc_id = recording.get("id", "unknown")
    file_url: str = recording.get("file", "")
    if not file_url:
        logger.warning("Recording %s has no file URL, skipping.", xc_id)
        return None

    # Ensure URL has a scheme
    if file_url.startswith("//"):
        file_url = "https:" + file_url

    wav_path = out_dir / f"xc{xc_id}.wav"
    if wav_path.exists():
        logger.debug("Already exists, skipping: %s", wav_path)
        return wav_path

    tmp_mp3 = out_dir / f"xc{xc_id}.mp3"
    try:
        urllib.request.urlretrieve(file_url, tmp_mp3)
        waveform, sr = torchaudio.load(str(tmp_mp3))
        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample
        if sr != target_sr:
            waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        torchaudio.save(str(wav_path), waveform, target_sr)
        tmp_mp3.unlink(missing_ok=True)
        logger.info("Saved: %s", wav_path)
        return wav_path
    except Exception as exc:
        logger.error("Failed to download xc%s: %s", xc_id, exc)
        tmp_mp3.unlink(missing_ok=True)
        return None


def download_species(
    species_label: str,
    out_dir: Path,
    max_recordings: int = 300,
    audio_cfg: AudioConfig | None = None,
) -> list[dict]:
    """Download up to max_recordings recordings for one species.

    Parameters
    ----------
    species_label : str
        Species display label.
    out_dir : Path
        Destination directory (will be created).
    max_recordings : int
        Maximum number of recordings to download.
    audio_cfg : AudioConfig | None
        Audio configuration (for target sample rate).

    Returns
    -------
    list[dict]
        List of metadata dicts for successfully downloaded recordings.
    """
    cfg = audio_cfg or AudioConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded_meta: list[dict] = []
    page = 1

    while len(downloaded_meta) < max_recordings:
        time.sleep(1.0)  # Rate limit: 1 req/sec
        try:
            data = _query_xeno_canto(species_label, page=page)
        except Exception as exc:
            logger.error("API query failed for '%s' page %d: %s", species_label, page, exc)
            break

        recordings = data.get("recordings", [])
        if not recordings:
            break

        for rec in recordings:
            if len(downloaded_meta) >= max_recordings:
                break
            time.sleep(0.5)
            path = _download_recording(rec, out_dir, target_sr=cfg.sample_rate)
            if path:
                rec["_local_path"] = str(path)
                rec["_species_label"] = species_label
                downloaded_meta.append(rec)

        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break
        page += 1

    logger.info(
        "Species '%s': downloaded %d recordings.", species_label, len(downloaded_meta)
    )
    return downloaded_meta


def main() -> None:
    """CLI entry point for data download."""
    parser = argparse.ArgumentParser(description="Download Xeno-canto recordings")
    parser.add_argument("--output", type=str, default="data/raw", help="Output root directory")
    parser.add_argument(
        "--max-per-species",
        type=int,
        default=300,
        help="Maximum recordings per species",
    )
    parser.add_argument(
        "--species-subset",
        nargs="+",
        default=None,
        help="Subset of species labels to download (default: all 16)",
    )
    args = parser.parse_args()

    setup_logging()
    out_root = Path(args.output)
    targets = args.species_subset or SPECIES_LIST
    audio_cfg = AudioConfig()

    all_meta: list[dict] = []
    for species in targets:
        species_dir = out_root / species.replace(" ", "_").replace("(", "").replace(")", "")
        meta = download_species(species, species_dir, args.max_per_species, audio_cfg)
        all_meta.extend(meta)

    meta_path = out_root / "metadata.json"
    with meta_path.open("w") as fh:
        json.dump(all_meta, fh, indent=2)
    print(f"\nDownload complete. {len(all_meta)} recordings saved.")
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()

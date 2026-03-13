# BogCall

ML and tooling for passive acoustic monitoring of wetland ecosystems — species detection from field recordings.

## Contents

- **[wetland-bioacoustics](./wetland-bioacoustics/)** — Production-grade multi-label CNN for detecting 16 target species (birds, amphibians, mammals, reptiles, insects) in Brazilian wetland audio. PyTorch training, ONNX export, config-driven pipeline.

  → [Full documentation → wetland-bioacoustics/README.md](./wetland-bioacoustics/README.md)

## Quick start

```bash
cd wetland-bioacoustics
pip install -r requirements.txt
# See wetland-bioacoustics/README.md for data prep, training, and inference.
```

## License

MIT (see [wetland-bioacoustics/LICENSE](./wetland-bioacoustics/LICENSE) if present).

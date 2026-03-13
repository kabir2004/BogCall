train:
	python -m src.train

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

predict:
	python -m src.inference $(FILE) --long --checkpoint checkpoints/best.pt

download:
	python scripts/download_data.py

export:
	python scripts/export_onnx.py

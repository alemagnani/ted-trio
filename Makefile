# Convenience targets (all CPU unless noted). PYTHON defaults to `python`.
PYTHON ?= python
export PYTHON

.PHONY: help test check-results check-data smoke quick beauty-headline reproduce
help:
	@echo "make test             unit tests (pytest, CPU, < 1 min)"
	@echo "make check-results    paper numbers == results/, paired tests re-run (CPU, ~5 min)"
	@echo "make check-data       validate prepared datasets (TT_DATA_DIR or data/)"
	@echo "make smoke            synthetic end-to-end run of every pipeline stage (CPU, ~1 h)"
	@echo "make quick            real-data subset (200 users) run of every stage, all datasets (CPU)"
	@echo "make beauty-headline  towers -> gate -> look -> masked rescore for Beauty from existing embeddings (GPU)"
	@echo "make reproduce        the full pipeline for all datasets (GPU; see README)"

test:
	$(PYTHON) -m pytest -q

check-results:
	$(PYTHON) scripts/check_results.py

check-data:
	$(PYTHON) scripts/check_data.py

smoke:
	bash scripts/smoke_e2e.sh

quick:
	bash scripts/reproduce.sh --quick --dataset all

beauty-headline:
	bash scripts/reproduce.sh --dataset beauty --stages towers,gate,look

reproduce:
	bash scripts/reproduce.sh --dataset all

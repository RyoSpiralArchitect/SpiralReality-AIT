.PHONY: whitepaper whitepaper-data whitepaper-figures whitepaper-pdf

WHITEPAPER_DIR := docs/whitepaper
WHITEPAPER_DATA := $(WHITEPAPER_DIR)/data/evaluation_metrics.json
WHITEPAPER_MD := $(WHITEPAPER_DIR)/whitepaper.md
WHITEPAPER_PDF := $(WHITEPAPER_DIR)/whitepaper.pdf

whitepaper: whitepaper-pdf
	@echo "Whitepaper artefacts generated in $(WHITEPAPER_DIR)"

whitepaper-data:
	python scripts/run_evaluation.py --output $(WHITEPAPER_DIR)/data

whitepaper-figures: whitepaper-data
	python $(WHITEPAPER_DIR)/generate_figures.py --input $(WHITEPAPER_DATA) --output-dir $(WHITEPAPER_DIR)/figures

whitepaper-pdf: whitepaper-figures $(WHITEPAPER_MD)
	python $(WHITEPAPER_DIR)/build_whitepaper.py --markdown $(WHITEPAPER_MD) --output $(WHITEPAPER_PDF)

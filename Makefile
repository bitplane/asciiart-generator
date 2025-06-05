.PHONY: all help venv install glyphs clean distclean

all: glyphs ## Default target - build glyph data

venv: .venv/bin/activate ## Create Python virtual environment

.venv/bin/activate:
	python3 -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

install: .venv/.install.done ## Install dependencies into venv

.venv/.install.done: .venv/bin/activate scripts/install.sh requirements.txt
	@echo "Installing dependencies..."
	@. .venv/bin/activate && ./scripts/install.sh
	@touch $@

glyphs: cache/.glyphs.done ## Generate glyph analysis data

cache/.glyphs.done: .venv/.install.done scripts/quarter_glyphs.py
	@mkdir -p cache
	@echo "Generating glyph analysis data..."
	@. .venv/bin/activate && python scripts/quarter_glyphs.py
	@touch $@

clean: ## Remove generated files but keep venv
	rm -rf cache/*

distclean: clean ## Remove everything including venv
	rm -rf .venv/

help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

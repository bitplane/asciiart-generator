# ANSI Canvas Project Makefile

.PHONY: all help venv install glyphs clean distclean

# Default target
all: glyphs

# Show available targets
help:
	@echo "ANSI Canvas Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make help      - Show this help message"
	@echo "  make venv      - Create Python virtual environment"
	@echo "  make install   - Install dependencies into venv"
	@echo "  make glyphs    - Generate glyph analysis data"
	@echo "  make clean     - Remove generated files"
	@echo "  make distclean - Remove everything including venv"
	@echo ""
	@echo "Quick start: make venv install glyphs"

# Create virtual environment
venv: .venv/bin/activate

.venv/bin/activate:
	python3 -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

# Install dependencies
install: .venv/.install.done

.venv/.install.done: .venv/bin/activate scripts/install.sh requirements.txt
	@echo "Installing dependencies..."
	@. .venv/bin/activate && ./scripts/install.sh
	@touch $@

# Generate glyph data
glyphs: cache/.glyphs.done

cache/.glyphs.done: .venv/.install.done scripts/quarter_glyphs.py
	@mkdir -p cache
	@echo "Generating glyph analysis data..."
	@. .venv/bin/activate && python scripts/quarter_glyphs.py
	@touch $@

# Clean generated files but keep venv
clean:
	rm -rf cache/

# Clean everything including venv
distclean: clean
	rm -rf .venv/
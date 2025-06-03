.PHONY: all clean dataset train convert test help

# Main consolidated script
SCRIPT = ./ansi_canvas.py

# Default settings
DATA_DIR = data
MODEL_PATH = model.pth
TEST_IMAGE = images/input/van.jpg

all: dataset train

help:
	@echo "ANSI Canvas - Image to ASCII Art Converter"
	@echo ""
	@echo "Commands:"
	@echo "  make dataset    - Generate training dataset from images"
	@echo "  make train      - Train the neural network model"
	@echo "  make convert    - Convert test image to ASCII art"
	@echo "  make clean      - Remove all generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make dataset    - Uses images/input/*.jpg"
	@echo "  make convert    - Creates output.ansi (with color)"
	@echo "  make convert-no-color - Creates output.ansi (no color)"

dataset:
	$(SCRIPT) dataset images/input/*.jpg --output $(DATA_DIR)

train:
	$(SCRIPT) train --data $(DATA_DIR) --output $(MODEL_PATH) --epochs 20

convert:
	$(SCRIPT) convert $(TEST_IMAGE) --model $(MODEL_PATH)

convert-no-color:
	$(SCRIPT) convert $(TEST_IMAGE) --model $(MODEL_PATH) --no-color

clean:
	rm -rf $(DATA_DIR)/
	rm -rf training_data/
	rm -rf extended_training_data/
	rm -rf test_images/
	rm -f *.pth
	rm -f output*.ansi
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Note: TensorBoard logs in runs/ are preserved"

# Quick test targets
test: convert
	@echo "Viewing output..."
	@cat output.ansi | head -20

test-no-color: convert-no-color
	@echo "Viewing output without color..."
	@cat output.ansi | head -20
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ANSI Canvas is a machine learning project that converts images to ASCII/Unicode art in the terminal. It uses neural networks to find the best matching character for each region of an image, considering the surrounding context.

## Key Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate training data (parallel version recommended)
python generate_extended_dataset_parallel.py

# Train the model
python train_model.py

# Start TensorBoard
tensorboard --logdir=runs

# Convert image to ASCII art
python inference.py images/input/van.jpg [--color]

# View output
cat output.ansi
cat output_color.ansi
```

## Architecture

The project uses a CNN-based approach:
1. Input: 3×3 character grid (context window)
2. Model: ConvASCIINet with conv layers + fully connected
3. Output: Best matching Unicode character for center position

Key components in `ansi_canvas.py`:
- Font extraction: `get_monospace_chars()` extracts unique monospace characters
- Dataset generation: Creates training data from images at multiple scales
- Training: PyTorch CNN with TensorBoard logging
- Inference: Converts images to colored ASCII/ANSI art with styles

## Development Notes

- The project uses PyTorch with CUDA support (RTX 3080, 16GB VRAM)
- Character dimensions are based on Ubuntu Sans Mono font (7×10 pixels)
- Extended character set uses FreeMono font (~500+ characters)
- Parallel processing uses all CPU cores for dataset generation
- ANSI color codes (256-color mode) are used for colored output
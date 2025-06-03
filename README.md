# ANSI Canvas

Convert images to ASCII/Unicode art using machine learning. Uses a CNN to intelligently map image regions to the best matching terminal characters.

## Features

- Neural network-based character selection (not just brightness mapping)
- Context-aware: considers surrounding pixels when choosing characters
- Unicode support: box drawing, block elements, geometric shapes
- ANSI styling: bold, reverse video based on brightness
- 256-color ANSI support
- Parallel processing for fast dataset generation

## Quick Start

```bash
# Clean any previous runs
make clean

# Generate dataset and train model
make all

# Convert an image
make convert

# Convert with color
make convert-color

# View the results
cat output.ansi
cat output_color.ansi
```

## Usage

The main script `ansi_canvas.py` has three commands:

### 1. Generate Dataset
```bash
./ansi_canvas.py dataset images/*.jpg --output data
```

### 2. Train Model
```bash
./ansi_canvas.py train --data data --epochs 20 --output model.pth
```

### 3. Convert Images
```bash
# Basic ASCII
./ansi_canvas.py convert image.jpg --model model.pth

# With colors and styles
./ansi_canvas.py convert image.jpg --model model.pth --color

# Without bold/reverse styles
./ansi_canvas.py convert image.jpg --model model.pth --no-style
```

## Character Set

Uses Ubuntu Sans Mono font with these Unicode blocks:
- Basic Latin (ASCII)
- Latin-1 Supplement
- Box Drawing (│├─┌└┐┘┤┬┴┼)
- Block Elements (▀▄█▌▐░▒▓)
- Geometric Shapes (■□▪▫●)

## Architecture

The CNN model (`ConvASCIINet`) takes a 3×3 grid of characters as input and predicts the best character for the center position:

```
Input: 3×3 character grid (21×30 pixels)
  ↓
Conv1: 16 filters → MaxPool
  ↓
Conv2: 32 filters → MaxPool
  ↓
FC1: 256 neurons
  ↓
Output: Character class
```

## Dependencies

- PyTorch (with CUDA support recommended)
- Pillow
- NumPy
- scikit-learn
- TensorBoard
- fonttools

## Files

- `ansi_canvas.py` - Main script (includes font extraction)
- `data/` - Generated training data
- `runs/` - TensorBoard logs
- `model.pth` - Trained model
- `output.ansi` - Generated ASCII art
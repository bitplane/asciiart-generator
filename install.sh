#!/bin/bash

# Install script for ansi-canvas dependencies

echo "Installing ansi-canvas dependencies..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    
    # Install the rest without torch/torchvision
    uv pip install "numpy>=1.24.0" "Pillow>=10.0.0" "fonttools>=4.38.0" "scikit-learn>=1.3.0" "tensorboard>=2.13.0"
    
    # Install NVIDIA tools
    uv pip install "nvidia-ml-py>=12.0.0" "pynvml>=11.5.0"
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    uv pip install -r requirements.txt
fi

echo "Installation complete!"

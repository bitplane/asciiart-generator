#!/usr/bin/env python3
"""Save character quarters as PNG files with systematic naming."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import json
import numpy as np
from PIL import Image
import argparse
from state import load_state

def save_quarter_png(quarter_array, filename):
    """Save quarter as PNG."""
    img_array = (quarter_array).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(filename)
    print(f"Saved {filename} - shape: {quarter_array.shape}")

def save_quarters(char, include_flips=False):
    """Save all quarters and optionally flips for a given character."""
    # Load data
    state = load_state()
    glyph_data = state["glyphs"]
    quarter_data = state["images"]

    if char not in glyph_data:
        print(f"Character '{char}' not found in glyph data")
        return

    char_data = glyph_data[char]
    quarter_hashes = char_data['data']
    
    # Quarter positions: (x, y) where 0=left/top, 1=right/bottom
    quarter_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR

    print(f"Character '{char}' from {char_data['font']}")
    
    for pos_idx, (x, y) in enumerate(quarter_positions):
        hash_val = quarter_hashes[pos_idx]
        if hash_val not in quarter_data:
            print(f"Missing quarter data for position {x}{y}")
            continue
            
        quarter = quarter_data[hash_val]
        
        if include_flips:
            # Generate all flip combinations
            flip_transforms = {
                (0, 0): lambda q: q,                    # No flips
                (1, 0): lambda q: np.fliplr(q),        # H flip only
                (0, 1): lambda q: np.flipud(q),        # V flip only  
                (1, 1): lambda q: np.flipud(np.fliplr(q))  # H+V flip
            }
        else:
            # Only original (no flips)
            flip_transforms = {
                (0, 0): lambda q: q,                    # No flips
            }
        
        for (h_flip, v_flip), transform in flip_transforms.items():
            transformed = transform(quarter)
            filename = f"{char}_{x}{y}_{h_flip}{v_flip}.png"
            save_quarter_png(transformed, filename)

def main():
    parser = argparse.ArgumentParser(description='Save character quarters as PNG files')
    parser.add_argument('character', help='Character to save quarters for (e.g., O)')
    parser.add_argument('--flip', action='store_true', 
                       help='Include flipped versions of quarters (default: original only)')
    
    args = parser.parse_args()
    save_quarters(args.character, args.flip)

if __name__ == "__main__":
    main()
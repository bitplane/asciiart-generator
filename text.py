#!/usr/bin/env python3
"""
Render text using quarter-based braille representation.
"""

import pickle
import json
import numpy as np
import argparse

def quarter_to_braille_grid(quarter_array, threshold=128, scale=4):
    """Convert a quarter image to a grid of braille characters."""
    if quarter_array.size == 0:
        return [' ' * scale] * scale
    
    # Get quarter dimensions
    height, width = quarter_array.shape
    
    # Create a grid of braille characters
    rows = []
    for by in range(scale):
        row = ""
        for bx in range(scale):
            # Each braille char represents a 2x4 region within this grid cell
            char_y_start = (by * height) // scale
            char_y_end = ((by + 1) * height) // scale
            char_x_start = (bx * width) // scale
            char_x_end = ((bx + 1) * width) // scale
            
            # Extract the region for this braille character
            char_region = quarter_array[char_y_start:char_y_end, char_x_start:char_x_end]
            
            # Convert this region to a single braille character
            braille_char = region_to_braille(char_region, threshold)
            row += braille_char
        
        rows.append(row)
    
    return rows

def region_to_braille(region, threshold=128):
    """Convert a small image region to a single braille character."""
    if region.size == 0:
        return ' '
    
    region_height, region_width = region.shape
    
    # Braille patterns use 2x4 dot matrix (8 dots total)
    braille_height = 4
    braille_width = 2
    
    # Sample the region into 2x4 grid
    dots = []
    for by in range(braille_height):
        for bx in range(braille_width):
            # Map braille dot position to region pixels
            y_start = (by * region_height) // braille_height
            y_end = ((by + 1) * region_height) // braille_height
            x_start = (bx * region_width) // braille_width  
            x_end = ((bx + 1) * region_width) // braille_width
            
            # Get average darkness in this dot area
            if y_end > y_start and x_end > x_start:
                dot_region = region[y_start:y_end, x_start:x_end]
                avg_darkness = np.mean(dot_region)
            else:
                avg_darkness = 255  # White if no pixels
            
            # If darker than threshold, mark dot as on
            dots.append(avg_darkness < threshold)
    
    # Convert dots to braille unicode
    # Braille pattern dots numbered:
    # 1 4
    # 2 5  
    # 3 6
    # 7 8
    
    # Map our 2x4 sampling to braille dot positions
    dot_map = [0, 3, 1, 4, 2, 5, 6, 7]  # Our order -> braille dot numbers
    
    braille_value = 0
    for i, dot_on in enumerate(dots):
        if dot_on:
            braille_value |= (1 << dot_map[i])
    
    # Convert to braille unicode character
    braille_char = chr(0x2800 + braille_value)
    return braille_char

def render_text(text, scale=4, threshold=128):
    """Render text using quarter-based font."""
    # Load data
    try:
        with open('glyph_quarters.json', 'r') as f:
            glyph_data = json.load(f)
        
        with open('quarter_data.pkl', 'rb') as f:
            quarter_data = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure glyph_quarters.json and quarter_data.pkl exist")
        return
    
    # Process each character
    char_blocks = []
    
    for char in text:
        if char not in glyph_data:
            # Use space for unknown characters
            empty_quarter = [' ' * scale] * scale
            char_blocks.append([empty_quarter, empty_quarter, empty_quarter, empty_quarter])
            continue
        
        char_data = glyph_data[char]
        quarter_hashes = char_data['data']
        
        # Get quarters: [TL, TR, BL, BR]
        quarters = []
        for hash_val in quarter_hashes:
            if hash_val in quarter_data:
                quarters.append(quarter_data[hash_val])
            else:
                quarters.append(np.full((32, 16), 255, dtype=np.uint8))  # White quarter
        
        # Convert quarters to braille grids
        tl_braille = quarter_to_braille_grid(quarters[0], threshold, scale)  # Top-left
        tr_braille = quarter_to_braille_grid(quarters[1], threshold, scale)  # Top-right
        bl_braille = quarter_to_braille_grid(quarters[2], threshold, scale)  # Bottom-left
        br_braille = quarter_to_braille_grid(quarters[3], threshold, scale)  # Bottom-right
        
        char_blocks.append([tl_braille, tr_braille, bl_braille, br_braille])
    
    # Combine all characters into output lines
    total_rows = scale * 2  # Top quarters + bottom quarters
    output_lines = [''] * total_rows
    
    for char_block in char_blocks:
        tl, tr, bl, br = char_block
        
        # Top half (TL + TR side by side)
        for row in range(scale):
            output_lines[row] += tl[row] + tr[row]
        
        # Bottom half (BL + BR side by side)  
        for row in range(scale):
            output_lines[scale + row] += bl[row] + br[row]
    
    # Print result
    for line in output_lines:
        print(line)

def main():
    parser = argparse.ArgumentParser(description='Render text using quarter-based braille font')
    parser.add_argument('text', help='Text to render')
    parser.add_argument('--scale', type=int, default=4, 
                       help='Size of each quarter in characters (default: 4)')
    parser.add_argument('--threshold', type=int, default=128, 
                       help='Darkness threshold for braille dots (default: 128)')
    
    args = parser.parse_args()
    render_text(args.text, args.scale, args.threshold)

if __name__ == "__main__":
    main()
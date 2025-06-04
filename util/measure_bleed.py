#!/usr/bin/env python3
"""Test which glyphs bleed outside their declared bounds."""

import os
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def find_font_path(font_basename):
    """Find the actual font file path from basename."""
    # Remove .1.txt suffix to get font name
    font_name = font_basename.replace('.1.txt', '')
    
    # Common font paths to search
    search_paths = [
        f"/usr/share/fonts/truetype/*/{font_name}.ttf",
        f"/usr/share/fonts/truetype/*/{font_name}.otf", 
        f"/usr/share/fonts/opentype/*/{font_name}.ttf",
        f"/usr/share/fonts/opentype/*/{font_name}.otf",
    ]
    
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def test_glyph_bleed(font_path, char, font_size=16):
    """Test if a glyph bleeds outside its declared bounds."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        return False
    
    # Use space character to determine actual terminal cell size
    # In monospace fonts, space char represents the exact cell the terminal uses
    space_bbox = font.getbbox(' ')
    if not space_bbox:
        return False
    
    # Get the space character's dimensions and baseline position
    ascent, descent = font.getmetrics()
    
    # Space character width is the cell width
    cell_width = space_bbox[2] - space_bbox[0]
    
    # Cell height is the full line height (what terminal allocates)
    cell_height = ascent + descent
    
    # The baseline position from space character
    space_baseline_y = space_bbox[1]  # This should be where baseline is positioned
    
    # Create 3x3 grid image
    grid_width = cell_width * 3
    grid_height = cell_height * 3
    
    # Create test image and render character in center
    test_img = Image.new('L', (grid_width, grid_height), color=255)
    draw = ImageDraw.Draw(test_img)
    
    # Position character in center cell using proper baseline anchoring
    center_x = cell_width
    # Place baseline in center cell with room for ascenders and descenders
    # Center cell runs from cell_height to 2*cell_height
    center_y = cell_height + ascent  # Baseline position in center cell
    try:
        # Use baseline anchor for proper positioning
        draw.text((center_x, center_y), char, font=font, fill=0, anchor='ls')
    except TypeError:
        # Fallback for older PIL versions without anchor support
        # Adjust position to simulate baseline anchoring
        adjusted_y = center_y - ascent
        draw.text((center_x, adjusted_y), char, font=font, fill=0)
    
    # Check for horizontal bleeding by blanking out center column and checking left/right
    test_img_blanked = test_img.copy()
    draw_blanked = ImageDraw.Draw(test_img_blanked)
    
    # Blank out the center column (where character should be contained)
    draw_blanked.rectangle([cell_width, 0, cell_width*2, grid_height], fill=255)
    
    # Check if there are any non-white pixels left in left or right columns
    test_array = np.array(test_img_blanked)
    
    # Check left column (horizontal bleeding to the left)
    left_bleed = np.any(test_array[:, :cell_width] < 255)
    
    # Check right column (horizontal bleeding to the right) 
    right_bleed = np.any(test_array[:, cell_width*2:] < 255)
    
    return left_bleed or right_bleed

def process_font_file(font_file_path, font_size=16):
    """Process a single .1.txt file and classify glyphs as bleeding or not."""
    print(f"Processing {os.path.basename(font_file_path)}...")
    
    # Find the actual font file
    font_path = find_font_path(os.path.basename(font_file_path))
    if not font_path:
        print(f"  Could not find font file for {font_file_path}")
        return
    
    print(f"  Using font: {font_path}")
    
    # Read characters from file
    try:
        with open(font_file_path, 'r', encoding='utf-8') as f:
            chars = f.read()
    except Exception as e:
        print(f"  Error reading {font_file_path}: {e}")
        return
    
    # Classify each character
    bleed_chars = []
    nobleed_chars = []
    
    for char in chars:
        if test_glyph_bleed(font_path, char, font_size):
            bleed_chars.append(char)
        else:
            nobleed_chars.append(char)
    
    # Generate output filenames
    base_name = os.path.basename(font_file_path).replace('.1.txt', '')
    bleed_file = f"data/fonts/{base_name}.1-bleed.txt"
    nobleed_file = f"data/fonts/{base_name}.1-nobleed.txt"
    
    # Save results
    with open(bleed_file, 'w', encoding='utf-8') as f:
        f.write(''.join(bleed_chars))
    
    with open(nobleed_file, 'w', encoding='utf-8') as f:
        f.write(''.join(nobleed_chars))
    
    print(f"  Bleed: {len(bleed_chars)} chars -> {bleed_file}")
    print(f"  No bleed: {len(nobleed_chars)} chars -> {nobleed_file}")

def main():
    parser = argparse.ArgumentParser(description='Test glyphs for bleeding outside bounds')
    parser.add_argument('--font-size', type=int, default=16, help='Font size for testing (default: 16)')
    parser.add_argument('--font', help='Process specific font file (optional)')
    args = parser.parse_args()
    
    if args.font:
        # Process single font
        process_font_file(args.font, args.font_size)
    else:
        # Process all .1.txt files
        pattern = "data/fonts/*.1.txt"
        files = glob.glob(pattern)
        
        print(f"Found {len(files)} font files to process")
        
        for font_file in files:
            process_font_file(font_file, args.font_size)

if __name__ == "__main__":
    main()
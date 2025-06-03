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
    
    # Get font metrics to determine cell size
    # Use 'M' as reference for monospace width
    m_bbox = font.getbbox('M')
    if not m_bbox:
        return False
    
    cell_width = m_bbox[2] - m_bbox[0]
    cell_height = font_size  # Use font size as height
    
    # Create 3x3 grid image
    grid_width = cell_width * 3
    grid_height = cell_height * 3
    
    # Create test image and render character in center
    test_img = Image.new('L', (grid_width, grid_height), color=255)
    draw = ImageDraw.Draw(test_img)
    
    # Position character in center cell (no offset needed, PIL handles positioning)
    center_x = cell_width
    center_y = cell_height
    draw.text((center_x, center_y), char, font=font, fill=0)
    
    # Blank out the center cell
    test_img_blanked = test_img.copy()
    draw_blanked = ImageDraw.Draw(test_img_blanked)
    draw_blanked.rectangle([cell_width, cell_height, cell_width*2, cell_height*2], fill=255)
    
    # Check if there are any non-white pixels left (indicating bleeding)
    test_array = np.array(test_img_blanked)
    return np.any(test_array < 255)

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
    bleed_file = f"data/fonts/{base_name}.1.bleed.txt"
    nobleed_file = f"data/fonts/{base_name}.1.nobleed.txt"
    
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
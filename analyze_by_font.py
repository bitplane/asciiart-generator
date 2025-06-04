#!/usr/bin/env python3
"""
Analyze glyphs by iterating through fonts instead of characters.
Much faster because we avoid font resolution lookups.
"""

import sys
import os
sys.path.append('util')

from PIL import Image, ImageDraw, ImageFont
import hashlib
import json
from multiprocessing import Pool, cpu_count
import time
import numpy as np
import glob
from fontTools.ttLib import TTFont

def get_quarter_hash(quarter_array):
    """Generate a full MD5 hash for a quarter image array."""
    quarter_bytes = quarter_array.tobytes()
    return hashlib.md5(quarter_bytes).hexdigest()

def extract_quarters(img_array):
    """Extract 4 quarters from an image array and return their hashes."""
    height, width = img_array.shape
    mid_y, mid_x = height // 2, width // 2
    
    quarters = [
        img_array[:mid_y, :mid_x],      # Top-left (h0)
        img_array[:mid_y, mid_x:],      # Top-right (h1)
        img_array[mid_y:, :mid_x],      # Bottom-left (h2)
        img_array[mid_y:, mid_x:]       # Bottom-right (h3)
    ]
    
    return [get_quarter_hash(q) for q in quarters]

def get_font_glyphs(font_path):
    """Get all character mappings from a font file."""
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        if cmap:
            # Return list of (codepoint, glyph_name) tuples
            return [(cp, glyph) for cp, glyph in cmap.items() if cp < 0x110000]
        return []
    except Exception:
        return []

def test_character_in_font(char, font, cell_width, cell_height, ascent):
    """Test if a character is quarterable in a specific font."""
    # Create 3x3 grid image
    grid_width = cell_width * 3
    grid_height = cell_height * 3
    
    test_img = Image.new('L', (grid_width, grid_height), color=255)
    draw = ImageDraw.Draw(test_img)
    
    # Position character in center cell
    center_x = cell_width
    center_y = cell_height + ascent
    
    try:
        draw.text((center_x, center_y), char, font=font, fill=0, anchor='ls')
    except TypeError:
        adjusted_y = center_y - ascent
        draw.text((center_x, adjusted_y), char, font=font, fill=0)
    except Exception:
        return None
    
    # Test for horizontal bleeding
    test_array = np.array(test_img)
    
    # Check if bleeds horizontally
    left_pixels = test_array[:, :cell_width]
    right_pixels = test_array[:, cell_width*2:]
    
    if np.any(left_pixels < 255) or np.any(right_pixels < 255):
        return None  # Bleeds
    
    # Extract center cell
    center_cell = test_array[cell_height:cell_height*2, cell_width:cell_width*2]
    
    # Only process if has content
    if np.any(center_cell < 255):
        return extract_quarters(center_cell)
    
    return None

def process_font(args):
    """Process all glyphs in a single font."""
    font_path, font_size = args
    font_name = os.path.basename(font_path)
    
    print(f"Processing {font_name}...", flush=True)
    
    # Get all glyphs in this font
    glyphs = get_font_glyphs(font_path)
    if not glyphs:
        return {}
    
    # Load font once
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return {}
    
    # Get font metrics once
    space_bbox = font.getbbox(' ')
    if not space_bbox:
        return {}
    
    cell_width = space_bbox[2] - space_bbox[0]
    ascent, descent = font.getmetrics()
    cell_height = ascent + descent
    
    # Test if this is actually monospace by comparing M and i
    m_bbox = font.getbbox('M')
    i_bbox = font.getbbox('i')
    if m_bbox and i_bbox:
        if (m_bbox[2] - m_bbox[0]) != (i_bbox[2] - i_bbox[0]):
            # Skip proportional fonts for now
            print(f"  Skipping {font_name} (proportional)", flush=True)
            return {}
    
    results = {}
    tested = 0
    found = 0
    
    # Process each glyph
    for codepoint, glyph_name in glyphs:
        if codepoint < 32 and codepoint != 9:  # Skip control chars
            continue
            
        try:
            char = chr(codepoint)
        except ValueError:
            continue
        
        tested += 1
        quarter_hashes = test_character_in_font(char, font, cell_width, cell_height, ascent)
        
        if quarter_hashes:
            results[char] = {
                'font': font_name,
                'data': quarter_hashes
            }
            found += 1
        
        # Progress update every 100 chars
        if tested % 100 == 0:
            print(f"  {font_name}: tested {tested}/{len(glyphs)}, found {found}", flush=True)
    
    print(f"  {font_name}: DONE - tested {tested}, found {found}", flush=True)
    return results

def main():
    print("Font-Based Glyph Analysis", flush=True)
    print("=" * 50, flush=True)
    
    # Find all font files
    font_patterns = [
        "/usr/share/fonts/truetype/*/*.ttf",
        "/usr/share/fonts/truetype/*/*.otf",
        "/usr/share/fonts/opentype/*/*.ttf",
        "/usr/share/fonts/opentype/*/*.otf",
    ]
    
    font_files = []
    for pattern in font_patterns:
        font_files.extend(glob.glob(pattern))
    
    # Filter to monospace fonts (heuristic based on name)
    monospace_keywords = ['mono', 'code', 'consol', 'courier', 'fixed', 'term']
    monospace_fonts = []
    
    for font_file in font_files:
        font_name = os.path.basename(font_file).lower()
        if any(keyword in font_name for keyword in monospace_keywords):
            monospace_fonts.append(font_file)
    
    print(f"Found {len(monospace_fonts)} monospace fonts to analyze", flush=True)
    
    # Process fonts in parallel
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores", flush=True)
    
    font_args = [(font, 16) for font in monospace_fonts]
    
    start_time = time.time()
    all_results = {}
    unique_hashes = set()
    fonts_processed = 0
    
    with Pool(num_cores) as pool:
        for font_results in pool.imap_unordered(process_font, font_args):
            fonts_processed += 1
            all_results.update(font_results)
            
            # Count unique hashes
            for char_data in font_results.values():
                for hash_val in char_data['data']:
                    unique_hashes.add(hash_val)
            
            # Progress
            elapsed = time.time() - start_time
            print(f"\nProgress: {fonts_processed}/{len(monospace_fonts)} fonts, "
                  f"{len(all_results):,} glyphs, "
                  f"{len(unique_hashes):,} unique quarters", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds", flush=True)
    
    # Save results
    output_file = "quarterable_glyphs_by_font.json"
    print(f"\nSaving results to {output_file}...", flush=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved!", flush=True)
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Fonts processed: {len(monospace_fonts)}")
    print(f"  Quarterable glyphs: {len(all_results):,}")
    print(f"  Unique quarter patterns: {len(unique_hashes):,}")

if __name__ == "__main__":
    main()
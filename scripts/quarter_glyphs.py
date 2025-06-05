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
import pickle
from multiprocessing import Pool, cpu_count
import time
import numpy as np
import glob
from fontTools.ttLib import TTFont
from state import load_state, save_state

def get_quarter_hash(quarter_array):
    """Generate a full MD5 hash for a quarter image array."""
    quarter_bytes = quarter_array.tobytes()
    return hashlib.md5(quarter_bytes).hexdigest()

def extract_quarters(img_array):
    """Extract 4 quarters from an image array and return their hashes and data."""
    height, width = img_array.shape
    mid_y, mid_x = height // 2, width // 2
    
    quarters = [
        img_array[:mid_y, :mid_x],      # Top-left (h0)
        img_array[:mid_y, mid_x:],      # Top-right (h1)
        img_array[mid_y:, :mid_x],      # Bottom-left (h2)
        img_array[mid_y:, mid_x:]       # Bottom-right (h3)
    ]
    
    hashes = [get_quarter_hash(q) for q in quarters]
    
    return hashes, quarters

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

def test_character_in_font(char, font_path, font_size, space_width, line_height):
    """Test if a character is quarterable in a specific font using space-based metrics."""
    # Render at high resolution for quality, then resize to our target
    HIGH_RES_SCALE = 4
    TARGET_WIDTH = 16  # Character width (narrower)
    TARGET_HEIGHT = 32  # Character height (taller)
    
    # Create high-res image based on space metrics
    hr_width = space_width * HIGH_RES_SCALE
    hr_height = line_height * HIGH_RES_SCALE
    
    # Create 3x3 grid for bleeding test
    grid_width = hr_width * 3
    grid_height = hr_height * 3
    
    test_img = Image.new('L', (grid_width, grid_height), color=255)
    draw = ImageDraw.Draw(test_img)
    
    # Create high-resolution font 
    try:
        hr_font = ImageFont.truetype(font_path, font_size * HIGH_RES_SCALE)
    except Exception:
        return None
    
    # Position character in center cell using proper terminal positioning
    try:
        # Get high-res font metrics for proper positioning
        ascent, descent = hr_font.getmetrics()
        
        # Position character properly within the center cell
        center_x = hr_width  # Left edge of center cell
        center_y = hr_height + ascent  # Baseline position in center cell
        
        # Draw character at high resolution
        draw.text((center_x, center_y), char, font=hr_font, fill=0, anchor='ls')
        
    except Exception:
        # Fallback positioning if anchor fails
        try:
            center_x = hr_width
            center_y = hr_height
            draw.text((center_x, center_y), char, font=hr_font, fill=0)
        except:
            return None
    
    # Test for horizontal bleeding at high resolution
    test_array = np.array(test_img)
    
    # Check if bleeds horizontally out of center cell
    left_pixels = test_array[:, :hr_width]
    right_pixels = test_array[:, hr_width*2:]
    
    if np.any(left_pixels < 255) or np.any(right_pixels < 255):
        return None  # Bleeds
    
    # Extract center cell at high resolution
    center_cell_hr = test_array[hr_height:hr_height*2, hr_width:hr_width*2]
    
    # Only process if has content
    if not np.any(center_cell_hr < 255):
        return None
    
    # Resize to our target dimensions (32×64)
    center_img = Image.fromarray(center_cell_hr, mode='L')
    resized_img = center_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    final_array = np.array(resized_img)
    
    # Extract quarters from resized image (16×32 each)
    return extract_quarters(final_array)

def process_font(args):
    """Process all glyphs in a single font."""
    font_path, font_size = args
    font_name = os.path.basename(font_path)
    
    print(f"Processing {font_name}...", flush=True)
    
    # Get all glyphs in this font
    glyphs = get_font_glyphs(font_path)
    if not glyphs:
        return {}, {}
    
    # Load font once
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return {}, {}
    
    # Get space-based font metrics
    space_bbox = font.getbbox(' ')
    if not space_bbox:
        return {}, {}
    
    space_width = space_bbox[2] - space_bbox[0]
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    
    # Test if this is actually monospace by comparing M and i
    m_bbox = font.getbbox('M')
    i_bbox = font.getbbox('i')
    if m_bbox and i_bbox:
        if (m_bbox[2] - m_bbox[0]) != (i_bbox[2] - i_bbox[0]):
            # Skip proportional fonts for now
            print(f"  Skipping {font_name} (proportional)", flush=True)
            return {}, {}
    
    results = {}
    quarter_data = {}
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
        quarter_result = test_character_in_font(char, font_path, font_size, space_width, line_height)
        
        if quarter_result:
            quarter_hashes, quarters = quarter_result
            results[char] = {
                'font': font_name,
                'data': quarter_hashes
            }
            
            # Store quarter image data by hash
            for hash_val, quarter_array in zip(quarter_hashes, quarters):
                if hash_val not in quarter_data:
                    quarter_data[hash_val] = quarter_array.copy()
            
            found += 1
        
        # Progress update every 100 chars
        if tested % 100 == 0:
            print(f"  {font_name}: tested {tested}/{len(glyphs)}, found {found}", flush=True)
    
    print(f"  {font_name}: DONE - tested {tested}, found {found}", flush=True)
    return results, quarter_data

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
    all_quarter_data = {}
    unique_hashes = set()
    fonts_processed = 0
    
    with Pool(num_cores) as pool:
        for font_results, font_quarter_data in pool.imap_unordered(process_font, font_args):
            fonts_processed += 1
            all_results.update(font_results)
            all_quarter_data.update(font_quarter_data)
            
            # Count unique hashes
            for char_data in font_results.values():
                for hash_val in char_data['data']:
                    unique_hashes.add(hash_val)
            
            # Progress
            elapsed = time.time() - start_time
            print(f"\nProgress: {fonts_processed}/{len(monospace_fonts)} fonts, "
                  f"{len(all_results):,} glyphs, "
                  f"{len(all_quarter_data):,} unique quarters", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds", flush=True)
    
    # Save results to state
    print(f"\nSaving state data...", flush=True)
    state = {
        "images": all_quarter_data,
        "glyphs": all_results
    }
    save_state(state)
    print(f"State saved!", flush=True)
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Fonts processed: {len(monospace_fonts)}")
    print(f"  Quarterable glyphs: {len(all_results):,}")
    print(f"  Unique quarter patterns: {len(all_quarter_data):,}")
    
    # Sample some quarter data info
    if all_quarter_data:
        sample_hash = next(iter(all_quarter_data.keys()))
        sample_quarter = all_quarter_data[sample_hash]
        print(f"  Quarter image size: {sample_quarter.shape}")
        print(f"  Quarter data type: {sample_quarter.dtype}")

if __name__ == "__main__":
    main()
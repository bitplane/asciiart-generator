#!/usr/bin/env python3
"""
Render text using quarter-based braille representation.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import json
import numpy as np
import argparse
from state import load_state

def hash_to_color(obj_repr):
    """Hash an object's repr and return ANSI color code."""
    import hashlib
    
    # Hash the repr
    hash_obj = hashlib.md5(obj_repr.encode('utf-8'))
    hex_hash = hash_obj.hexdigest()
    
    # Use first 6 chars as RGB hex
    rgb_hex = hex_hash[:6]
    r = int(rgb_hex[0:2], 16)
    g = int(rgb_hex[2:4], 16) 
    b = int(rgb_hex[4:6], 16)
    
    # Ensure minimum brightness (at least one component > 128)
    max_component = max(r, g, b)
    if max_component < 128:
        # Scale up to ensure visibility
        scale_factor = 128 / max_component if max_component > 0 else 2
        r = min(255, int(r * scale_factor))
        g = min(255, int(g * scale_factor))
        b = min(255, int(b * scale_factor))
    
    # Convert to ANSI 256-color
    r_index = (r * 5) // 255
    g_index = (g * 5) // 255
    b_index = (b * 5) // 255
    color_code = 16 + (36 * r_index) + (6 * g_index) + b_index
    
    return f"\033[38;5;{color_code}m"

def hamming_distance_match(img1, img2, threshold=0.9):
    """Compare using Hamming distance with threshold."""
    if img1.shape != img2.shape:
        return False
    binary1 = (img1 < 128).astype(np.uint8)
    binary2 = (img2 < 128).astype(np.uint8)
    diff_pixels = np.sum(binary1 != binary2)
    similarity = 1 - (diff_pixels / img1.size)
    return similarity >= threshold

def simple_erosion(binary_img):
    """Simple erosion: pixel is True only if all neighbors are True."""
    eroded = np.zeros_like(binary_img)
    h, w = binary_img.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if np.all(binary_img[y-1:y+2, x-1:x+2]):
                eroded[y, x] = True
    return eroded

def simple_dilation(binary_img):
    """Simple dilation: pixel is True if any neighbor is True."""
    dilated = np.copy(binary_img)
    h, w = binary_img.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if np.any(binary_img[y-1:y+2, x-1:x+2]):
                dilated[y, x] = True
    return dilated

def erosion_dilation_match(img1, img2, threshold=0.9):
    """Compare after erosion/dilation to handle aliasing."""
    if img1.shape != img2.shape:
        return False
    binary1 = (img1 < 128)
    binary2 = (img2 < 128)
    
    eroded1 = simple_erosion(binary1)
    eroded2 = simple_erosion(binary2)
    dilated1 = simple_dilation(binary1)
    dilated2 = simple_dilation(binary2)
    
    eroded_diff = np.sum(eroded1 != eroded2) / eroded1.size
    dilated_diff = np.sum(dilated1 != dilated2) / dilated1.size
    best_similarity = 1 - min(eroded_diff, dilated_diff)
    return best_similarity >= threshold

def correlation_match(img1, img2, threshold=0.8):
    """Compare using simple correlation."""
    if img1.shape != img2.shape:
        return False
    
    flat1 = img1.flatten().astype(float)
    flat2 = img2.flatten().astype(float)
    
    correlation = np.corrcoef(flat1, flat2)[0, 1]
    return not np.isnan(correlation) and correlation >= threshold

def distance_transform_match(img1, img2, threshold=0.9):
    """Compare using simple distance from edges."""
    if img1.shape != img2.shape:
        return False
    
    binary1 = (img1 < 128).astype(np.uint8)
    binary2 = (img2 < 128).astype(np.uint8)
    
    # Simple edge detection - count neighbors
    def count_edge_pixels(binary):
        h, w = binary.shape
        edges = np.zeros_like(binary, dtype=float)
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if binary[y, x]:
                    # Count non-filled neighbors
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            if binary[y+dy, x+dx]:
                                neighbors += 1
                    # Pixel is edge if it has non-filled neighbors
                    edges[y, x] = 8 - neighbors
        return edges
    
    edges1 = count_edge_pixels(binary1)
    edges2 = count_edge_pixels(binary2)
    
    # Compare edge patterns
    if edges1.max() > 0 and edges2.max() > 0:
        diff = np.abs(edges1 - edges2).sum() / (edges1.sum() + edges2.sum())
        return (1 - diff) >= threshold
    elif edges1.max() == 0 and edges2.max() == 0:
        # Both empty
        return True
    else:
        return False

def perceptual_hash_match(img1, img2, threshold=0.8):
    """Compare using perceptual hash difference."""
    if img1.shape != img2.shape:
        return False
    
    def compute_phash(img):
        # Simple resize to 8x8 using nearest neighbor
        h, w = img.shape
        resized = np.zeros((8, 8))
        
        for y in range(8):
            for x in range(8):
                # Map to original image
                orig_y = (y * h) // 8
                orig_x = (x * w) // 8
                resized[y, x] = img[orig_y, orig_x]
        
        # Simple DCT approximation (just use average-based hash)
        avg = np.mean(resized)
        hash_bits = resized > avg
        return hash_bits.flatten()
    
    try:
        hash1 = compute_phash(img1)
        hash2 = compute_phash(img2)
        
        # Calculate similarity (64 bits total)
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1 - (hamming_dist / 64.0)
        return similarity >= threshold
    except:
        return False

def evaluate_quarter(quarter_hash, quarter_data, seen_quarters, method='md5', confidence=None):
    """Evaluate if we've seen a similar quarter before. Return (color, substitute_hash)."""
    current_quarter = quarter_data[quarter_hash]
    
    if method == 'md5':
        # Exact match using hash
        if quarter_hash in seen_quarters:
            # Return the first occurrence we saw
            return "\033[37;5m", quarter_hash  # Flashing white, use original
        else:
            seen_quarters.add(quarter_hash)
            return hash_to_color(quarter_hash), quarter_hash
    
    # Fuzzy matching - check against all seen quarters
    match_functions = {
        'hamming': (hamming_distance_match, confidence if confidence is not None else 0.9),
        'erosion': (erosion_dilation_match, confidence if confidence is not None else 0.9),
        'correlation': (correlation_match, confidence if confidence is not None else 0.8),
        'distance': (distance_transform_match, confidence if confidence is not None else 0.9),
        'phash': (perceptual_hash_match, confidence if confidence is not None else 0.8),
    }
    
    if method not in match_functions:
        method = 'hamming'  # Default fallback
    
    match_func, threshold = match_functions[method]
    
    # Check if current quarter matches any seen quarter
    for seen_hash, seen_quarter in seen_quarters.items():
        if match_func(current_quarter, seen_quarter, threshold):
            # Found a match - return the seen quarter instead
            return "\033[37;5m", seen_hash  # Flashing white, use substitute
    
    # No match found - add to seen and return original
    seen_quarters[quarter_hash] = current_quarter
    return hash_to_color(quarter_hash), quarter_hash

def quarter_to_braille_grid(quarter_array, color_code, threshold=128, scale=4):
    """Convert a quarter image to a grid of colored braille characters."""
    if quarter_array.size == 0:
        return [' ' * scale] * scale
    
    # Use provided color code
    reset_code = "\033[0m"
    
    # Get quarter dimensions
    height, width = quarter_array.shape
    
    # Create a grid of braille characters
    rows = []
    for by in range(scale):
        row = color_code  # Start each row with color
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
        
        row += reset_code  # Reset color at end of row
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

def render_text(text, scale=4, threshold=128, method='md5', confidence=None):
    """Render text using quarter-based font with line wrapping."""
    # Handle 'all' method
    if method == 'all':
        methods = ['md5', 'hamming', 'erosion', 'correlation', 'distance', 'phash']
        for m in methods:
            print(f"\n{'='*60}")
            print(f"Method: {m.upper()}")
            print(f"{'='*60}")
            render_text(text, scale, threshold, m, confidence)
        return
    # Load data
    state = load_state()
    glyph_data = state["glyphs"]
    quarter_data = state["images"]
    
    seen_quarters = set() if method == 'md5' else {}
    total_rows = scale * 2  # Top quarters + bottom quarters
    output_buffer = [''] * total_rows
    MAX_WIDTH = 80
    
    for char in text:
        if char not in glyph_data:
            # Use space for unknown characters
            empty_quarter = [' ' * scale] * scale
            char_braille = [empty_quarter, empty_quarter, empty_quarter, empty_quarter]
        else:
            char_data = glyph_data[char]
            quarter_hashes = char_data['data']
            
            # Get quarters and evaluate colors: [TL, TR, BL, BR]
            quarters = []
            colors = []
            substitute_hashes = []
            for hash_val in quarter_hashes:
                if hash_val in quarter_data:
                    color, substitute_hash = evaluate_quarter(hash_val, quarter_data, seen_quarters, method, confidence)
                    colors.append(color)
                    substitute_hashes.append(substitute_hash)
                    # Use the substitute quarter instead of the original
                    quarters.append(quarter_data[substitute_hash])
                else:
                    quarters.append(np.full((32, 16), 255, dtype=np.uint8))  # White quarter
                    colors.append("\033[37m")  # White
            
            # Convert quarters to colored braille grids (using substituted quarters)
            char_braille = [
                quarter_to_braille_grid(quarters[0], colors[0], threshold, scale),  # TL
                quarter_to_braille_grid(quarters[1], colors[1], threshold, scale),  # TR
                quarter_to_braille_grid(quarters[2], colors[2], threshold, scale),  # BL
                quarter_to_braille_grid(quarters[3], colors[3], threshold, scale),  # BR
            ]
        
        # Check if adding this character would exceed width
        char_width = scale * 2  # Each char is 2 quarters wide
        
        # Count visible characters (excluding ANSI escape sequences)
        import re
        clean_line = re.sub(r'\033\[[0-9;]*m', '', output_buffer[0])
        visible_width = len(clean_line)
        
        if visible_width + char_width > MAX_WIDTH:
            # Print current buffer and reset
            for line in output_buffer:
                print(line + "\033[0m")  # Reset at end of line
            output_buffer = [''] * total_rows
        
        # Add character to buffer
        tl, tr, bl, br = char_braille
        
        # Top half (TL + TR side by side)
        for row in range(scale):
            output_buffer[row] += tl[row] + tr[row]
        
        # Bottom half (BL + BR side by side)  
        for row in range(scale):
            output_buffer[scale + row] += bl[row] + br[row]
    
    # Print final buffer
    for line in output_buffer:
        print(line + "\033[0m")

def main():
    parser = argparse.ArgumentParser(description='Render text using quarter-based braille font')
    parser.add_argument('text', help='Text to render')
    parser.add_argument('--scale', type=int, default=3, 
                       help='Size of each quarter in characters (default: 3)')
    parser.add_argument('--threshold', type=int, default=180, 
                       help='Darkness threshold for braille dots (default: 180)')
    parser.add_argument('--method', default='md5', 
                       choices=['md5', 'hamming', 'erosion', 'correlation', 'distance', 'phash', 'all'],
                       help='Similarity method (default: md5)')
    parser.add_argument('--confidence', type=float, default=None,
                       help='Confidence threshold 0.0-1.0 (default: varies by method)')
    
    args = parser.parse_args()
    render_text(args.text, args.scale, args.threshold, args.method, args.confidence)

if __name__ == "__main__":
    main()

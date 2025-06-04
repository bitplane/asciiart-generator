#!/usr/bin/env python3
"""Debug the bleed detection to see what's going wrong."""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def debug_char(font_path, char, font_size=16):
    """Debug a single character to see what's happening."""
    font = ImageFont.truetype(font_path, font_size)
    
    # Get font metrics using space character for true cell size
    space_bbox = font.getbbox(' ')
    ascent, descent = font.getmetrics()
    print(f"Character '{char}' (U+{ord(char):04X}):")
    print(f"  Space bounding box: {space_bbox}")
    print(f"  Font metrics: ascent={ascent}, descent={descent}")
    
    if not space_bbox:
        print("  No space bounding box!")
        return
    
    cell_width = space_bbox[2] - space_bbox[0]
    cell_height = ascent + descent
    print(f"  Cell size: {cell_width}×{cell_height}")
    
    # Create 3x3 grid image
    grid_width = cell_width * 3
    grid_height = cell_height * 3
    print(f"  Grid size: {grid_width}×{grid_height}")
    
    # Create test image and render character in center
    test_img = Image.new('L', (grid_width, grid_height), color=255)
    draw = ImageDraw.Draw(test_img)
    
    # Position character in center cell using proper baseline anchoring
    center_x = cell_width
    # Place baseline in center cell 
    center_y = cell_height + ascent  # Baseline position in center cell
    try:
        # Use baseline anchor for proper positioning
        draw.text((center_x, center_y), char, font=font, fill=0, anchor='ls')
        print(f"  Using baseline anchor at ({center_x}, {center_y})")
    except TypeError:
        # Fallback for older PIL versions without anchor support
        adjusted_y = center_y - ascent
        draw.text((center_x, adjusted_y), char, font=font, fill=0)
        print(f"  Using adjusted position at ({center_x}, {adjusted_y})")
    
    # Save original for inspection
    test_img.save(f"debug_{ord(char):04X}_original.png")
    
    # Check for horizontal bleeding by blanking out center column
    test_img_blanked = test_img.copy()
    draw_blanked = ImageDraw.Draw(test_img_blanked)
    
    # Blank out the center column (where character should be contained)
    draw_blanked.rectangle([cell_width, 0, cell_width*2, grid_height], fill=255)
    
    # Save blanked for inspection
    test_img_blanked.save(f"debug_{ord(char):04X}_blanked.png")
    
    # Check if there's any non-white pixels left in left or right columns
    test_array = np.array(test_img_blanked)
    
    # Check left and right bleeding
    left_bleed = np.any(test_array[:, :cell_width] < 255)
    right_bleed = np.any(test_array[:, cell_width*2:] < 255)
    
    if left_bleed or right_bleed:
        print(f"  HORIZONTAL BLEEDING DETECTED!")
        print(f"  Left bleed: {left_bleed}, Right bleed: {right_bleed}")
        
        # Show where the bleeding is
        coords = np.where(test_array < 255)
        if len(coords[0]) > 0:
            bleeding_coords = list(zip(coords[1][:10], coords[0][:10]))
            print(f"  Bleeding pixels at: {bleeding_coords}")
    else:
        print(f"  No horizontal bleeding")
    
    # Show character's actual bounding box
    char_bbox = font.getbbox(char)
    print(f"  Character bbox: {char_bbox}")

def main():
    font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
    
    # Test known bleeding and non-bleeding characters based on user observations
    test_chars = [
        'A', 'Å',  # User says Å doesn't bleed in terminal
        '🄞', '🄟', '🄠', '🄡', '🄢',  # User says these bleed
        '⸺',  # User says this cuts through 1.5 chars 
        '🅱', '🅲', '🅳', '🅴', '🅵'  # User says these are multi-width that bleed
    ]
    
    for char in test_chars:
        print()
        debug_char(font_path, char)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Test ⸺ character rendering in different fonts to see which ones bleed."""

from PIL import Image, ImageDraw, ImageFont
import os

def test_char_in_font(font_path, font_name, char='⸺', font_size=16):
    """Test how a character renders in a specific font."""
    if not os.path.exists(font_path):
        print(f"Font not found: {font_path}")
        return
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading {font_name}: {e}")
        return
    
    # Get space character for cell sizing
    space_bbox = font.getbbox(' ')
    if not space_bbox:
        print(f"No space bbox for {font_name}")
        return
    
    cell_width = space_bbox[2] - space_bbox[0]
    ascent, descent = font.getmetrics()
    cell_height = ascent + descent
    
    # Create 3x3 grid
    grid_width = cell_width * 3
    grid_height = cell_height * 3
    
    # Draw character in center
    img = Image.new('L', (grid_width, grid_height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Draw grid lines for reference
    for i in range(4):
        x = i * cell_width
        draw.line([(x, 0), (x, grid_height)], fill=128)
    for i in range(4):
        y = i * cell_height
        draw.line([(0, y), (grid_width, y)], fill=192)
    
    # Position character in center cell
    center_x = cell_width
    center_y = cell_height + ascent
    
    try:
        draw.text((center_x, center_y), char, font=font, fill=0, anchor='ls')
    except TypeError:
        # Fallback for older PIL
        adjusted_y = center_y - ascent
        draw.text((center_x, adjusted_y), char, font=font, fill=0)
    
    # Save image
    safe_name = font_name.replace('/', '_').replace(' ', '_')
    img.save(f"bleed_test_{safe_name}.png")
    
    # Get character bbox
    char_bbox = font.getbbox(char)
    print(f"{font_name}:")
    print(f"  Cell: {cell_width}×{cell_height}, Char bbox: {char_bbox}")
    
    # Check if character width exceeds cell width
    if char_bbox:
        char_width = char_bbox[2] - char_bbox[0]
        if char_width > cell_width:
            print(f"  POTENTIAL BLEED: char width {char_width} > cell width {cell_width}")
        else:
            print(f"  OK: char width {char_width} <= cell width {cell_width}")

def main():
    # Test key fonts from the fallback chain
    fonts_to_test = [
        ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "NotoSans-Regular"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans"),  
        ("/usr/share/fonts/truetype/noto/NotoSansMath-Regular.ttf", "NotoSansMath"),
        ("/usr/share/fonts/truetype/noto/NotoSansSymbols-Regular.ttf", "NotoSansSymbols"),
        ("/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf", "NotoSansSymbols2"),
        ("/usr/share/fonts/opentype/stix-word/STIX-Regular.otf", "STIX-Regular"),
        ("/usr/share/fonts/truetype/freefont/FreeMono.ttf", "FreeMono"),
        ("/usr/share/fonts/truetype/freefont/FreeSans.ttf", "FreeSans"),
    ]
    
    char = '⸺'
    print(f"Testing character '{char}' (U+{ord(char):04X}) in different fonts:")
    print()
    
    for font_path, font_name in fonts_to_test:
        test_char_in_font(font_path, font_name, char)
        print()

if __name__ == "__main__":
    main()
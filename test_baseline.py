#!/usr/bin/env python3
"""Test to understand PIL's text anchor/baseline behavior."""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def test_baselines():
    font_path = "/usr/share/fonts/truetype/noto/NotoSansMono-SemiCondensedSemiBold.ttf"
    font = ImageFont.truetype(font_path, 16)
    
    # Get character metrics
    char_bbox = font.getbbox('A')
    ascent, descent = font.getmetrics()
    
    print(f"'A' bbox: {char_bbox}")
    print(f"Ascent: {ascent}, Descent: {descent}")
    
    # Create test image
    test_img = Image.new('L', (50, 100), color=255)
    draw = ImageDraw.Draw(test_img)
    
    # Draw reference line at y=50
    draw.line([(0, 50), (50, 50)], fill=128)
    
    # Try different anchor points
    try:
        # Modern PIL has anchor parameter
        draw.text((10, 50), 'A', font=font, fill=0, anchor='ls')  # left-baseline
        print("Used anchor='ls' (left-baseline)")
    except TypeError:
        # Older PIL doesn't have anchor
        draw.text((10, 50), 'A', font=font, fill=0)
        print("Used default positioning (no anchor)")
    
    test_img.save("baseline_test.png")
    
    # Find where 'A' actually appears
    test_array = np.array(test_img)
    black_coords = np.where(test_array == 0)
    if len(black_coords[0]) > 0:
        min_y = np.min(black_coords[0])
        max_y = np.max(black_coords[0])
        print(f"'A' appears from y={min_y} to y={max_y}")
        print(f"Reference line at y=50")
        print(f"Character extends {50 - min_y} pixels above and {max_y - 50} pixels below reference")

if __name__ == "__main__":
    test_baselines()
#!/usr/bin/env python3
"""Test to understand PIL text positioning."""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def test_positioning():
    font_path = "/usr/share/fonts/truetype/noto/NotoSansMono-SemiCondensedSemiBold.ttf"
    font = ImageFont.truetype(font_path, 16)
    
    # Get space character metrics
    space_bbox = font.getbbox(' ')
    ascent, descent = font.getmetrics()
    
    print(f"Space bbox: {space_bbox}")
    print(f"Ascent: {ascent}, Descent: {descent}")
    
    # Create a test image and draw grid lines
    test_img = Image.new('L', (50, 80), color=255)
    draw = ImageDraw.Draw(test_img)
    
    # Draw grid lines every 10 pixels for reference
    for y in range(0, 80, 10):
        draw.line([(0, y), (50, y)], fill=128)
    for x in range(0, 50, 10):
        draw.line([(x, 0), (x, 80)], fill=128)
    
    # Test different baseline positions for 'A'
    positions = [20, 30, 40, 50]
    
    for i, y_pos in enumerate(positions):
        x_pos = 10 + i * 10
        draw.text((x_pos, y_pos), 'A', font=font, fill=0)
        print(f"Drew 'A' at position ({x_pos}, {y_pos})")
    
    # Save the test image
    test_img.save("positioning_test.png")
    
    # Check where pixels are non-white (excluding grid lines)
    test_array = np.array(test_img)
    black_coords = np.where(test_array == 0)  # Only pure black (text)
    black_pixels = list(zip(black_coords[1][:20], black_coords[0][:20]))
    print(f"First 20 black pixels (text) at: {black_pixels}")
    
    # Find the range of y-coordinates where text appears
    if len(black_coords[0]) > 0:
        min_y = np.min(black_coords[0])
        max_y = np.max(black_coords[0])
        print(f"Text appears from y={min_y} to y={max_y}")
        
        # Show which y-positions we requested vs where text actually appeared
        print(f"Requested y-positions: {positions}")
        print(f"Actual text spans: y={min_y} to y={max_y}")

if __name__ == "__main__":
    test_positioning()
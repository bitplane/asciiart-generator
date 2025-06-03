#!/usr/bin/env python3

import os
import glob
import subprocess
from fontTools.ttLib import TTFont
import unicodedata

def get_font_chars(font_path):
    """Get all characters defined in a font."""
    try:
        ttf = TTFont(font_path)
        chars = set()
        for table in ttf['cmap'].tables:
            chars.update(table.cmap.keys())
        ttf.close()
        return sorted(chars)
    except:
        return []

def get_cursor_pos():
    """Get cursor position using ANSI escape sequence."""
    import sys, tty, termios
    
    # Save terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        # Set terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        
        # Query cursor position
        sys.stdout.write('\033[6n')
        sys.stdout.flush()
        
        # Read response
        response = ''
        while True:
            ch = sys.stdin.read(1)
            response += ch
            if ch == 'R':
                break
        
        # Parse response: ESC[row;colR
        parts = response[2:-1].split(';')
        row = int(parts[0])
        col = int(parts[1])
        
        return row, col
        
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def measure_char_width(char):
    """Measure character width by printing it and checking cursor movement."""
    import sys
    
    # Move to start of line
    sys.stdout.write('\r')
    sys.stdout.flush()
    
    # Print the character
    sys.stdout.write(char)
    sys.stdout.flush()
    
    # Get cursor position
    row, col = get_cursor_pos()
    
    # Clear line
    sys.stdout.write('\r\033[K')
    sys.stdout.flush()
    
    return col - 1  # Subtract 1 because columns are 1-indexed

def find_monospace_fonts():
    """Find all monospace fonts on the system."""
    font_paths = []
    font_dirs = [
        "/usr/share/fonts/truetype/",
        "/usr/share/fonts/opentype/",
    ]
    
    mono_keywords = ['mono', 'code', 'term', 'fixed', 'console']
    
    for font_dir in font_dirs:
        if not os.path.exists(font_dir):
            continue
        for ext in ['*.ttf', '*.otf']:
            for font_path in glob.glob(os.path.join(font_dir, '**', ext), recursive=True):
                basename = os.path.basename(font_path).lower()
                if any(keyword in basename for keyword in mono_keywords):
                    font_paths.append(font_path)
    
    return sorted(list(set(font_paths)))

def main():
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print("Finding monospace fonts...")
    fonts = find_monospace_fonts()
    print(f"Found {len(fonts)} fonts")
    
    # Set up terminal for raw mode operations
    import sys
    sys.stdout.write('\033[?25l')  # Hide cursor
    sys.stdout.flush()
    
    try:
        for font_path in fonts:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            print(f"\nProcessing {font_name}...")
            
            # Get characters from font
            chars = get_font_chars(font_path)
            if not chars:
                print(f"  Could not read font")
                continue
            
            print(f"  Found {len(chars)} characters")
            
            # Group characters by width
            width_groups = {}
            
            for code in chars:
                try:
                    char = chr(code)
                    
                    # Skip control characters
                    if unicodedata.category(char).startswith('C'):
                        continue
                    
                    # Measure actual width using cursor position
                    width = measure_char_width(char)
                    
                    if width not in width_groups:
                        width_groups[width] = []
                    width_groups[width].append(char)
                    
                except:
                    continue
            
            # Save each width group to a file
            for width, chars in width_groups.items():
                output_file = f"data/{font_name}.{width}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(''.join(chars))
                print(f"  Width {width}: {len(chars)} chars -> {output_file}")
    
    finally:
        # Show cursor again
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()

if __name__ == "__main__":
    main()
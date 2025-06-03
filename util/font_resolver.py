#!/usr/bin/env python3
"""Font resolution utilities for terminal applications."""

import subprocess
import re
import argparse
from PIL import ImageFont
from fontTools.ttLib import TTFont

def get_gnome_terminal_font():
    """Get the configured font for gnome-terminal."""
    try:
        # Check if custom font is enabled
        result = subprocess.run([
            'gsettings', 'get', 'org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:default/', 'use-system-font'
        ], capture_output=True, text=True)
        
        use_system = result.stdout.strip() == 'true'
        
        if use_system:
            # Get system monospace font
            result = subprocess.run([
                'gsettings', 'get', 'org.gnome.desktop.interface', 'monospace-font-name'
            ], capture_output=True, text=True)
        else:
            # Get terminal-specific font
            result = subprocess.run([
                'gsettings', 'get', 'org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:default/', 'font'
            ], capture_output=True, text=True)
        
        font_string = result.stdout.strip().strip("'\"")
        return parse_font_string(font_string)
        
    except Exception as e:
        print(f"Error getting gnome-terminal font: {e}")
        return None

def parse_font_string(font_string):
    """Parse a font string like 'Ubuntu Mono 12' into (family, size)."""
    # Split on last space to separate size
    parts = font_string.rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    else:
        return (font_string, 12)  # Default size

def get_fontconfig_fallbacks(font_family):
    """Get the fontconfig fallback chain for a font family."""
    try:
        result = subprocess.run([
            'fc-match', '-s', font_family
        ], capture_output=True, text=True)
        
        fallbacks = []
        for line in result.stdout.strip().split('\n'):
            # Parse lines like: "Ubuntu Mono:style=Regular:file=/path/to/font.ttf"
            match = re.match(r'^([^:]+)', line)
            if match:
                fallbacks.append(match.group(1))
        
        return fallbacks
        
    except Exception as e:
        print(f"Error getting fontconfig fallbacks: {e}")
        return []

def find_font_path(font_family):
    """Find the actual file path for a font family."""
    try:
        result = subprocess.run([
            'fc-match', '-f', '%{file}', font_family
        ], capture_output=True, text=True)
        
        return result.stdout.strip()
        
    except Exception as e:
        print(f"Error finding font path for {font_family}: {e}")
        return None

def font_has_glyph(font_path, char):
    """Check if a font file contains a specific glyph."""
    try:
        ttf = TTFont(font_path)
        char_code = ord(char)
        
        for table in ttf['cmap'].tables:
            if char_code in table.cmap:
                ttf.close()
                return True
        
        ttf.close()
        return False
        
    except Exception:
        return False

def resolve_char_font(char):
    """Determine which font will be used to render a character in the terminal."""
    # Get terminal font configuration
    terminal_font = get_gnome_terminal_font()
    if not terminal_font:
        return None
    
    font_family, font_size = terminal_font
    
    # Get fallback chain
    fallbacks = get_fontconfig_fallbacks(font_family)
    
    # Test each font in the chain
    for font_name in fallbacks:
        font_path = find_font_path(font_name)
        if font_path and font_has_glyph(font_path, char):
            return {
                'font_family': font_name,
                'font_path': font_path,
                'font_size': font_size
            }
    
    return None

def get_terminal_font_chain():
    """Get the complete font fallback chain for the terminal."""
    terminal_font = get_gnome_terminal_font()
    if not terminal_font:
        return []
    
    font_family, font_size = terminal_font
    fallbacks = get_fontconfig_fallbacks(font_family)
    
    font_chain = []
    for font_name in fallbacks:
        font_path = find_font_path(font_name)
        if font_path:
            font_chain.append({
                'font_family': font_name,
                'font_path': font_path,
                'font_size': font_size
            })
    
    return font_chain

def main():
    parser = argparse.ArgumentParser(description='Font resolution utilities')
    parser.add_argument('--char', help='Test which font renders a specific character')
    parser.add_argument('--show-chain', action='store_true', help='Show complete font fallback chain')
    parser.add_argument('--terminal-font', action='store_true', help='Show terminal font configuration')
    args = parser.parse_args()
    
    if args.terminal_font:
        font = get_gnome_terminal_font()
        if font:
            print(f"Terminal font: {font[0]} {font[1]}pt")
        else:
            print("Could not determine terminal font")
    
    if args.show_chain:
        print("Font fallback chain:")
        chain = get_terminal_font_chain()
        for i, font_info in enumerate(chain):
            print(f"  {i+1}. {font_info['font_family']}")
            print(f"     Path: {font_info['font_path']}")
    
    if args.char:
        result = resolve_char_font(args.char)
        if result:
            print(f"Character '{args.char}' (U+{ord(args.char):04X}) will use:")
            print(f"  Font: {result['font_family']}")
            print(f"  Path: {result['font_path']}")
            print(f"  Size: {result['font_size']}pt")
        else:
            print(f"Could not resolve font for character '{args.char}'")

if __name__ == "__main__":
    main()
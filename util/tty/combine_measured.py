#!/usr/bin/env python3

import os
import glob

def combine_width_files(width):
    """Combine all files with given width into a single sorted set."""
    all_chars = set()
    
    # Find all files matching pattern
    pattern = f"data/fonts/*.{width}.txt"
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} files for width {width}")
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                all_chars.update(content)
                print(f"  {os.path.basename(filepath)}: {len(content)} chars")
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
    
    return all_chars

def main():
    # Create output directory
    os.makedirs('data/glyphs', exist_ok=True)
    
    # Process width 1 and 2 files
    for width in [1, 2]:
        print(f"\nProcessing width {width} files...")
        
        chars = combine_width_files(width)
        
        if chars:
            # Sort and save
            sorted_chars = ''.join(sorted(chars))
            output_file = f"data/glyphs/{width}-char.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(sorted_chars)
            
            print(f"Saved {len(chars)} unique characters to {output_file}")
        else:
            print(f"No characters found for width {width}")

if __name__ == "__main__":
    main()

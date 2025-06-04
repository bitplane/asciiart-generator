#!/usr/bin/env python3

import os
import glob
import argparse

def combine_by_suffix(suffix):
    """Combine all files with given suffix into a single sorted set."""
    all_chars = set()
    
    # Find all files matching pattern
    pattern = f"data/fonts/*.{suffix}.txt"
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} files for suffix '{suffix}'")
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                all_chars.update(content)
                print(f"  {os.path.basename(filepath)}: {len(content)} chars")
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
    
    return all_chars

def auto_combine_all():
    """Automatically find all suffixes and combine them."""
    # Find all .txt files in fonts directory
    files = glob.glob("data/fonts/*.txt")
    
    # Extract suffixes
    suffixes = set()
    for filepath in files:
        basename = os.path.basename(filepath)
        # Remove .txt and split on dots
        name_parts = basename[:-4].split('.')
        if len(name_parts) > 1:
            suffix = '.'.join(name_parts[1:])  # Everything after first dot
            suffixes.add(suffix)
    
    print(f"Found suffixes: {sorted(suffixes)}")
    
    # Create output directory
    os.makedirs('data/glyphs', exist_ok=True)
    
    # Process each suffix
    for suffix in sorted(suffixes):
        print(f"\nProcessing suffix '{suffix}'...")
        
        chars = combine_by_suffix(suffix)
        
        if chars:
            # Sort and save
            sorted_chars = ''.join(sorted(chars))
            output_file = f"data/glyphs/{suffix}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(sorted_chars)
            
            print(f"Saved {len(chars)} unique characters to {output_file}")
        else:
            print(f"No characters found for suffix '{suffix}'")

def main():
    parser = argparse.ArgumentParser(description='Combine measured font character files')
    parser.add_argument('--suffix', help='Combine files with specific suffix (e.g., "1", "1-bleed")')
    parser.add_argument('--auto', action='store_true', help='Auto-discover and combine all suffixes')
    args = parser.parse_args()
    
    if args.suffix:
        chars = combine_by_suffix(args.suffix)
        if chars:
            os.makedirs('data/glyphs', exist_ok=True)
            sorted_chars = ''.join(sorted(chars))
            output_file = f"data/glyphs/{args.suffix}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(sorted_chars)
            print(f"Saved {len(chars)} unique characters to {output_file}")
    
    elif args.auto:
        auto_combine_all()
    
    else:
        # Default behavior - combine width 1 and 2
        os.makedirs('data/glyphs', exist_ok=True)
        for suffix in ['1', '2']:
            chars = combine_by_suffix(suffix)
            if chars:
                sorted_chars = ''.join(sorted(chars))
                output_file = f"data/glyphs/{suffix}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(sorted_chars)
                print(f"Saved {len(chars)} unique characters to {output_file}")

if __name__ == "__main__":
    main()

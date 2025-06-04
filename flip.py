#!/usr/bin/env python3
"""
Create flipped dataset with (flipx, flipy, md5) keys.
"""

import pickle
import numpy as np
import hashlib


def hash_array(arr):
    """Create MD5 hash of numpy array."""
    return hashlib.md5(arr.tobytes()).hexdigest()


def main():
    # Load original data
    print("Loading quarter_data.pkl...")
    with open('quarter_data.pkl', 'rb') as f:
        quarter_data = pickle.load(f)
    
    print(f"Loaded {len(quarter_data):,} quarters")
    
    # Create flipped dataset
    flipped_data = {}
    
    print("Creating flipped dataset...")
    for i, (original_md5, quarter) in enumerate(quarter_data.items()):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(quarter_data):,}")
        
        # Original (0, 0, md5)
        flipped_data[(0, 0, original_md5)] = quarter
        
        # Flip X (1, 0, md5) - horizontal flip
        flipped_x = np.fliplr(quarter)
        flipped_x_md5 = hash_array(flipped_x)
        flipped_data[(1, 0, flipped_x_md5)] = flipped_x
        
        # Flip Y (0, 1, md5) - vertical flip
        flipped_y = np.flipud(quarter)
        flipped_y_md5 = hash_array(flipped_y)
        flipped_data[(0, 1, flipped_y_md5)] = flipped_y
        
        # Flip both X and Y (1, 1, md5)
        flipped_xy = np.flipud(np.fliplr(quarter))
        flipped_xy_md5 = hash_array(flipped_xy)
        flipped_data[(1, 1, flipped_xy_md5)] = flipped_xy
    
    print(f"Created {len(flipped_data):,} total quarters")
    
    # Save flipped dataset
    print("Saving quarter_data_flipped.pkl...")
    with open('quarter_data_flipped.pkl', 'wb') as f:
        pickle.dump(flipped_data, f)
    
    print("Complete!")


if __name__ == "__main__":
    main()
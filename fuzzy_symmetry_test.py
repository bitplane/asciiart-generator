#!/usr/bin/env python3
"""
Test different fuzzy comparison methods for detecting symmetric quarters.
"""

import pickle
import numpy as np
import hashlib
import time
from scipy import ndimage
from scipy.signal import correlate2d
from skimage.metrics import structural_similarity as ssim
import cv2
from collections import defaultdict

def get_quarter_hash(quarter_array):
    """Generate MD5 hash for a quarter image array."""
    quarter_bytes = quarter_array.tobytes()
    return hashlib.md5(quarter_bytes).hexdigest()

def hamming_distance_match(img1, img2, threshold=0.1):
    """Compare using Hamming distance with threshold."""
    if img1.shape != img2.shape:
        return False
    
    # Ensure same size and type
    if img1.shape != img2.shape:
        return False
    
    # Convert to binary if needed
    binary1 = (img1 < 128).astype(np.uint8)
    binary2 = (img2 < 128).astype(np.uint8)
    
    # Count differing pixels
    diff_pixels = np.sum(binary1 != binary2)
    total_pixels = img1.size
    
    similarity = 1 - (diff_pixels / total_pixels)
    return similarity > (1 - threshold)

def cross_correlation_match(img1, img2, threshold=0.8):
    """Compare using cross-correlation with small shifts."""
    if img1.shape != img2.shape:
        return False
    
    # Convert to binary
    binary1 = (img1 < 128).astype(np.float32)
    binary2 = (img2 < 128).astype(np.float32)
    
    best_correlation = 0
    
    # Try small shifts
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            # Shift img2
            shifted = np.roll(np.roll(binary2, dx, axis=1), dy, axis=0)
            
            # Calculate normalized cross-correlation
            correlation = np.corrcoef(binary1.flatten(), shifted.flatten())[0, 1]
            if not np.isnan(correlation):
                best_correlation = max(best_correlation, correlation)
    
    return best_correlation > threshold

def erosion_dilation_match(img1, img2, threshold=0.1):
    """Compare after erosion/dilation to handle aliasing."""
    if img1.shape != img2.shape:
        return False
    
    # Convert to binary
    binary1 = (img1 < 128).astype(np.uint8)
    binary2 = (img2 < 128).astype(np.uint8)
    
    # Apply morphological operations
    kernel = np.ones((2,2), np.uint8)
    
    # Try both erosion and dilation
    eroded1 = cv2.erode(binary1, kernel, iterations=1)
    eroded2 = cv2.erode(binary2, kernel, iterations=1)
    
    dilated1 = cv2.dilate(binary1, kernel, iterations=1)
    dilated2 = cv2.dilate(binary2, kernel, iterations=1)
    
    # Test both versions
    eroded_diff = np.sum(eroded1 != eroded2) / eroded1.size
    dilated_diff = np.sum(dilated1 != dilated2) / dilated1.size
    
    best_similarity = 1 - min(eroded_diff, dilated_diff)
    return best_similarity > (1 - threshold)

def distance_transform_match(img1, img2, threshold=0.8):
    """Compare using distance transforms."""
    if img1.shape != img2.shape:
        return False
    
    # Convert to binary
    binary1 = (img1 < 128).astype(np.uint8)
    binary2 = (img2 < 128).astype(np.uint8)
    
    # Calculate distance transforms
    dt1 = ndimage.distance_transform_edt(1 - binary1)
    dt2 = ndimage.distance_transform_edt(1 - binary2)
    
    # Normalize
    if dt1.max() > 0:
        dt1 = dt1 / dt1.max()
    if dt2.max() > 0:
        dt2 = dt2 / dt2.max()
    
    # Calculate correlation
    correlation = np.corrcoef(dt1.flatten(), dt2.flatten())[0, 1]
    
    if np.isnan(correlation):
        return False
    
    return correlation > threshold

def structural_similarity_match(img1, img2, threshold=0.7):
    """Compare using structural similarity index."""
    if img1.shape != img2.shape:
        return False
    
    # SSIM requires at least 7x7 images
    if min(img1.shape) < 7:
        # Pad smaller images
        pad_y = max(0, 7 - img1.shape[0])
        pad_x = max(0, 7 - img1.shape[1])
        
        img1_padded = np.pad(img1, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=255)
        img2_padded = np.pad(img2, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=255)
        
        try:
            similarity = ssim(img1_padded, img2_padded, data_range=255)
            return similarity > threshold
        except:
            return False
    
    try:
        similarity = ssim(img1, img2, data_range=255)
        return similarity > threshold
    except:
        return False

def perceptual_hash_match(img1, img2, threshold=10):
    """Compare using perceptual hash difference."""
    if img1.shape != img2.shape:
        return False
    
    def compute_phash(img):
        # Resize to 8x8
        resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        # Convert to grayscale if needed
        if len(resized.shape) > 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate DCT
        dct = cv2.dct(np.float32(resized))
        
        # Keep only top-left 8x8
        dct_reduced = dct[:8, :8]
        
        # Calculate median
        median = np.median(dct_reduced)
        
        # Generate hash
        hash_bits = dct_reduced > median
        return hash_bits.flatten()
    
    try:
        hash1 = compute_phash(img1)
        hash2 = compute_phash(img2)
        
        # Calculate Hamming distance
        hamming_dist = np.sum(hash1 != hash2)
        return hamming_dist <= threshold
    except:
        return False

def test_all_methods():
    """Test all fuzzy comparison methods."""
    print("Loading quarter data...")
    with open('quarter_data.pkl', 'rb') as f:
        quarter_data = pickle.load(f)
    
    print(f"Loaded {len(quarter_data):,} unique quarter patterns")
    
    # Create hash lookup for exact matches
    hash_set = set(quarter_data.keys())
    
    # Use full dataset
    sample_items = list(quarter_data.items())
    print(f"Testing on full dataset of {len(sample_items)} quarters")
    
    methods = {
        'hamming_distance': hamming_distance_match,
        'erosion_dilation': erosion_dilation_match,
        'distance_transform': distance_transform_match,
        'perceptual_hash': perceptual_hash_match,
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\nTesting {method_name}...")
        
        start_time = time.time()
        no_flip = 0
        h_matches = 0
        v_matches = 0
        hv_matches = 0
        total_tested = 0
        
        for original_hash, quarter_array in sample_items:
            # Test no flip (identity)
            if method_func(quarter_array, quarter_array):
                no_flip += 1
            
            # Test horizontal flip
            h_flipped = np.fliplr(quarter_array)
            if method_func(quarter_array, h_flipped):
                h_matches += 1
            
            # Test vertical flip
            v_flipped = np.flipud(quarter_array)
            if method_func(quarter_array, v_flipped):
                v_matches += 1
            
            # Test horizontal + vertical flip
            hv_flipped = np.flipud(np.fliplr(quarter_array))
            if method_func(quarter_array, hv_flipped):
                hv_matches += 1
            
            total_tested += 1
        
        elapsed = time.time() - start_time
        
        results[method_name] = {
            'no_flip': no_flip,
            'h_matches': h_matches,
            'v_matches': v_matches,
            'hv_matches': hv_matches,
            'total_tested': total_tested,
            'time_seconds': elapsed,
            'quarters_per_second': total_tested / elapsed if elapsed > 0 else 0
        }
        
        print(f"  No-flip: {no_flip}/{total_tested} ({no_flip/total_tested*100:.1f}%)")
        print(f"  H-flip: {h_matches}/{total_tested} ({h_matches/total_tested*100:.1f}%)")
        print(f"  V-flip: {v_matches}/{total_tested} ({v_matches/total_tested*100:.1f}%)")
        print(f"  H+V-flip: {hv_matches}/{total_tested} ({hv_matches/total_tested*100:.1f}%)")
        print(f"  Time: {elapsed:.2f}s ({total_tested/elapsed:.0f} qtr/sec)")
    
    # Add exact pixel comparison for reference
    print(f"\nExact pixel matches (from hash lookup)...")
    start_time = time.time()
    exact_no_flip = len(sample_items)  # Always 100%
    exact_h = 0
    exact_v = 0
    exact_hv = 0
    
    for original_hash, quarter_array in sample_items:
        h_flipped = np.fliplr(quarter_array)
        h_hash = get_quarter_hash(h_flipped)
        if h_hash in hash_set and h_hash != original_hash:
            exact_h += 1
        
        v_flipped = np.flipud(quarter_array)
        v_hash = get_quarter_hash(v_flipped)
        if v_hash in hash_set and v_hash != original_hash:
            exact_v += 1
        
        hv_flipped = np.flipud(np.fliplr(quarter_array))
        hv_hash = get_quarter_hash(hv_flipped)
        if hv_hash in hash_set and hv_hash != original_hash:
            exact_hv += 1
    
    elapsed = time.time() - start_time
    results['exact_pixel'] = {
        'no_flip': exact_no_flip,
        'h_matches': exact_h,
        'v_matches': exact_v,
        'hv_matches': exact_hv,
        'total_tested': len(sample_items),
        'time_seconds': elapsed,
        'quarters_per_second': len(sample_items) / elapsed
    }
    
    print(f"  No-flip: {exact_no_flip}/{len(sample_items)} (100.0%)")
    print(f"  H-flip: {exact_h}/{len(sample_items)} ({exact_h/len(sample_items)*100:.1f}%)")
    print(f"  V-flip: {exact_v}/{len(sample_items)} ({exact_v/len(sample_items)*100:.1f}%)")
    print(f"  H+V-flip: {exact_hv}/{len(sample_items)} ({exact_hv/len(sample_items)*100:.1f}%)")
    print(f"  Time: {elapsed:.2f}s ({len(sample_items)/elapsed:.0f} qtr/sec)")
    
    # Print summary table
    print("\n" + "="*78)
    print("SYMMETRY DETECTION COMPARISON")
    print("="*78)
    print(f"{'Method':<16} {'No-Flip':<8} {'H-Flip':<8} {'V-Flip':<8} {'H+V-Flip':<8} {'Time':<6} {'Qtr/s':<6}")
    print("-" * 78)
    
    for method_name, data in results.items():
        no_pct = data['no_flip'] / data['total_tested'] * 100
        h_pct = data['h_matches'] / data['total_tested'] * 100
        v_pct = data['v_matches'] / data['total_tested'] * 100
        hv_pct = data['hv_matches'] / data['total_tested'] * 100
        
        print(f"{method_name:<16} {no_pct:5.1f}%   {h_pct:5.1f}%   {v_pct:5.1f}%   {hv_pct:5.1f}%     "
              f"{data['time_seconds']:4.1f}s {data['quarters_per_second']:5.0f}")
    
    print("="*78)

if __name__ == "__main__":
    test_all_methods()
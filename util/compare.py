#!/usr/bin/env python3
"""
Quarter comparison functions for glyph analysis.
"""

import numpy as np
import argparse


def hamming_distance_match(img1, img2, threshold=0.9):
    """Compare using Hamming distance with threshold."""
    if img1.shape != img2.shape:
        return False
    binary1 = (img1 < 128).astype(np.uint8)
    binary2 = (img2 < 128).astype(np.uint8)
    diff_pixels = np.sum(binary1 != binary2)
    similarity = 1 - (diff_pixels / img1.size)
    return similarity >= threshold


def simple_erosion(binary_img):
    """Simple erosion: pixel is True only if all neighbors are True."""
    eroded = np.zeros_like(binary_img)
    h, w = binary_img.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if np.all(binary_img[y-1:y+2, x-1:x+2]):
                eroded[y, x] = True
    return eroded


def simple_dilation(binary_img):
    """Simple dilation: pixel is True if any neighbor is True."""
    dilated = np.copy(binary_img)
    h, w = binary_img.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if np.any(binary_img[y-1:y+2, x-1:x+2]):
                dilated[y, x] = True
    return dilated


def erosion_dilation_match(img1, img2, threshold=0.9):
    """Compare after erosion/dilation to handle aliasing."""
    if img1.shape != img2.shape:
        return False
    binary1 = (img1 < 128)
    binary2 = (img2 < 128)
    
    eroded1 = simple_erosion(binary1)
    eroded2 = simple_erosion(binary2)
    dilated1 = simple_dilation(binary1)
    dilated2 = simple_dilation(binary2)
    
    eroded_diff = np.sum(eroded1 != eroded2) / eroded1.size
    dilated_diff = np.sum(dilated1 != dilated2) / dilated1.size
    best_similarity = 1 - min(eroded_diff, dilated_diff)
    return best_similarity >= threshold


def correlation_match(img1, img2, threshold=0.8):
    """Compare using simple correlation."""
    if img1.shape != img2.shape:
        return False
    
    flat1 = img1.flatten().astype(float)
    flat2 = img2.flatten().astype(float)
    
    correlation = np.corrcoef(flat1, flat2)[0, 1]
    return not np.isnan(correlation) and correlation >= threshold


def distance_transform_match(img1, img2, threshold=0.9):
    """Compare using simple distance from edges."""
    if img1.shape != img2.shape:
        return False
    
    binary1 = (img1 < 128).astype(np.uint8)
    binary2 = (img2 < 128).astype(np.uint8)
    
    def count_edge_pixels(binary):
        h, w = binary.shape
        edges = np.zeros_like(binary, dtype=float)
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if binary[y, x]:
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            if binary[y+dy, x+dx]:
                                neighbors += 1
                    edges[y, x] = 8 - neighbors
        return edges
    
    edges1 = count_edge_pixels(binary1)
    edges2 = count_edge_pixels(binary2)
    
    if edges1.max() > 0 and edges2.max() > 0:
        diff = np.abs(edges1 - edges2).sum() / (edges1.sum() + edges2.sum())
        return (1 - diff) >= threshold
    elif edges1.max() == 0 and edges2.max() == 0:
        return True
    else:
        return False


def perceptual_hash_match(img1, img2, threshold=0.8):
    """Compare using perceptual hash difference."""
    if img1.shape != img2.shape:
        return False
    
    def compute_phash(img):
        h, w = img.shape
        resized = np.zeros((8, 8))
        
        for y in range(8):
            for x in range(8):
                orig_y = (y * h) // 8
                orig_x = (x * w) // 8
                resized[y, x] = img[orig_y, orig_x]
        
        avg = np.mean(resized)
        hash_bits = resized > avg
        return hash_bits.flatten()
    
    try:
        hash1 = compute_phash(img1)
        hash2 = compute_phash(img2)
        
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1 - (hamming_dist / 64.0)
        return similarity >= threshold
    except:
        return False


def get_match_function(method):
    """Get the matching function and default threshold for a method."""
    match_functions = {
        'hamming': (hamming_distance_match, 0.9),
        'erosion': (erosion_dilation_match, 0.9),
        'correlation': (correlation_match, 0.8),
        'distance': (distance_transform_match, 0.9),
        'phash': (perceptual_hash_match, 0.8),
    }
    
    if method not in match_functions:
        raise ValueError(f"Unknown method: {method}. Available: {list(match_functions.keys())}")
    
    return match_functions[method]


def compare_quarters(quarter1, quarter2, method='correlation', threshold=None):
    """Compare two quarters using the specified method."""
    match_func, default_threshold = get_match_function(method)
    if threshold is None:
        threshold = default_threshold
    
    return match_func(quarter1, quarter2, threshold)


def main():
    """Demo function for testing quarter comparison."""
    parser = argparse.ArgumentParser(description='Test quarter comparison functions')
    parser.add_argument('--method', default='correlation',
                       choices=['hamming', 'erosion', 'correlation', 'distance', 'phash'],
                       help='Comparison method (default: correlation)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Similarity threshold (default: varies by method)')
    
    args = parser.parse_args()
    
    # Create test quarters
    quarter1 = np.random.randint(0, 256, (16, 32), dtype=np.uint8)
    quarter2 = quarter1 + np.random.randint(-10, 10, quarter1.shape).astype(np.int16)
    quarter2 = np.clip(quarter2, 0, 255).astype(np.uint8)
    
    result = compare_quarters(quarter1, quarter2, args.method, args.threshold)
    print(f"Method: {args.method}")
    print(f"Threshold: {args.threshold or get_match_function(args.method)[1]}")
    print(f"Match: {result}")


if __name__ == "__main__":
    main()
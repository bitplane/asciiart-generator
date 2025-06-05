#!/usr/bin/env python3
"""
State management module for ansi-canvas project.
"""
import os
import pickle
from pathlib import Path

def get_state_path():
    """Get the path to the state file in the cache directory."""
    project_root = Path(__file__).parent.parent
    return project_root / "cache" / "state.pkl"

def load_state():
    """
    Load the project state from cache/state.pkl.
    
    Returns a dict with:
        - images: image data (what was in quarter_data.pkl)
        - glyphs: glyph metadata (what was in glyph_quarters.json)
    """
    state_path = get_state_path()
    
    with open(state_path, 'rb') as f:
        return pickle.load(f)

def save_state(state):
    """
    Save the project state to cache/state.pkl.
    
    Args:
        state: dict containing images and glyphs data
    """
    state_path = get_state_path()
    
    # Ensure cache directory exists
    state_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)
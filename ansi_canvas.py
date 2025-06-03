#!/usr/bin/env python3

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
import datetime
import sys
import argparse
import unicodedata
import hashlib
from fontTools.ttLib import TTFont


# Font extraction functionality
def get_monospace_chars(font_path, font_size):
    """
    Yield all unique Unicode characters defined in the font that render at exactly the same width as 'M'.
    
    Args:
        font_path: Path to the font file
        font_size: Size of the font
        
    Yields:
        tuple: (char, image) for each unique monospace character
    """
    
    # Get all characters defined in the font
    ttf = TTFont(font_path)
    defined_chars = set()
    
    for table in ttf['cmap'].tables:
        defined_chars.update(table.cmap.keys())
    
    ttf.close()
    
    # Load font for rendering
    font = ImageFont.truetype(font_path, font_size)
    
    # Create a temporary image for measurements
    temp_img = Image.new('L', (200, 100), color=255)
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Get the dimensions of 'M' as reference
    m_bbox = temp_draw.textbbox((0, 0), "M", font=font)
    m_width = m_bbox[2] - m_bbox[0]
    m_height = m_bbox[3] - m_bbox[1]
    
    # Image dimensions with small padding
    img_width = m_width + 4
    img_height = m_height + 4
    
    # Track seen images by their hash
    seen_hashes = set()
    
    # Check each defined character
    for char_code in sorted(defined_chars):
        try:
            char = chr(char_code)
            
            # Skip control characters
            if unicodedata.category(char).startswith('C'):
                continue
            
            # Measure the character width
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            
            # Check if it matches M width exactly
            if char_width != m_width:
                continue
            
            # Render the character
            img = Image.new('L', (img_width, img_height), color=255)
            draw = ImageDraw.Draw(img)
            
            # Center the character
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img_width - text_width) // 2 - bbox[0]
            y = (img_height - text_height) // 2 - bbox[1]
            
            draw.text((x, y), char, font=font, fill=0)
            
            # Get hash of the image
            img_bytes = img.tobytes()
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            
            # Skip if we've seen this exact image before
            if img_hash in seen_hashes:
                continue
                
            seen_hashes.add(img_hash)
            yield (char, img)
                
        except (ValueError, TypeError):
            # Character not renderable
            continue


# ============= Configuration =============
DEFAULT_FONT = "/usr/share/fonts/truetype/ubuntu/UbuntuSansMono[wght].ttf"
DEFAULT_FONT_SIZE = 13

UNICODE_BLOCKS = [
    (0x0020, 0x007F),   # Basic Latin (ASCII)
    (0x00A0, 0x00FF),   # Latin-1 Supplement
    (0x2500, 0x257F),   # Box Drawing (│├─┌└┐┘┤┬┴┼ etc)
    (0x2580, 0x259F),   # Block Elements (▀▄█▌▐░▒▓ etc)
    (0x25A0, 0x25FF),   # Geometric Shapes (■□▪▫● etc)
]

# ANSI style codes
RESET = '\033[0m'
BOLD = '\033[1m'
REVERSE = '\033[7m'


# ============= Neural Network =============
class ConvASCIINet(nn.Module):
    """CNN that predicts character from 3x3 patch of character-sized regions."""
    
    def __init__(self, char_dims, num_classes):
        super().__init__()
        char_w, char_h = char_dims
        
        # Input: (batch, 1, 3*char_h, 3*char_w) - 3x3 grid of characters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size after convolutions
        # After 3 pooling operations: (3*char_h//8) * (3*char_w//8) * 64
        conv_output_size = (3 * char_h // 8) * (3 * char_w // 8) * 64
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, 1, 3*char_h, 3*char_w)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class ASCIIDataset(Dataset):
    """Dataset for training ASCII prediction model with pixel patches."""
    
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Return pixel patch and target
        patch = torch.tensor(sample['patch'], dtype=torch.float32).unsqueeze(0)  # Add channel dim
        target = torch.tensor(sample['best_char_idx'], dtype=torch.long)
        return patch, target


# ============= Dataset Generation with Rendering =============
def evaluate_character_with_bleed(char, char_img, position, original_img, char_dims, downscale_factor=0.5):
    """
    Evaluate how well a character matches by rendering it in context and downscaling.
    
    Args:
        char: The character to evaluate
        char_img: Pre-rendered character image
        position: (x, y) position in character grid
        original_img: The original image
        char_dims: (width, height) of a character
        downscale_factor: How much to downscale for evaluation
        
    Returns:
        float: Score (lower is better)
    """
    char_w, char_h = char_dims
    x, y = position
    
    # Create a copy of the 3x3 region from original
    region_img = original_img.crop((
        (x-1) * char_w,
        (y-1) * char_h,
        (x+2) * char_w,
        (y+2) * char_h
    )).copy()
    
    # Paste the character in the center (remove padding from char_img)
    char_cropped = char_img.crop((2, 2, char_img.width-2, char_img.height-2))
    region_img.paste(char_cropped, (char_w, char_h))
    
    # Downscale the region
    new_size = (int(region_img.width * downscale_factor), 
                int(region_img.height * downscale_factor))
    downscaled = region_img.resize(new_size, Image.LANCZOS)
    
    # Also downscale the original region for comparison
    original_region = original_img.crop((
        (x-1) * char_w,
        (y-1) * char_h,
        (x+2) * char_w,
        (y+2) * char_h
    ))
    original_downscaled = original_region.resize(new_size, Image.LANCZOS)
    
    # Calculate MSE
    diff = np.array(downscaled, dtype=float) - np.array(original_downscaled, dtype=float)
    score = np.mean(diff ** 2)
    
    return score


def extract_features_for_position(position, original_img, char_dims):
    """Extract features for a position that can be used to predict best character."""
    char_w, char_h = char_dims
    x, y = position
    
    # Get the 3x3 region
    region = original_img.crop((
        max(0, (x-1) * char_w),
        max(0, (y-1) * char_h),
        min(original_img.width, (x+2) * char_w),
        min(original_img.height, (y+2) * char_h)
    ))
    
    # Extract various features
    features = []
    
    # 1. Average brightness in each of the 9 cells
    for dy in range(3):
        for dx in range(3):
            cell = region.crop((dx * char_w, dy * char_h, 
                               (dx + 1) * char_w, (dy + 1) * char_h))
            features.append(np.mean(cell) / 255.0)
    
    # 2. Gradients (horizontal and vertical) in center cell
    center = region.crop((char_w, char_h, 2 * char_w, 2 * char_h))
    center_arr = np.array(center)
    
    # Horizontal gradient
    h_grad = np.mean(np.abs(np.diff(center_arr, axis=1))) / 255.0
    features.append(h_grad)
    
    # Vertical gradient  
    v_grad = np.mean(np.abs(np.diff(center_arr, axis=0))) / 255.0
    features.append(v_grad)
    
    # 3. Standard deviation (texture)
    features.append(np.std(center_arr) / 255.0)
    
    # 4. Edge density (using simple edge detection)
    h_edges = np.abs(np.diff(center_arr, axis=1))  # horizontal edges
    v_edges = np.abs(np.diff(center_arr, axis=0))  # vertical edges
    edge_density = (np.mean(h_edges > 30) + np.mean(v_edges > 30)) / 2.0
    features.append(edge_density)
    
    return features


def create_training_samples_with_rendering(image_path, char_images, char_arrays, char_dims, 
                                         scale=1.0, font_path=DEFAULT_FONT, font_size=DEFAULT_FONT_SIZE):
    """Create training samples using proper rendering evaluation."""
    
    img = Image.open(image_path).convert('L')
    
    # Scale the image
    if scale != 1.0:
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    char_w, char_h = char_dims
    
    # Resize to character grid
    grid_w = img.width // char_w
    grid_h = img.height // char_h
    img = img.resize((grid_w * char_w, grid_h * char_h), Image.LANCZOS)
    
    print(f"Grid size: {grid_w}x{grid_h}")
    
    # Create character index mapping
    chars = list(char_images.keys())
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    
    # Process each position
    samples = []
    font = ImageFont.truetype(font_path, font_size)
    
    for y in range(1, grid_h - 1):
        for x in range(1, grid_w - 1):
            # Extract features for this position
            features = extract_features_for_position((x, y), img, char_dims)
            
            # Find best character using rendering evaluation
            best_char = None
            best_score = float('inf')
            
            for char, char_img in char_images.items():
                score = evaluate_character_with_bleed(
                    char, char_img, (x, y), img, char_dims
                )
                
                if score < best_score:
                    best_score = score
                    best_char = char
            
            sample = {
                'position': (x, y),
                'features': features,
                'best_char': best_char,
                'best_char_idx': char_to_idx[best_char],
                'score': float(best_score)
            }
            samples.append(sample)
        
        if y % 10 == 0:
            print(f"Progress: {y}/{grid_h-1} rows")
    
    return samples, chars


def generate_dataset(image_paths, output_dir="data", font_path=DEFAULT_FONT, 
                    font_size=DEFAULT_FONT_SIZE, scales=[1.0, 0.75, 1.25]):
    """Generate training dataset from images with rendering-based evaluation."""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/chars", exist_ok=True)
    
    print(f"Font: {font_path}")
    print("Loading character set...")
    
    # Get all monospace characters
    char_images = {}
    char_arrays = {}
    char_dims = None
    
    for char, img in get_monospace_chars(font_path, font_size):
        char_code = ord(char)
        
        # Filter by Unicode blocks
        in_block = any(start <= char_code <= end for start, end in UNICODE_BLOCKS)
        
        if in_block:
            char_images[char] = img
            if char_dims is None:
                char_dims = (img.width - 4, img.height - 4)
            
            # Pre-compute arrays
            char_crop = img.crop((2, 2, img.width-2, img.height-2))
            char_arrays[char] = np.array(char_crop)
            
            # Save character image
            img.save(f"{output_dir}/chars/{char_code:05d}.png")
    
    print(f"Loaded {len(char_images)} characters")
    print(f"Character dimensions: {char_dims}")
    
    # Process each image at multiple scales
    all_samples = []
    
    for img_path in image_paths:
        print(f"\nProcessing {img_path}...")
        for scale in scales:
            print(f"  Scale: {scale:.2f}x")
            samples, chars = create_training_samples_with_rendering(
                img_path, char_images, char_arrays, char_dims, scale, font_path, font_size
            )
            all_samples.extend(samples)
    
    # Save dataset
    with open(f"{output_dir}/dataset.json", 'w') as f:
        json.dump({
            'char_dims': char_dims,
            'num_samples': len(all_samples),
            'num_classes': len(chars),
            'chars': chars,
            'samples': all_samples
        }, f)
    
    print(f"\nTotal samples: {len(all_samples)}")
    return char_images, char_dims


# ============= Visualization =============
def create_training_patch_visualization(samples, model, char_dims, chars, idx_to_char, device, num_samples=9):
    """Create a grid showing feature patches with ground truth vs predicted characters."""
    
    # Select random samples
    import random
    selected_samples = random.sample(samples, min(num_samples, len(samples)))
    
    # Simpler 3x3 grid
    grid_size = 3
    cell_size = 120
    char_render_size = 40
    
    viz_img = Image.new('RGB', (grid_size * cell_size, grid_size * cell_size), color=(240, 240, 240))
    
    try:
        font = ImageFont.truetype(DEFAULT_FONT, 14)
        char_font = ImageFont.truetype(DEFAULT_FONT, 24)
    except:
        font = ImageFont.load_default()
        char_font = ImageFont.load_default()
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(selected_samples):
            if i >= 9:  # Safety check
                break
                
            row = i // grid_size
            col = i % grid_size
            
            x_offset = col * cell_size
            y_offset = row * cell_size
            
            # Get model prediction
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(features)
            _, predicted_idx = torch.max(outputs.data, 1)
            predicted_char = idx_to_char[predicted_idx.item()]
            ground_truth_char = sample['best_char']
            
            # Simple feature visualization - just the first 9 brightness features as a 3x3 grid
            feature_size = 30
            for fy in range(3):
                for fx in range(3):
                    feat_idx = fy * 3 + fx
                    if feat_idx < min(9, len(sample['features'])):
                        brightness = max(0, min(1, sample['features'][feat_idx]))
                        gray_val = int(brightness * 255)
                        color = (gray_val, gray_val, gray_val)
                        
                        viz_img.paste(Image.new('RGB', (feature_size//3, feature_size//3), color), 
                                    (x_offset + fx * (feature_size//3), y_offset + fy * (feature_size//3)))
            
            # Draw characters below the patch
            draw = ImageDraw.Draw(viz_img)
            
            # Ground truth (green)
            draw.text((x_offset + 5, y_offset + feature_size + 5), 
                     f"GT: {ground_truth_char}", font=font, fill=(0, 128, 0))
            
            # Prediction (green if correct, red if wrong)
            color = (0, 128, 0) if predicted_char == ground_truth_char else (128, 0, 0)
            draw.text((x_offset + 5, y_offset + feature_size + 25), 
                     f"Pred: {predicted_char}", font=font, fill=color)
            
            # Accuracy indicator
            acc_text = "✓" if predicted_char == ground_truth_char else "✗"
            draw.text((x_offset + 5, y_offset + feature_size + 45), 
                     acc_text, font=char_font, fill=color)
    
    return viz_img


# ============= Training =============
def train_model(data_dir="data", model_path="model.pth", epochs=50):
    """Train the ASCII prediction model."""
    
    print("Loading training data...")
    with open(f'{data_dir}/dataset.json', 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    chars = data['chars']
    char_dims = data['char_dims']
    
    # Create mappings
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    print(f"Classes: {len(chars)}, Samples: {len(samples)}")
    
    # Get feature size from first sample
    feature_size = len(samples[0]['features'])
    print(f"Feature size: {feature_size}")
    
    # Split data
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
    
    # Create datasets and loaders
    train_dataset = ASCIIDataset(train_samples)
    val_dataset = ASCIIDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Model, loss, optimizer
    model = CharacterPredictor(feature_size, len(chars))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Training on {device}")
    
    # TensorBoard
    writer = SummaryWriter(f'runs/ansi_canvas_v2_{datetime.datetime.now():%Y%m%d_%H%M%S}')
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == targets).sum().item()
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
        
        # Metrics
        train_acc = 100 * train_correct / len(train_dataset)
        val_acc = 100 * val_correct / len(val_dataset)
        avg_val_loss = val_loss / len(val_loader)
        
        # Scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log training patch visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            try:
                viz_img = create_training_patch_visualization(
                    val_samples, model, char_dims, chars, idx_to_char, device, num_samples=9
                )
                viz_tensor = transforms.ToTensor()(viz_img)
                writer.add_image('Training_Patches', viz_tensor, epoch)
            except Exception as e:
                print(f"  Warning: Could not create patch visualization: {e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'chars': chars,
                'char_dims': char_dims,
                'feature_size': feature_size,
            }, model_path)
            print(f"  Saved best model (val acc: {val_acc:.2f}%)")
    
    writer.close()
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print("Run: tensorboard --logdir=runs")


# ============= Inference =============
def image_to_ascii(image_path, model_path="model.pth", output_path="output.ansi", 
                   use_styles=True, use_color=True, scale=1.0):
    """Convert image to ASCII art using the trained model."""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    chars = checkpoint['chars']
    char_dims = checkpoint['char_dims']
    feature_size = checkpoint['feature_size']
    idx_to_char = checkpoint['idx_to_char']
    
    model = CharacterPredictor(feature_size, len(chars))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load and scale image
    img = Image.open(image_path).convert('L')
    
    if scale != 1.0:
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        print(f"Scaled image to {new_width}x{new_height} ({scale:.2f}x)")
    
    char_w, char_h = char_dims
    
    grid_w = img.width // char_w
    grid_h = img.height // char_h
    img = img.resize((grid_w * char_w, grid_h * char_h), Image.LANCZOS)
    
    print(f"Converting to {grid_w}x{grid_h} ASCII art...")
    
    # Generate ASCII
    ascii_grid = []
    
    with torch.no_grad():
        for y in range(grid_h):
            row = []
            for x in range(grid_w):
                # Extract features
                features = extract_features_for_position((x, y), img, char_dims)
                
                # Predict character
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                outputs = model(features_tensor)
                _, predicted = torch.max(outputs.data, 1)
                char = idx_to_char[predicted.item()]
                
                # Apply styles based on brightness
                if use_styles and not use_color:
                    center_brightness = features[4] * 255  # Center cell brightness
                    if center_brightness < 85:
                        char = f"{REVERSE}{char}{RESET}"
                    elif center_brightness < 170:
                        char = f"{BOLD}{char}{RESET}"
                
                row.append(char)
            
            ascii_grid.append(''.join(row))
            if (y + 1) % 10 == 0:
                print(f"Progress: {y+1}/{grid_h}")
    
    # Add color if requested
    if use_color:
        ascii_grid = add_color_to_ascii(image_path, ascii_grid, char_dims, use_styles)
    
    # Save
    with open(output_path, 'w') as f:
        f.write('\n'.join(ascii_grid))
    
    print(f"Saved to {output_path}")
    print(f"View with: cat {output_path}")


def add_color_to_ascii(image_path, ascii_grid, char_dims, use_styles):
    """Add ANSI colors to ASCII art."""
    
    img = Image.open(image_path).convert('RGB')
    img_gray = img.convert('L')
    
    char_w, char_h = char_dims
    grid_h = len(ascii_grid)
    grid_w = len(ascii_grid[0]) if grid_h > 0 else 0
    
    img = img.resize((grid_w * char_w, grid_h * char_h), Image.LANCZOS)
    img_gray = img_gray.resize((grid_w * char_w, grid_h * char_h), Image.LANCZOS)
    
    def rgb_to_ansi(r, g, b):
        if r == g == b:
            gray = r * 23 // 255
            return 232 + gray
        else:
            r = r * 5 // 255
            g = g * 5 // 255
            b = b * 5 // 255
            return 16 + 36 * r + 6 * g + b
    
    colored_grid = []
    for y, row in enumerate(ascii_grid):
        colored_row = []
        for x, char in enumerate(row):
            # Get average color
            cell = img.crop((x * char_w, y * char_h, (x + 1) * char_w, (y + 1) * char_h))
            avg_color = np.array(cell).mean(axis=(0, 1)).astype(int)
            color_code = rgb_to_ansi(*avg_color)
            
            # Get brightness for styles
            if use_styles:
                cell_gray = img_gray.crop((x * char_w, y * char_h, (x + 1) * char_w, (y + 1) * char_h))
                brightness = np.array(cell_gray).mean()
                
                styles = []
                if brightness < 85:
                    styles.append('7')  # Reverse
                elif brightness < 170:
                    styles.append('1')  # Bold
                
                if styles:
                    colored_char = f"\033[{';'.join(styles)};38;5;{color_code}m{char}\033[0m"
                else:
                    colored_char = f"\033[38;5;{color_code}m{char}\033[0m"
            else:
                colored_char = f"\033[38;5;{color_code}m{char}\033[0m"
            
            colored_row.append(colored_char)
        
        colored_grid.append(''.join(colored_row))
    
    return colored_grid


# ============= CLI =============
def main():
    parser = argparse.ArgumentParser(description="ANSI Canvas V2 - Image to ASCII art with rendering")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Dataset generation
    dataset_parser = subparsers.add_parser('dataset', help='Generate training dataset')
    dataset_parser.add_argument('images', nargs='+', help='Image files to process')
    dataset_parser.add_argument('--output', default='data', help='Output directory')
    dataset_parser.add_argument('--scales', nargs='+', type=float, default=[0.75, 1.0, 1.25], 
                               help='Image scales to use (default: 0.75 1.0 1.25)')
    
    # Training
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', default='data', help='Dataset directory')
    train_parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    train_parser.add_argument('--output', default='model.pth', help='Model output path')
    
    # Inference
    convert_parser = subparsers.add_parser('convert', help='Convert image to ASCII')
    convert_parser.add_argument('image', help='Image to convert')
    convert_parser.add_argument('--model', default='model.pth', help='Model path')
    convert_parser.add_argument('--output', default='output.ansi', help='Output file')
    convert_parser.add_argument('--no-color', action='store_true', help='Disable ANSI colors')
    convert_parser.add_argument('--no-style', action='store_true', help='Disable bold/reverse styles')
    convert_parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for image (default: 1.0)')
    
    args = parser.parse_args()
    
    if args.command == 'dataset':
        generate_dataset(args.images, args.output, scales=args.scales)
    
    elif args.command == 'train':
        train_model(args.data, args.output, args.epochs)
    
    elif args.command == 'convert':
        image_to_ascii(
            args.image,
            args.model,
            args.output,
            use_styles=not args.no_style,
            use_color=not args.no_color,
            scale=args.scale
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

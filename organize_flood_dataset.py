#!/usr/bin/env python3
"""
Converts flood detection segmentation dataset to classification format.

Takes images and their corresponding binary segmentation masks and converts 
them into a binary classification problem based on flood percentage.
"""
import argparse
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


def calculate_flood_percentage(mask_path: Path, threshold: float = 0.05) -> str:
    """
    Calculate the flood percentage in a mask and classify as flooded or not_flooded.
    
    Args:
        mask_path: Path to the binary mask image
        threshold: Minimum percentage of flood pixels to classify as "flooded"
    
    Returns:
        "flooded" or "not_flooded" based on the flood percentage
    """
    try:
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # Calculate the percentage of flood pixels (value 1)
        total_pixels = mask_array.size
        flood_pixels = np.sum(mask_array == 1)
        flood_percentage = flood_pixels / total_pixels
        
        # Classify based on threshold
        return "flooded" if flood_percentage > threshold else "not_flooded"
    except Exception as e:
        print(f"Warning: Could not process mask {mask_path}: {e}")
        return "not_flooded"  # Default to not flooded if we can't read the mask


def organize_dataset(
    images_dir: Path, 
    labels_dir: Path, 
    output_dir: Path, 
    flood_threshold: float = 0.05,
    force: bool = False
):
    """
    Organize the dataset into ImageFolder format.
    
    Args:
        images_dir: Directory containing original images
        labels_dir: Directory containing binary masks
        output_dir: Output directory for organized dataset
        flood_threshold: Minimum flood percentage to classify as flooded
        force: Whether to overwrite existing output directory
    """
    if output_dir.exists():
        if force:
            print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory already exists: {output_dir}")
            print("Use --force to overwrite")
            return
    
    # Create class directories
    flooded_dir = output_dir / "flooded"
    not_flooded_dir = output_dir / "not_flooded"
    flooded_dir.mkdir(parents=True, exist_ok=True)
    not_flooded_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(images_dir.glob(ext))
        image_files.extend(images_dir.glob(ext.upper()))
    
    print(f"Found {len(image_files)} images")
    
    flooded_count = 0
    not_flooded_count = 0
    processed = 0
    
    for image_path in image_files:
        # Find corresponding label
        # Assuming labels follow pattern: image_X.jpg -> label_X.png
        image_stem = image_path.stem
        if image_stem.startswith('image_'):
            label_num = image_stem.replace('image_', '')
            label_path = labels_dir / f"label_{label_num}.png"
        else:
            # Try direct name match with different extension
            label_path = labels_dir / f"{image_stem}.png"
        
        if not label_path.exists():
            print(f"Warning: No corresponding label found for {image_path}")
            continue
        
        # Classify the image
        classification = calculate_flood_percentage(label_path, flood_threshold)
        
        # Copy image to appropriate class folder
        if classification == "flooded":
            target_dir = flooded_dir
            flooded_count += 1
        else:
            target_dir = not_flooded_dir
            not_flooded_count += 1
        
        target_path = target_dir / image_path.name
        shutil.copy2(image_path, target_path)
        processed += 1
        
        if processed % 50 == 0:
            print(f"Processed {processed} images...")
    
    print(f"\nDataset organization complete!")
    print(f"Total processed: {processed}")
    print(f"Flooded: {flooded_count}")
    print(f"Not flooded: {not_flooded_count}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Organize flood dataset for classification")
    parser.add_argument("--images_dir", required=True, help="Directory containing images")
    parser.add_argument("--labels_dir", required=True, help="Directory containing label masks")
    parser.add_argument("--output_dir", required=True, help="Output directory for organized dataset")
    parser.add_argument("--flood_threshold", type=float, default=0.05, 
                       help="Minimum flood percentage to classify as flooded (default: 0.05 = 5%)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    organize_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        flood_threshold=args.flood_threshold,
        force=args.force
    )


if __name__ == "__main__":
    main()
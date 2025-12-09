#!/usr/bin/env python3
"""
Merge the existing flood dataset with the Comprehensive Disaster Dataset (CDD).
This script combines:
- data/_organized/flooded + CDD/Water_Disaster -> merged flooded class
- data/_organized/not_flooded + CDD/Non_Damage -> merged not_flooded class
"""
import argparse
import shutil
from pathlib import Path
from typing import Set

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    """Check if a path is an image file."""
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def copy_images(source_dir: Path, dest_dir: Path, prefix: str = "") -> int:
    """Copy all images from source_dir to dest_dir with optional prefix.
    
    Returns the number of images copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    
    # Recursively find all images
    for img_path in source_dir.rglob("*"):
        if is_image(img_path):
            # Generate destination filename
            if prefix:
                dest_name = f"{prefix}_{img_path.name}"
            else:
                dest_name = img_path.name
            
            dest_path = dest_dir / dest_name
            
            # Handle duplicate names by adding a counter
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                stem = original_dest.stem
                suffix = original_dest.suffix
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.copy2(img_path, dest_path)
            count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Merge existing flood dataset with CDD dataset"
    )
    parser.add_argument(
        "--existing_data",
        default="data/_organized",
        help="Path to existing organized dataset",
    )
    parser.add_argument(
        "--cdd_root",
        default="Comprehensive Disaster Dataset(CDD)",
        help="Path to CDD root directory",
    )
    parser.add_argument(
        "--output_dir",
        default="data/_merged",
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing output directory before merging",
    )
    args = parser.parse_args()

    existing_data = Path(args.existing_data)
    cdd_root = Path(args.cdd_root)
    output_dir = Path(args.output_dir)

    # Validate input directories
    if not existing_data.exists():
        raise FileNotFoundError(f"Existing data directory not found: {existing_data}")
    if not cdd_root.exists():
        raise FileNotFoundError(f"CDD root directory not found: {cdd_root}")

    # Clean output directory if --force
    if args.force and output_dir.exists():
        print(f"[merge_datasets] Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create output directories
    flooded_dir = output_dir / "flooded"
    not_flooded_dir = output_dir / "not_flooded"

    print("[merge_datasets] Starting dataset merge...")
    print()

    # Merge flooded class
    print("Merging FLOODED class:")
    print(f"  - Copying from {existing_data / 'flooded'}...")
    count1 = copy_images(existing_data / "flooded", flooded_dir, prefix="orig")
    print(f"    Copied {count1} images")
    
    print(f"  - Copying from {cdd_root / 'Water_Disaster'}...")
    count2 = copy_images(cdd_root / "Water_Disaster", flooded_dir, prefix="cdd")
    print(f"    Copied {count2} images")
    
    total_flooded = count1 + count2
    print(f"  Total flooded images: {total_flooded}")
    print()

    # Merge not_flooded class
    print("Merging NOT_FLOODED class:")
    print(f"  - Copying from {existing_data / 'not_flooded'}...")
    count3 = copy_images(existing_data / "not_flooded", not_flooded_dir, prefix="orig")
    print(f"    Copied {count3} images")
    
    print(f"  - Copying from {cdd_root / 'Non_Damage'}...")
    count4 = copy_images(cdd_root / "Non_Damage", not_flooded_dir, prefix="cdd")
    print(f"    Copied {count4} images")
    
    total_not_flooded = count3 + count4
    print(f"  Total not_flooded images: {total_not_flooded}")
    print()

    # Summary
    print("=" * 60)
    print("[merge_datasets] Merge complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - flooded: {total_flooded} images")
    print(f"  - not_flooded: {total_not_flooded} images")
    print(f"  - Total: {total_flooded + total_not_flooded} images")
    print("=" * 60)


if __name__ == "__main__":
    main()

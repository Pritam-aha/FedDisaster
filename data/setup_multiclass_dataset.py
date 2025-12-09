#!/usr/bin/env python3
"""
Enhanced dataset setup for multi-class disaster detection.
Supports adding multiple disaster types to the existing flood dataset.

Usage:
  python setup_multiclass_dataset.py \
    --disaster_sources flood=data/_organized fire=path/to/fire landslide=path/to/landslide \
    --target_root data \
    --num_clients 3 \
    --force
"""
import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def collect_images_from_flat_dir(dir_path: Path) -> List[Path]:
    """Collect all images from a flat directory (non-ImageFolder format)."""
    if not dir_path.exists():
        return []
    return [p for p in dir_path.rglob("*") if is_image(p)]


def collect_images_from_imagefolder(root: Path) -> Dict[str, List[Path]]:
    """Collect images organized in ImageFolder format (class subdirectories)."""
    if not root.exists():
        return {}
    
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    mapping = {}
    for subdir in subdirs:
        imgs = [p for p in subdir.rglob("*") if is_image(p)]
        if imgs:
            mapping[subdir.name] = imgs
    return mapping


def auto_detect_structure(source_path: Path) -> Tuple[str, Dict[str, List[Path]]]:
    """
    Auto-detect whether source is ImageFolder format or flat directory.
    Returns: (format_type, {class_name: [image_paths]})
    """
    source_path = source_path.resolve()
    
    # Check if it looks like ImageFolder (has subdirectories with images)
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    if len(subdirs) >= 1:
        # Try ImageFolder format
        mapping = collect_images_from_imagefolder(source_path)
        if mapping:
            return "imagefolder", mapping
    
    # Try flat directory
    imgs = collect_images_from_flat_dir(source_path)
    if imgs:
        # Use parent directory name as class
        class_name = source_path.name
        return "flat", {class_name: imgs}
    
    raise RuntimeError(f"No images found in {source_path}")


def parse_disaster_sources(sources_arg: List[str]) -> Dict[str, Path]:
    """
    Parse disaster source arguments.
    Format: disaster_type=path or disaster_type:subclass=path
    Examples:
      - flood=data/_organized
      - fire=path/to/fire
      - flood:flooded=path/to/flooded (specific subclass override)
    """
    result = {}
    for item in sources_arg:
        if "=" not in item:
            raise ValueError(f"Invalid format: {item}. Expected format: type=path or type:subclass=path")
        
        key, path_str = item.split("=", 1)
        path = Path(path_str)
        
        if not path.exists():
            raise FileNotFoundError(f"Source path does not exist: {path}")
        
        result[key] = path
    
    return result


def collect_all_disaster_data(disaster_sources: Dict[str, Path]) -> Dict[str, List[Path]]:
    """
    Collect all images from multiple disaster sources.
    Returns: {class_name: [image_paths]}
    """
    all_classes = defaultdict(list)
    
    for disaster_key, source_path in disaster_sources.items():
        print(f"\n[Collecting] {disaster_key} from {source_path}")
        
        # Check if disaster_key contains subclass override (e.g., "flood:flooded")
        if ":" in disaster_key:
            disaster_type, subclass = disaster_key.split(":", 1)
            # Collect images and assign to specific subclass
            imgs = collect_images_from_flat_dir(source_path)
            if imgs:
                all_classes[subclass].extend(imgs)
                print(f"  ✓ Found {len(imgs)} images for class '{subclass}'")
        else:
            # Auto-detect structure
            fmt, mapping = auto_detect_structure(source_path)
            print(f"  Format: {fmt}")
            
            for class_name, imgs in mapping.items():
                all_classes[class_name].extend(imgs)
                print(f"  ✓ Class '{class_name}': {len(imgs)} images")
    
    return dict(all_classes)


def copy_images(paths: List[Path], dst_dir: Path):
    """Copy images to destination directory with collision handling."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for src in paths:
        # Handle filename collisions by adding counter
        dst_path = dst_dir / src.name
        counter = 1
        while dst_path.exists():
            stem = src.stem
            suffix = src.suffix
            dst_path = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(src, dst_path)


def split_and_distribute(
    class_to_paths: Dict[str, List[Path]],
    target_root: Path,
    num_clients: int,
    client_train_ratio: float,
    global_test_ratio: float,
    seed: int,
):
    """Split data across clients and global test set."""
    rng = random.Random(seed)
    
    # Prepare target directories
    global_test_root = target_root / "global_test"
    client_roots = [target_root / f"client_{i+1}" for i in range(num_clients)]
    
    for cr in client_roots:
        (cr / "train").mkdir(parents=True, exist_ok=True)
        (cr / "test").mkdir(parents=True, exist_ok=True)
    global_test_root.mkdir(parents=True, exist_ok=True)
    
    print("\n[Distribution Summary]")
    print(f"Classes: {list(class_to_paths.keys())}")
    print(f"Clients: {num_clients}")
    print(f"Global test ratio: {global_test_ratio:.1%}")
    print(f"Client train/test ratio: {client_train_ratio:.1%}/{1-client_train_ratio:.1%}")
    
    # Distribute per class
    for cls, paths in class_to_paths.items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        
        # Split global test
        n_global = int(n * global_test_ratio)
        global_paths = paths[:n_global]
        remaining = paths[n_global:]
        
        # Distribute remaining among clients
        base = len(remaining) // num_clients
        rem_count = len(remaining) % num_clients
        
        offsets: List[Tuple[int, int]] = []
        start = 0
        for i in range(num_clients):
            take = base + (1 if i < rem_count else 0)
            offsets.append((start, start + take))
            start += take
        
        # Copy to global test
        copy_images(global_paths, global_test_root / cls)
        
        # Copy to clients
        for i, (lo, hi) in enumerate(offsets):
            client_paths = remaining[lo:hi]
            if not client_paths:
                continue
            
            split_idx = int(len(client_paths) * client_train_ratio)
            train_paths = client_paths[:split_idx]
            test_paths = client_paths[split_idx:]
            
            cr = client_roots[i]
            copy_images(train_paths, cr / "train" / cls)
            copy_images(test_paths, cr / "test" / cls)
        
        print(f"\n  Class '{cls}': {n} total images")
        print(f"    - Global test: {len(global_paths)}")
        for i in range(num_clients):
            lo, hi = offsets[i]
            client_total = hi - lo
            client_train = int(client_total * client_train_ratio)
            client_test = client_total - client_train
            print(f"    - Client {i+1}: {client_train} train, {client_test} test")


def main():
    parser = argparse.ArgumentParser(
        description="Setup multi-class disaster detection dataset for federated learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with existing flood data
  python setup_multiclass_dataset.py --disaster_sources flood=data/_organized

  # Add multiple disaster types
  python setup_multiclass_dataset.py \\
    --disaster_sources flood=data/_organized fire=data/fire_dataset landslide=data/landslide_dataset

  # Override specific subclass names
  python setup_multiclass_dataset.py \\
    --disaster_sources flood:flooded=data/_organized/flooded flood:not_flooded=data/_organized/not_flooded
        """
    )
    
    parser.add_argument(
        "--disaster_sources",
        nargs="+",
        required=True,
        help="Disaster sources in format: type=path (e.g., flood=data/_organized fire=path/to/fire)"
    )
    parser.add_argument(
        "--target_root",
        default="data",
        help="Output directory for client_*/ and global_test/ (default: data)"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="Number of federated clients (default: 3)"
    )
    parser.add_argument(
        "--client_train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data per client (default: 0.8)"
    )
    parser.add_argument(
        "--global_test_ratio",
        type=float,
        default=0.1,
        help="Ratio of data for global test set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing client and global_test directories before creating new ones"
    )
    
    args = parser.parse_args()
    
    # Validation
    if not (0.0 < args.client_train_ratio < 1.0):
        raise SystemExit("--client_train_ratio must be in (0,1)")
    if not (0.0 <= args.global_test_ratio < 1.0):
        raise SystemExit("--global_test_ratio must be in [0,1)")
    if args.num_clients < 1:
        raise SystemExit("--num_clients must be >=1")
    
    target_root = Path(args.target_root)
    
    # Clean existing directories if --force
    if args.force:
        print("[Cleaning] Removing existing client and global_test directories...")
        for i in range(args.num_clients):
            shutil.rmtree(target_root / f"client_{i+1}", ignore_errors=True)
        shutil.rmtree(target_root / "global_test", ignore_errors=True)
    
    # Parse disaster sources
    print("\n" + "="*60)
    print("Multi-Class Disaster Dataset Setup")
    print("="*60)
    
    disaster_sources = parse_disaster_sources(args.disaster_sources)
    
    # Collect all data
    all_class_data = collect_all_disaster_data(disaster_sources)
    
    if not all_class_data:
        raise SystemExit("No images collected from any source!")
    
    if len(all_class_data) < 2:
        print("\nWARNING: Only 1 class detected. Multi-class classification requires >=2 classes.")
    
    # Create base directory
    target_root.mkdir(parents=True, exist_ok=True)
    
    # Split and distribute
    split_and_distribute(
        class_to_paths=all_class_data,
        target_root=target_root,
        num_clients=args.num_clients,
        client_train_ratio=args.client_train_ratio,
        global_test_ratio=args.global_test_ratio,
        seed=args.seed,
    )
    
    print("\n" + "="*60)
    print("[SUCCESS] Dataset setup complete!")
    print("="*60)
    print(f"\nCreated structure:")
    for i in range(args.num_clients):
        print(f"  ✓ {target_root / f'client_{i+1}'}")
    print(f"  ✓ {target_root / 'global_test'}")
    print(f"\nTotal classes: {len(all_class_data)}")
    print(f"Classes: {', '.join(all_class_data.keys())}")


if __name__ == "__main__":
    main()

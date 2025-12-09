#!/usr/bin/env python3
"""
Download and prepare the disaster-images-dataset from Kaggle.
Dataset: varpit94/disaster-images-dataset

Prerequisites:
1. Install kaggle: pip install kaggle
2. Setup Kaggle API credentials:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - Place kaggle.json in: C:\Users\<YourUsername>\.kaggle\
   - Or set: $env:KAGGLE_CONFIG_DIR="path/to/your/kaggle/config"

Usage:
    python download_kaggle_dataset.py
"""

import os
import sys
import zipfile
from pathlib import Path
import shutil

def check_kaggle_installed():
    """Check if kaggle package is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        return False

def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        return True
    
    # Check KAGGLE_CONFIG_DIR environment variable
    kaggle_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if kaggle_config_dir:
        kaggle_json_alt = Path(kaggle_config_dir) / "kaggle.json"
        if kaggle_json_alt.exists():
            return True
    
    return False

def download_dataset(dataset_name: str, download_path: Path):
    """Download dataset from Kaggle."""
    try:
        import kaggle
        print(f"\n[Downloading] {dataset_name} from Kaggle...")
        print(f"[Target] {download_path}")
        
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(download_path),
            unzip=True,
            quiet=False
        )
        
        print(f"✓ Download complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False

def explore_dataset_structure(base_path: Path):
    """Explore and display the downloaded dataset structure."""
    print("\n[Dataset Structure]")
    print("=" * 60)
    
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return None
    
    # Find directories with images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    structure = {}
    
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        image_files = [f for f in files if Path(f).suffix.lower() in image_exts]
        
        if image_files:
            rel_path = root_path.relative_to(base_path)
            structure[str(rel_path)] = len(image_files)
    
    # Display structure
    for path, count in sorted(structure.items()):
        indent = "  " * (path.count(os.sep))
        folder_name = Path(path).name if path != "." else base_path.name
        print(f"{indent}📁 {folder_name}: {count} images")
    
    print("=" * 60)
    return structure

def create_integration_command(download_path: Path):
    """Generate the command to integrate this dataset with existing data."""
    print("\n[Next Steps]")
    print("=" * 60)
    print("Run the following command to integrate this dataset:\n")
    
    # Auto-detect likely structure
    subdirs = [d for d in download_path.iterdir() if d.is_dir()]
    
    if subdirs:
        print("# Option 1: Use auto-detected structure")
        print(f"python data/setup_multiclass_dataset.py `")
        print(f"  --disaster_sources disaster={download_path} `")
        print(f"  --target_root data `")
        print(f"  --num_clients 3 `")
        print(f"  --force")
        print()
        
        print("# Option 2: Combine with existing flood data")
        print(f"python data/setup_multiclass_dataset.py `")
        print(f"  --disaster_sources flood=data/_organized disaster={download_path} `")
        print(f"  --target_root data `")
        print(f"  --num_clients 3 `")
        print(f"  --force")
    
    print("\n" + "=" * 60)

def main():
    print("=" * 60)
    print("Kaggle Disaster Dataset Downloader")
    print("=" * 60)
    
    # Configuration
    dataset_name = "varpit94/disaster-images-dataset"
    download_dir = Path("data/_downloads/kaggle_disaster_dataset")
    
    # Check prerequisites
    print("\n[Checking Prerequisites]")
    
    if not check_kaggle_installed():
        print("✗ Kaggle package not installed")
        print("\nInstall it with:")
        print("  pip install kaggle")
        sys.exit(1)
    print("✓ Kaggle package installed")
    
    if not check_kaggle_credentials():
        print("✗ Kaggle API credentials not found")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Place kaggle.json in: C:\\Users\\<YourUsername>\\.kaggle\\")
        print("   Or set environment variable: KAGGLE_CONFIG_DIR")
        sys.exit(1)
    print("✓ Kaggle API credentials configured")
    
    # Download dataset
    if download_dataset(dataset_name, download_dir):
        # Explore structure
        explore_dataset_structure(download_dir)
        
        # Generate integration command
        create_integration_command(download_dir)
        
        print("\n✓ All done! Dataset ready for integration.")
    else:
        print("\n✗ Failed to download dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()

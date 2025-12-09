# Guide: Adding Multiple Disaster Datasets

This guide explains how to expand your federated learning project from binary flood detection to multi-class disaster detection (flood, fire, landslide, earthquake, etc.).

## Current Setup

Your project currently has:
- **2 classes**: `flooded`, `not_flooded`
- **3 clients**: each with train/test splits
- **1 global test set**: for server-side evaluation

## Quick Start - Kaggle Dataset

### Using the varpit94/disaster-images-dataset from Kaggle

**Step 1: Install Kaggle API**
```powershell
pip install kaggle
```

**Step 2: Setup Kaggle Credentials**
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Place `kaggle.json` in `C:\Users\<YourUsername>\.kaggle\`

**Step 3: Download the Dataset**
```powershell
python download_kaggle_dataset.py
```

The script will:
- Download the disaster dataset from Kaggle
- Extract it to `data/_downloads/kaggle_disaster_dataset/`
- Show you the dataset structure
- Provide the exact command to integrate it

**Step 4: Integrate with Your Project**
After downloading, use the command provided by the script, or run:
```powershell
# Replace existing flood data with multi-class disasters
python data/setup_multiclass_dataset.py `
  --disaster_sources disaster=data/_downloads/kaggle_disaster_dataset `
  --target_root data `
  --num_clients 3 `
  --force

# OR combine with existing flood data
python data/setup_multiclass_dataset.py `
  --disaster_sources flood=data/_organized disaster=data/_downloads/kaggle_disaster_dataset `
  --target_root data `
  --num_clients 3 `
  --force
```

## Quick Start - Custom Datasets

### Option 1: Keep Existing Flood Data + Add New Disasters

```powershell
# Example: Add fire and landslide datasets
python data/setup_multiclass_dataset.py `
  --disaster_sources flood=data/_organized fire=D:\datasets\fire landslide=D:\datasets\landslide `
  --target_root data `
  --num_clients 3 `
  --force
```

### Option 2: Start Fresh with All Disaster Types

```powershell
# Rebuild everything from scratch with multiple disaster types
python data/setup_multiclass_dataset.py `
  --disaster_sources `
    flood:flooded=data/_organized/flooded `
    flood:not_flooded=data/_organized/not_flooded `
    fire:fire_damage=D:\datasets\fire `
    landslide:landslide_damage=D:\datasets\landslide `
    earthquake:earthquake_damage=D:\datasets\earthquake `
  --target_root data `
  --num_clients 3 `
  --force
```

## Dataset Format Requirements

The script auto-detects two formats:

### Format 1: ImageFolder (Recommended)
```
your_dataset/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   └── ...
└── class_3/
    └── ...
```

### Format 2: Flat Directory
```
your_dataset/
├── image1.jpg
├── image2.jpg
└── ...
```
Note: For flat directories, the folder name becomes the class name.

## Syntax Explained

### Basic Syntax
```
--disaster_sources type=path
```

### Examples

**Use existing folder structure:**
```powershell
--disaster_sources flood=data/_organized
```
This will detect `flooded/` and `not_flooded/` subdirectories automatically.

**Add multiple disaster types:**
```powershell
--disaster_sources `
  flood=data/_organized `
  fire=D:\datasets\fire_disaster `
  landslide=D:\datasets\landslide_data
```

**Override class names:**
```powershell
--disaster_sources `
  flood:flooded=data/_organized/flooded `
  fire:fire_damage=D:\fire\damaged
```
The format is `disaster_type:class_name=path`

## Complete Example Workflow

### Step 1: Organize Your Datasets

Create a folder structure like this:
```
D:\datasets\
├── flood\
│   ├── flooded\
│   └── not_flooded\
├── fire\
│   ├── fire_damage\
│   └── no_fire_damage\
├── landslide\
│   ├── landslide_damage\
│   └── no_landslide_damage\
└── earthquake\
    ├── earthquake_damage\
    └── no_earthquake_damage\
```

### Step 2: Run the Setup Script

```powershell
python data/setup_multiclass_dataset.py `
  --disaster_sources `
    flood=D:\datasets\flood `
    fire=D:\datasets\fire `
    landslide=D:\datasets\landslide `
    earthquake=D:\datasets\earthquake `
  --target_root data `
  --num_clients 3 `
  --client_train_ratio 0.8 `
  --global_test_ratio 0.1 `
  --seed 42 `
  --force
```

### Step 3: Verify the Output

After running, you should see:
```
data/
├── client_1/
│   ├── train/
│   │   ├── flooded/
│   │   ├── not_flooded/
│   │   ├── fire_damage/
│   │   ├── no_fire_damage/
│   │   ├── landslide_damage/
│   │   ├── no_landslide_damage/
│   │   ├── earthquake_damage/
│   │   └── no_earthquake_damage/
│   └── test/
│       └── [same classes]
├── client_2/
│   └── [same structure]
├── client_3/
│   └── [same structure]
└── global_test/
    └── [same classes]
```

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--disaster_sources` | Required | Space-separated list of disaster sources |
| `--target_root` | `data` | Output directory |
| `--num_clients` | `3` | Number of federated clients |
| `--client_train_ratio` | `0.8` | Train/test split per client (80/20) |
| `--global_test_ratio` | `0.1` | Portion reserved for global test |
| `--seed` | `42` | Random seed for reproducibility |
| `--force` | `false` | Overwrite existing data |

## Distribution Logic

The script follows this distribution:
1. **10%** of all data → `global_test/` (held-out test set)
2. **90%** of all data → split evenly among clients
   - Each client gets **80%** for training
   - Each client gets **20%** for local testing

Example with 1000 images:
- Global test: 100 images
- Client 1: 240 train + 60 test = 300 images
- Client 2: 240 train + 60 test = 300 images  
- Client 3: 240 train + 60 test = 300 images

## Important Notes

### Model Changes Required

After adding more classes, you **must update** `num_classes` in your model initialization:

```python
# In client.py and server.py, change:
model = SimpleCNN(num_classes=2)  # Old: binary classification

# To:
model = SimpleCNN(num_classes=8)  # New: 8 classes (example)
```

Or better yet, auto-detect the number of classes:

```python
from dataset_loader import load_imagefolder_dataloaders

train_loader, test_loader, num_classes = load_imagefolder_dataloaders(
    train_dir=f"data/client_{cid}/train",
    test_dir=f"data/client_{cid}/test",
    batch_size=32
)

model = SimpleCNN(num_classes=num_classes)  # Auto-detect
```

### File Name Collisions

If multiple datasets have files with the same name (e.g., `image_1.jpg`), the script automatically renames duplicates:
- Original: `image_1.jpg`
- Duplicate: `image_1_1.jpg`
- Duplicate: `image_1_2.jpg`

### Class Balancing

The script distributes each class independently, so:
- ✅ Different classes can have different numbers of images
- ✅ Each class maintains the same train/test/global ratios
- ✅ All clients receive balanced splits per class

## Troubleshooting

### Error: "No images found"
- Check that your source directories contain image files (jpg, png, etc.)
- Ensure paths are correct (use absolute paths on Windows)

### Error: "Expected >=2 classes"
- Make sure your dataset has subdirectories with images
- For flat directories, add more disaster types

### Classes not showing up
- Verify the directory structure matches ImageFolder format
- Check that image files have valid extensions (.jpg, .png, etc.)

## Next Steps

After setting up your multi-class dataset:

1. **Update your model** to handle the new number of classes
2. **Test locally** with one client before running federated learning
3. **Adjust training parameters** (epochs, batch size) if needed
4. **Monitor class distribution** to ensure balanced learning

## Example Commands

### Minimal Example (Keep Current Flood Data)
```powershell
python data/setup_multiclass_dataset.py --disaster_sources flood=data/_organized --force
```

### Add Fire Dataset Only
```powershell
python data/setup_multiclass_dataset.py `
  --disaster_sources flood=data/_organized fire=D:\fire_data `
  --force
```

### Full Multi-Class Setup
```powershell
python data/setup_multiclass_dataset.py `
  --disaster_sources `
    flood=data/_organized `
    fire=D:\datasets\fire `
    landslide=D:\datasets\landslide `
    earthquake=D:\datasets\earthquake `
    tornado=D:\datasets\tornado `
  --num_clients 5 `
  --force
```

## Questions?

- Check the script help: `python data/setup_multiclass_dataset.py --help`
- Review the current data structure: `Get-ChildItem data -Directory`
- Verify class counts after setup

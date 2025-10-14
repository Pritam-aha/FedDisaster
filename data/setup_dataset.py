#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def find_imagefolder_root(start: Path) -> Path:
    """Find a directory that looks like an ImageFolder root.

    Heuristic: directory containing >=2 subdirectories, each with >=1 image file
    (recursively or directly). If not found at `start`, try first child dir.
    """
    def has_class_dirs(d: Path) -> bool:
        subdirs = [x for x in d.iterdir() if x.is_dir()]
        if len(subdirs) < 2:
            return False
        classes_with_imgs = 0
        for c in subdirs:
            # search images up to depth 2
            imgs = [p for p in c.rglob("*") if is_image(p)]
            if imgs:
                classes_with_imgs += 1
        return classes_with_imgs >= 2

    start = start.resolve()
    if has_class_dirs(start):
        return start
    # Try descending one level if there's a single subdir
    subdirs = [x for x in start.iterdir() if x.is_dir()]
    if len(subdirs) == 1 and has_class_dirs(subdirs[0]):
        return subdirs[0]
    # As a fallback, if any immediate subdir qualifies, use the first
    for sd in subdirs:
        if (sd.is_dir()) and has_class_dirs(sd):
            return sd
    raise RuntimeError(
        f"Could not locate an ImageFolder-style root under: {start}. "
        "Ensure your dataset has class subfolders with images."
    )


def collect_class_images(root: Path) -> Dict[str, List[Path]]:
    classes = [d for d in root.iterdir() if d.is_dir()]
    if len(classes) < 2:
        raise RuntimeError("Expected >=2 class subfolders in source_dir.")
    mapping: Dict[str, List[Path]] = {}
    for c in classes:
        imgs = [p for p in c.rglob("*") if is_image(p)]
        if not imgs:
            continue
        mapping[c.name] = imgs
    if len(mapping) < 2:
        raise RuntimeError("Found <2 classes with images under source_dir.")
    return mapping


def ensure_clean_dir(path: Path, force: bool):
    if path.exists():
        if force:
            shutil.rmtree(path)
        else:
            return
    path.mkdir(parents=True, exist_ok=True)


def copy_many(paths: List[Path], dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        rel_name = src.name
        shutil.copy2(src, dst_dir / rel_name)


def split_and_copy(
    class_to_paths: Dict[str, List[Path]],
    target_root: Path,
    num_clients: int,
    client_train_ratio: float,
    global_test_ratio: float,
    seed: int,
):
    rng = random.Random(seed)

    # Prepare target dirs
    global_test_root = target_root / "global_test"
    client_roots = [target_root / f"client_{i+1}" for i in range(num_clients)]
    for cr in client_roots:
        (cr / "train").mkdir(parents=True, exist_ok=True)
        (cr / "test").mkdir(parents=True, exist_ok=True)
    global_test_root.mkdir(parents=True, exist_ok=True)

    # Per class distribution
    for cls, paths in class_to_paths.items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_global = int(n * global_test_ratio)
        global_paths = paths[:n_global]
        remaining = paths[n_global:]

        base = len(remaining) // num_clients
        rem = len(remaining) % num_clients
        offsets: List[Tuple[int, int]] = []
        start = 0
        for i in range(num_clients):
            take = base + (1 if i < rem else 0)
            offsets.append((start, start + take))
            start += take

        # Copy global test
        copy_many(global_paths, global_test_root / cls)

        # Copy clients train/test
        for i, (lo, hi) in enumerate(offsets):
            client_paths = remaining[lo:hi]
            if not client_paths:
                continue
            split_idx = int(len(client_paths) * client_train_ratio)
            train_paths = client_paths[:split_idx]
            test_paths = client_paths[split_idx:]

            cr = client_roots[i]
            copy_many(train_paths, cr / "train" / cls)
            copy_many(test_paths, cr / "test" / cls)


def main():
    p = argparse.ArgumentParser(description="Prepare dataset for federated clients + global test.")
    p.add_argument("--source_dir", required=True, help="Path to raw dataset root (or parent).")
    p.add_argument("--target_root", default="data", help="Where to create client_*/ and global_test/.")
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--client_train_ratio", type=float, default=0.8)
    p.add_argument("--global_test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Delete existing target dirs before writing.")
    args = p.parse_args()

    if not (0.0 < args.client_train_ratio < 1.0):
        raise SystemExit("--client_train_ratio must be in (0,1)")
    if not (0.0 <= args.global_test_ratio < 1.0):
        raise SystemExit("--global_test_ratio must be in [0,1)")
    if args.num_clients < 1:
        raise SystemExit("--num_clients must be >=1")

    source = Path(args.source_dir)
    target_root = Path(args.target_root)

    # Clean target if --force
    if args.force:
        for i in range(args.num_clients):
            shutil.rmtree(target_root / f"client_{i+1}", ignore_errors=True)
        shutil.rmtree(target_root / "global_test", ignore_errors=True)

    image_root = find_imagefolder_root(source)
    class_to_paths = collect_class_images(image_root)

    # Create base dirs
    (target_root).mkdir(parents=True, exist_ok=True)

    split_and_copy(
        class_to_paths=class_to_paths,
        target_root=target_root,
        num_clients=args.num_clients,
        client_train_ratio=args.client_train_ratio,
        global_test_ratio=args.global_test_ratio,
        seed=args.seed,
    )

    # Summary
    print("[setup_dataset] Done. Created:")
    for i in range(args.num_clients):
        cr = target_root / f"client_{i+1}"
        print(f"  - {cr}")
    print(f"  - {target_root / 'global_test'}")


if __name__ == "__main__":
    main()

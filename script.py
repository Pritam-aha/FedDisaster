from pathlib import Path
import itertools as it

root = Path("data")
def count_files(p): return sum(1 for _ in p.rglob("*") if _.is_file())

  # Global test
gt = root / "global_test"
print("Global test total:", count_files(gt))
for cls in sorted([d for d in gt.iterdir() if d.is_dir()]):
    print("  class", cls.name, ":", count_files(cls))

  # Clients
for c in sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("client_")]):
    tr = count_files(c / "train")
    te = count_files(c / "test")
    print(f"{c.name}: train={tr}, test={te}, total={tr+te}")
    for split in ["train","test"]:
        split_dir = c / split
        for cls in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            print(f"  {split:5} {cls.name:15} {count_files(cls)}")
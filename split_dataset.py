from pathlib import Path
import random
import shutil

random.seed(42)

source_root = Path("dataset")
output_root = Path("dataset_split")

classes = ["laptop_open", "laptop_closed"]

for class_name in classes:
    files = list((source_root / class_name).glob("*.jpg"))
    files += list((source_root / class_name).glob("*.jpeg"))

    random.shuffle(files)

    total = len(files)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, split_files in splits.items():
        target_dir = output_root / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in split_files:
            shutil.copy2(file, target_dir / file.name)

        print(f"{class_name} - {split_name}: {len(split_files)}")

print("Dataset split complete.")

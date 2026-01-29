import os
import random
import shutil
from pathlib import Path

# Configuration
DATA_DIR = "./"
SOURCE_DIRS = ["cible", "nocible"]
DEST_DIRS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.80, "val": 0.20, "test": 0.0}


def balance_data():
    """Balances the number of images between cible and nocible by randomly removing excess images"""
    cible_dir = os.path.join(DATA_DIR, "cible")
    nocible_dir = os.path.join(DATA_DIR, "nocible")

    # Count images in each folder
    cible_files = [
        f for f in os.listdir(cible_dir)
        if os.path.isfile(os.path.join(cible_dir, f))
    ] if os.path.exists(cible_dir) else []

    nocible_files = [
        f for f in os.listdir(nocible_dir)
        if os.path.isfile(os.path.join(nocible_dir, f))
    ] if os.path.exists(nocible_dir) else []

    cible_count = len(cible_files)
    nocible_count = len(nocible_files)

    print(f"Images before balancing: cible={cible_count}, nocible={nocible_count}")

    if cible_count == nocible_count:
        print("Data is already balanced!")
        return

    # Determine which folder has more images
    if cible_count > nocible_count:
        larger_dir = cible_dir
        larger_files = cible_files
        target_count = nocible_count
        larger_name = "cible"
    else:
        larger_dir = nocible_dir
        larger_files = nocible_files
        target_count = cible_count
        larger_name = "nocible"

    # Randomly shuffle and select files to remove
    random.shuffle(larger_files)
    files_to_remove = larger_files[target_count:]

    # Remove excess files
    for filename in files_to_remove:
        file_path = os.path.join(larger_dir, filename)
        os.remove(file_path)

    print(f"Removed {len(files_to_remove)} images from '{larger_name}'")
    print(f"Images after balancing: cible={target_count}, nocible={target_count}")


def create_directory_structure():
    """Creates the train/val/test directory structure with cible/nocible"""
    for dest in DEST_DIRS:
        for source in SOURCE_DIRS:
            os.makedirs(os.path.join(DATA_DIR, dest, source), exist_ok=True)


def split_dataset():
    """Splits images between train/val/test according to defined ratios"""
    for source_dir in SOURCE_DIRS:
        full_source_dir = os.path.join(DATA_DIR, source_dir)
        if not os.path.exists(full_source_dir):
            print(f"Warning: Folder {full_source_dir} does not exist")
            continue

        # Get all files
        files = [
            f
            for f in os.listdir(full_source_dir)
            if os.path.isfile(os.path.join(full_source_dir, f))
        ]

        # Randomly shuffle
        random.shuffle(files)

        total = len(files)
        train_end = int(total * SPLIT_RATIOS["train"])
        val_end = train_end + int(total * SPLIT_RATIOS["val"])

        # File distribution (exclusive)
        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        # Create symbolic links instead of moving files
        for split_name, file_list in splits.items():
            for filename in file_list:
                src = os.path.abspath(os.path.join(full_source_dir, filename))
                dst = os.path.join(DATA_DIR, split_name, source_dir, filename)

                # Remove existing link if necessary to avoid errors
                if os.path.lexists(dst):
                    os.remove(dst)

                os.symlink(src, dst)
        print(
            f"{full_source_dir}: {len(splits['train'])} train, "
            f"{len(splits['val'])} val, {len(splits['test'])} test"
        )


if __name__ == "__main__":
    random.seed(42)  # For reproducibility

    print("Balancing data...")
    balance_data()

    print("\nCreating directory structure...")
    create_directory_structure()

    print("Splitting dataset...")
    split_dataset()

    print("\nDone!")

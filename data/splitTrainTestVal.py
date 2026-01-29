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
    """Balances the number of images between cible and nocible by selecting a subset of files without deletion"""
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

    print(f"Total images: cible={cible_count}, nocible={nocible_count}")

    # Determine the minimum count for balancing
    target_count = min(cible_count, nocible_count)

    if cible_count == nocible_count:
        print("Data is already balanced!")
        return {"cible": cible_files, "nocible": nocible_files}

    # Randomly select files to use (without deleting)
    random.shuffle(cible_files)
    random.shuffle(nocible_files)

    selected_cible = cible_files[:target_count]
    selected_nocible = nocible_files[:target_count]

    if cible_count > nocible_count:
        print(f"Using {target_count} out of {cible_count} images from 'cible' (skipping {cible_count - target_count})")
    else:
        print(f"Using {target_count} out of {nocible_count} images from 'nocible' (skipping {nocible_count - target_count})")

    print(f"Balanced dataset: cible={len(selected_cible)}, nocible={len(selected_nocible)}")

    return {"cible": selected_cible, "nocible": selected_nocible}


def create_directory_structure():
    """Creates the train/val/test directory structure with cible/nocible"""
    for dest in DEST_DIRS:
        for source in SOURCE_DIRS:
            os.makedirs(os.path.join(DATA_DIR, dest, source), exist_ok=True)


def split_dataset(selected_files):
    """Splits images between train/val/test according to defined ratios"""
    for source_dir in SOURCE_DIRS:
        full_source_dir = os.path.join(DATA_DIR, source_dir)
        if not os.path.exists(full_source_dir):
            print(f"Warning: Folder {full_source_dir} does not exist")
            continue

        # Use only the selected files (balanced subset)
        files = selected_files.get(source_dir, [])
        if not files:
            print(f"Warning: No files selected for {source_dir}")
            continue

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
    selected_files = balance_data()

    print("\nCreating directory structure...")
    create_directory_structure()

    print("\nSplitting dataset...")
    split_dataset(selected_files)

    print("\nDone!")

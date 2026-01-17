import os
import random
import shutil
from pathlib import Path

# Configuration
DATA_DIR = "./data/robot/"
SOURCE_DIRS = ["cible", "nocible"]
DEST_DIRS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def create_directory_structure():
    """Crée la structure de répertoires train/val/test avec cible/nocible"""
    for dest in DEST_DIRS:
        for source in SOURCE_DIRS:
            os.makedirs(os.path.join(DATA_DIR, dest, source), exist_ok=True)


def split_dataset():
    """Répartit les images entre train/val/test selon les ratios définis"""
    for source_dir in SOURCE_DIRS:
        full_source_dir = os.path.join(DATA_DIR, source_dir)
        if not os.path.exists(full_source_dir):
            print(f"Attention: Le dossier {full_source_dir} n'existe pas")
            continue

        # Récupère tous les fichiers
        files = [
            f
            for f in os.listdir(full_source_dir)
            if os.path.isfile(os.path.join(full_source_dir, f))
        ]

        # Mélange aléatoirement
        random.shuffle(files)

        total = len(files)
        train_end = int(total * SPLIT_RATIOS["train"])
        val_end = train_end + int(total * SPLIT_RATIOS["val"])

        # Répartition des fichiers (exclusifs)
        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        # Création de liens symboliques au lieu de déplacement
        for split_name, file_list in splits.items():
            for filename in file_list:
                src = os.path.abspath(os.path.join(full_source_dir, filename))
                dst = os.path.join(DATA_DIR, split_name, source_dir, filename)

                # Suppression du lien existant si nécessaire pour éviter les erreurs
                if os.path.lexists(dst):
                    os.remove(dst)

                os.symlink(src, dst)
        print(
            f"{full_source_dir}: {len(splits['train'])} train, "
            f"{len(splits['val'])} val, {len(splits['test'])} test"
        )


if __name__ == "__main__":
    print("Création de la structure de répertoires...")
    create_directory_structure()

    print("Répartition des données...")
    random.seed(42)  # Pour reproductibilité
    split_dataset()

    print("Terminé!")

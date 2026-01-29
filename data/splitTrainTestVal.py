import os
import random
import shutil
from pathlib import Path

# Configuration
DATA_DIR = "./"
SOURCE_DIRS = ["cible", "nocible"]
DEST_DIRS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.80, "val": 0.20}


def balance_data():
    """Equilibre le nombre d'images entre cible et nocible en supprimant aléatoirement les images excédentaires"""
    cible_dir = os.path.join(DATA_DIR, "cible")
    nocible_dir = os.path.join(DATA_DIR, "nocible")

    # Compte les images dans chaque dossier
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

    print(f"Images avant équilibrage: cible={cible_count}, nocible={nocible_count}")

    if cible_count == nocible_count:
        print("Les données sont déjà équilibrées!")
        return

    # Détermine quel dossier a plus d'images
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

    # Mélange aléatoirement et sélectionne les fichiers à supprimer
    random.shuffle(larger_files)
    files_to_remove = larger_files[target_count:]

    # Supprime les fichiers excédentaires
    for filename in files_to_remove:
        file_path = os.path.join(larger_dir, filename)
        os.remove(file_path)

    print(f"Supprimé {len(files_to_remove)} images de '{larger_name}'")
    print(f"Images après équilibrage: cible={target_count}, nocible={target_count}")


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
    random.seed(42)  # Pour reproductibilité

    print("Équilibrage des données...")
    balance_data()

    print("\nCréation de la structure de répertoires...")
    create_directory_structure()

    print("Répartition des données...")
    split_dataset()

    print("\nTerminé!")

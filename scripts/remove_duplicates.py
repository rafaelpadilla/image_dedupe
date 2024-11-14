from pathlib import Path
from image_dedupe import HashingSimilarity, HashType
from image_dedupe.constants import ACCEPTED_IMAGE_EXTENSIONS
import fire
import shutil
import os

def process_images(
    source_folder: str,
    dest_folder: str,
    similarity_threshold: float = 0.1,
    hash_type: str = "perceptual",
    use_multiprocessing: bool = True
) -> None:
    """
    Process images to find and move duplicates to a destination folder.

    Args:
        source_folder: Path to folder containing source images
        dest_folder: Path where duplicate images will be moved
        similarity_threshold: Threshold for image similarity (0-1), default 0.1
        hash_type: Type of hashing to use ('perceptual', 'average', 'difference'), default 'perceptual'
        use_multiprocessing: Whether to use multiprocessing for faster processing, default True
    """
    # Convert string paths to Path objects
    source_path = Path(source_folder)
    dest_path = Path(dest_folder)

    # Validate source folder exists
    if not source_path.exists():
        raise ValueError(f"Source folder does not exist: {source_folder}")

    # Check if destination exists and is not empty
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"Error: Destination directory '{dest_path}' already exists and is not empty.")
        print("Please provide an empty directory to avoid overwriting existing files.")
        return

    dest_path.mkdir(parents=True, exist_ok=True)

    # Validate hash_type
    hash_type = hash_type.lower()
    if not HashType.exists(hash_type):
        raise ValueError(f"Invalid hash_type. Must be one of: {list(HashType.get_values().keys())}")

    # Initialize hasher
    hasher = HashingSimilarity(hash_type, use_multiprocessing=use_multiprocessing)

    # Get list of image files
    image_files = [
        f for f in source_path.glob("**/*")
        if f.is_file() and f.suffix.lower() in ACCEPTED_IMAGE_EXTENSIONS
    ]
    print(f"Found {len(image_files)} image files in source folder")

    # Find duplicates
    dict_duplicates = hasher.find_duplicates(
        image_files,
        distance_metric="hamming",
        similarity_threshold=similarity_threshold,
        return_dictionary=True,
    )
    print(f"Found {len(dict_duplicates)} groups of duplicates")

    all_duplicated_files = []
    for k, v in dict_duplicates.items():
        print(f"{k}: {len(v)} duplicates")
        for duplicated_file in v:
            print(f"  {duplicated_file}")
        all_duplicated_files.extend(v)

    # Move duplicate files to destination folder
    moved_count = 0
    for duplicated_file in all_duplicated_files:
        shutil.move(str(duplicated_file), str(dest_path / duplicated_file.name))
        moved_count += 1

    print(f"Moved {moved_count} duplicate files to {dest_folder}")

if __name__ == "__main__":
    fire.Fire(process_images)

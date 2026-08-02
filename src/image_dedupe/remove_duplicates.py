"""CLI to find and move duplicate images."""

from pathlib import Path
import shutil

import fire

from image_dedupe import HashingSimilarity, HashType
from image_dedupe.constants import ACCEPTED_IMAGE_EXTENSIONS


def process_images(
    source_folder: str,
    dest_folder: str,
    similarity_threshold: float = 0.1,
    hash_type: str = "perceptual",
    use_multiprocessing: bool = True,
) -> None:
    """Process images to find and move duplicates to a destination folder.

    Args:
        source_folder: Path to folder containing source images.
        dest_folder: Path where duplicate images will be moved.
        similarity_threshold: Threshold for image similarity (0-1).
        hash_type: Hashing type ('perceptual', 'average', 'difference').
        use_multiprocessing: Whether to use multiprocessing.
    """
    source_path = Path(source_folder)
    dest_path = Path(dest_folder)

    print(f"\n\n🔍 Source folder: {source_path}")

    if not source_path.exists():
        raise ValueError(f"🚨 Source folder does not exist: {source_folder}")

    total_files = len(list(dest_path.glob("**/*")))
    if dest_path.exists() and total_files > 0:
        print(
            f"🚨 Error: Destination directory '{dest_path}' already exists "
            "and is not empty."
        )
        print(
            "Please provide an empty directory to avoid overwriting "
            "existing files."
        )
        return

    hash_type = hash_type.lower()
    if not HashType.exists(hash_type):
        raise ValueError(
            "Invalid hash_type. Must be one of: "
            f"{list(HashType.get_values().keys())}"
        )

    hasher = HashingSimilarity(hash_type, use_multiprocessing=use_multiprocessing)

    image_files = [
        f
        for f in source_path.glob("**/*")
        if f.is_file() and f.suffix.lower() in ACCEPTED_IMAGE_EXTENSIONS
    ]
    print(f"❗ Found {len(image_files)} image files in source folder")

    dict_duplicates = hasher.find_duplicates(
        image_files,
        distance_metric="hamming",
        similarity_threshold=similarity_threshold,
        return_dictionary=True,
    )
    print(f"❗ Found {len(dict_duplicates)} groups of duplicates")
    if len(dict_duplicates) == 0:
        return

    dest_path.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    last_group_folder = dest_path
    for group_id, (ref_image, group_duplicates) in enumerate(
        dict_duplicates.items()
    ):
        print(
            f"👯 [{group_id}]: {ref_image.name}: "
            f"{len(group_duplicates)} duplicates"
        )
        group_folder = dest_path / f"group_{group_id}"
        group_folder.mkdir(parents=True, exist_ok=True)
        last_group_folder = group_folder
        if not (group_folder / f"reference_{ref_image.name}").exists():
            shutil.copy(
                str(ref_image),
                str(group_folder / f"reference_{ref_image.name}"),
            )
        for duplicated_file in group_duplicates:
            destination_path = (
                group_folder / f"duplicated_{duplicated_file.name}"
            )
            if not destination_path.exists() and duplicated_file.exists():
                print(f"  {duplicated_file.name}")
                shutil.move(str(duplicated_file), destination_path)
                moved_count += 1

    print(f"💾 Moved {moved_count} duplicate files to {last_group_folder}")


def main() -> None:
    """Entry point for the remove_duplicates CLI."""
    fire.Fire(process_images)


if __name__ == "__main__":
    main()

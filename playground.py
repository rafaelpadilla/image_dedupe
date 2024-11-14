from pathlib import Path
from image_dedupe.image_hashing import HashingSimilarity, HashType

def main():
    # Initialize the HashingSimilarity class with perceptual hashing
    hasher = HashingSimilarity(hash_type=HashType.PERCEPTUAL)

    # Specify the folder containing your images
    image_folder = Path("/media/rafael/partition-old-wi/scoreboards/failed-scoreboard/scoreboard_type_A/")
    image_folder = Path("/data/deleteme")

    # Get list of image files
    # image_files = list(image_folder.glob("*.*"))
    image_files = [Path("/data/deleteme/imagem_E.jpg"),
        Path("/data/deleteme/imagem_A.jpg"),
        Path("/data/deleteme/imagem_C.jpg"),
        Path("/data/deleteme/quase_imagem_B.jpg"),
        Path("/data/deleteme/igual_imagem_B.jpg"),
        Path("/data/deleteme/imagem_B.jpg"),
        Path("/data/deleteme/imagem_D.jpg")]

    print(f"Found {len(image_files)} files")

    results = hasher.remove_duplicates(image_files, distance_metric="hamming", similarity_threshold=0.05)
    results = hasher.get_min_max_distances(
        lst_files=image_files,
        distance_metric="hamming"
    )

    # Find duplicate/similar images
    # distance_threshold: 0 = identical, 1 = completely different
    # A threshold of 0.1 means images that are 90% similar will be considered duplicates
    duplicates = hasher.find_duplicates(
        lst_files=image_files,
        distance_metric="hamming",
        similarity_threshold=0.1
    )

    # Print results
    print(f"\nFound {len(duplicates)} groups of similar images:")
    for i, duplicate_group in enumerate(duplicates, 1):
        print(f"\nGroup {i}:")
        for image_path in duplicate_group:
            print(f"  - {image_path}")

    # Example of comparing two specific images
    if len(image_files) >= 2:
        distance = hasher.compute_image_hash_distance(
            image_files[0],
            image_files[1],
            hash_type=HashType.PERCEPTUAL
        )
        print(f"\nDistance between first two images: {distance:.3f}")

if __name__ == "__main__":
    main()
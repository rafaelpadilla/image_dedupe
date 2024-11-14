from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Callable
from tqdm import tqdm
import numpy as np
import imagehash
from PIL import Image
from annoy import AnnoyIndex
from .utils import load_image
from .generic import ExplicitEnum, FilePathType
from .constants import ACCEPTED_IMAGE_EXTENSIONS, NUM_PROC

class HashType(ExplicitEnum):
    AVERAGE = "average"
    PERCEPTUAL = "perceptual"
    DIFFERENCE = "difference"
    WAVELET = "wavelet"
    COLOR = "color"
    WHASH_HAAR = "whash_haar"
    WHASH_DB4 = "whash_db4"

def _func_whash_db4(img):
    return imagehash.whash(img, mode="db4")
def _func_whash_haar(img):
    return imagehash.whash(img, mode="haar")

# Dictionary mapping hash types to their corresponding functions
HASH_FUNCTIONS: Dict[HashType, Callable] = {
    HashType.AVERAGE: imagehash.average_hash,
    HashType.PERCEPTUAL: imagehash.phash,
    HashType.DIFFERENCE: imagehash.dhash,
    HashType.WAVELET: imagehash.whash,
    HashType.COLOR: imagehash.colorhash,
    HashType.WHASH_HAAR: _func_whash_haar,
    HashType.WHASH_DB4: _func_whash_db4
}

HASH_SIZES = {
    HashType.AVERAGE: 64,
    HashType.PERCEPTUAL: 64,
    HashType.DIFFERENCE: 64,
    HashType.WAVELET: 64,
    HashType.COLOR: 42,
    HashType.WHASH_HAAR: 64,
    HashType.WHASH_DB4: 196,
}

class HashingSimilarity():
    """Class to compute image similarity using hashing techniques."""

    def __init__(
        self, hash_type: Union[str, ExplicitEnum], use_multiprocessing: bool = True
    ):
        """Initialize the HashingSimilarity class with the specified hashing type.
        Args:
            hash_type (Union[str, ExplicitEnum]): The hashing type to use.
            use_multiprocessing (bool): Enable/Disable multiprocessing.
        """
        if not HashType.exists(hash_type):
            raise ValueError(
                f"Invalid hash type {hash_type}. Available options are: {list(HashType.get_values().keys())}"
            )
        self.hash_type = hash_type
        self.hash_func = HASH_FUNCTIONS[hash_type]
        self._img2hash: dict = {}
        self.num_workers = NUM_PROC if use_multiprocessing else 1


    def extract_hash_from_image(self, image: Union[Image.Image, np.ndarray]):
        """Extract hash from a given image.

            Args:
                image (Union[Image.Image, np.ndarray]): Input image as PIL Image or numpy array

            Returns:
                imagehash.ImageHash: Computed hash of the image
        """
        if not isinstance(image, Image.Image):
            image = load_image(image, output_type="pillow")
        return self.hash_func(image)

    def extract_hash_from_image_path(self, image_path: Path):
        """Extract hash from an image path and store it in internal dictionary.

        Args:
            image_path (Path): Path to the image file
        """
        img = load_image(image_path, output_type="pillow")
        img_hash = self.extract_hash_from_image(img)
        # Include image path and hash in the dictionary
        self._img2hash[image_path] = img_hash

    def extract_hashes_from_multiple_image_paths(
        self,
        image_paths: List[FilePathType],
    ) -> dict:
        """
        Extract hashes for given file list.

        Args:
            image_paths List[TypeFilePath]: absolute paths of image files. They can be local images, URLs or GCloud bucket URLs. If these are urls, they will be downloaded to a temporary folder.
        """
        # Select images
        lst_images = [
            Path(i)
            for i in image_paths
            if Path(i).suffix.lower() in ACCEPTED_IMAGE_EXTENSIONS
        ]
        if len(lst_images) != len(image_paths):
            print(
                f"🚨 Alert! Found {len(image_paths) - len(lst_images)} files that are not images. Ignoring them."
            )
        # Filter out images that do not exist
        lst_existing_images = [i for i in lst_images if Path(i).exists()]
        num_existing_images = len(lst_existing_images)

        if num_existing_images != len(lst_images):
            num_missing_images = len(lst_images) - num_existing_images
            print(f"🚨 Alert! Found {num_missing_images} images that do not exist. Ignoring them.")

        self._img2hash = Manager().dict()
        desc = "Extracting hashes from images"
        with ProcessPoolExecutor(self.num_workers) as pool:
            list(
                tqdm(
                    pool.map(self.extract_hash_from_image_path, lst_existing_images),
                    total=num_existing_images,
                    desc=desc,
                )
            )
        return self._img2hash

    def compute_annoy_index(
        self,
        lst_hashes: list,
        annoy_metric: str = "hamming",
        annoy_vector_length: int = 64,
        annoy_n_trees: int = 100,
        n_jobs: int = NUM_PROC,
    ) -> AnnoyIndex:
        """Build annoy index by using given image hash dictionary.

        Args:
            lst_hashes (List[imagehash.ImageHash]): List of image hashes
            annoy_metric (str): Metric used for the Annoy index
            annoy_vector_length (int): Feature length (e.g. phash: 64)
            annoy_n_trees (int): More trees gives higher precision when querying
            n_jobs (int): Number of parallel jobs

        Returns:
            AnnoyIndex: Built Annoy index
        """
        ann_index = AnnoyIndex(annoy_vector_length, annoy_metric)
        for ind, (image_hash) in tqdm(enumerate(lst_hashes)):
            ann_index.add_item(ind, image_hash.hash.reshape((annoy_vector_length)))

        ann_index.build(annoy_n_trees, n_jobs=n_jobs)
        return ann_index

    def find_duplicates_from_hashes(
        self,
        dict_hashes: Dict[str, Union[str, imagehash.ImageHash]],
        distance_metric: str = "hamming",
        max_n_similar: Optional[int] = None,  # noqa: F821
        similarity_threshold: float = 0.1,
    ) -> List[set]:
        """Find near duplicates in the given a dictionary containing Paths and hashes."""

        if len(dict_hashes) == 0:
            return []

        # Make a list of hashes objects (imagehash.ImageHash)
        lst_hashes = [
            imagehash.hex_to_hash(h) if isinstance(h, str) else h
            for h in dict_hashes.values()
        ]

        all_sizes = [i.hash.size for i in lst_hashes]
        ref_size = all_sizes[0]
        assert all(size == ref_size for size in all_sizes), "All hashes must be of the same size"
        assert all_sizes[0] == HASH_SIZES[self.hash_type], "Hash size must be the same as the one used to build the Annoy index"

        # Get length of the hash vector
        annoy_index = self.compute_annoy_index(
            lst_hashes=lst_hashes,
            annoy_metric=distance_metric,
            annoy_vector_length=ref_size,
        )

        # As when computing distances with annoy, they are not sorted, we need to keep track of the original order
        lst_ref_images = list(dict_hashes.keys())

        near_duplicates = []
        desc = "Finding duplicates"
        for idx in tqdm(range(0, annoy_index.get_n_items()), desc=desc):
            similar_inds, distances = annoy_index.get_nns_by_item(
                idx,
                max_n_similar,
                include_distances=True,
            )

            # distances are ordered from the min to the max
            # distances[0] is the distance of the image to itself
            # distances[1] is the distance of the image to the next most similar image
            # distances[1:] are the distances to the other images
            # similar_inds[0] is the index of the image itself
            # similar_inds[1] is the index of the next most similar image
            # similar_inds[1:] are the indices of the other images

            # Normalize distances to the range [0, 1]
            norm_distances = np.array(distances) / ref_size
            # Apply threshold to consider the most similar images with the idx-th image
            idx_closest_items = np.where(norm_distances <= similarity_threshold)
            # Optional: selected_distances ====> np.array(distances)[idx_closest_items]
            selected_files_ids = np.array(similar_inds)[idx_closest_items]
            near_dup = set(np.array(lst_ref_images)[selected_files_ids])
            near_dup.add(lst_ref_images[idx])
            # If it is similar to other images besides itself
            if len(near_dup) > 1 and near_dup not in near_duplicates:
                near_duplicates.append(near_dup)

        return near_duplicates

    def find_duplicates(
        self,
        lst_files: List[FilePathType],
        distance_metric: str = "hamming",
        max_n_similar: Optional[int] = None,  # noqa: F821
        similarity_threshold: int = 0.1,
        return_dictionary: bool = False,
    ) -> List[set]:
        """
        Find near duplicates in the given list of images.

        Args:
            lst_files (List[TypeFilePath]): List of image paths.
            distance_metric (str): Distance metric to use.
            max_n_similar (Optional[int]): Maximum number of similar images to find.
            similarity_threshold (int): Similarity threshold to consider two images as similar. (0 = identical, 1 = completely different)
            return_dictionary (bool): Return a dictionary with the duplicates.
        Returns:
            List[set]: List of sets with the near duplicates.
        """
        # Validations
        assert distance_metric in ["hamming", "euclidean"], "Distance metric must be either 'hamming' or 'euclidean'"
        # assert 0 <= similarity_threshold <= 1, "Similarity threshold must be between 0 and 1" # AQUI
        # Set max_n_similar to the number of images if not provided
        max_n_similar = max_n_similar or len(lst_files)
        # Extract hashes from image paths
        hashes = self.extract_hashes_from_multiple_image_paths(lst_files)
        # Transform DictProxy from multiprocessing to dict
        dict_hashes = dict(hashes)
        duplicates = self.find_duplicates_from_hashes(
            dict_hashes, distance_metric, max_n_similar, similarity_threshold)
        if not return_dictionary:
            return duplicates
        duplicates_dict = {}
        for group_duplicate in duplicates:
            lst_duplicates = sorted(group_duplicate, key=lambda p : p.name)
            duplicates_dict[lst_duplicates[0]] = lst_duplicates[1:]
        return duplicates_dict


    def get_all_distances_from_hashes(
        self,
        dict_hashes: Dict[str, Union[str, imagehash.ImageHash]],
        distance_metric: str = "hamming",
    ) -> List[set]:
        if len(dict_hashes) == 0:
            return []

        # Make a list of hashes objects (imagehash.ImageHash)
        lst_hashes = [
            imagehash.hex_to_hash(h) if isinstance(h, str) else h
            for h in dict_hashes.values()
        ]

        all_sizes = [i.hash.size for i in lst_hashes]
        ref_size = all_sizes[0]
        assert all(size == ref_size for size in all_sizes), "All hashes must be of the same size"
        assert all_sizes[0] == HASH_SIZES[self.hash_type], "Hash size must be the same as the one used to build the Annoy index"

        # Get length of the hash vector
        annoy_index = self.compute_annoy_index(
            lst_hashes=lst_hashes,
            annoy_metric=distance_metric,
            annoy_vector_length=ref_size,
        )

        # As when computing distances with annoy, they are not sorted, we need to keep track of the original order
        lst_ref_images = list(dict_hashes.keys())

        all_distances = []
        min_distance = float('inf')
        max_distance = float('-inf')
        pair_with_min_distance = None
        pair_with_max_distance = None
        desc = "Finding duplicates"
        for idx in tqdm(range(0, annoy_index.get_n_items()), desc=desc):
            similar_inds, distances = annoy_index.get_nns_by_item(
                idx,
                len(dict_hashes),
                include_distances=True,
            )

            # distances are ordered from the min to the max
            # distances[0] is the distance of the image to itself
            # distances[1] is the distance of the image to the next most similar image
            # distances[1:] are the distances to the other images
            # similar_inds[0] is the index of the image itself
            # similar_inds[1] is the index of the next most similar image
            # similar_inds[1:] are the indices of the other images

            # Normalize distances to the range [0, 1]
            norm_distances = np.array(distances) / ref_size
            norm_distances = norm_distances.tolist()
            all_distances.extend(norm_distances[1:])

            idx_min_distance = similar_inds[1]
            idx_max_distance = similar_inds[-1]
            current_min = norm_distances[1]
            current_max = norm_distances[-1]

            if current_min < min_distance:
                min_distance = current_min
                pair_with_min_distance = (lst_ref_images[idx], lst_ref_images[idx_min_distance])
            if current_max > max_distance:
                max_distance = current_max
                pair_with_max_distance = (lst_ref_images[idx], lst_ref_images[idx_max_distance])

        lst_all_distances = list(set(all_distances))
        return {"all_distances": lst_all_distances, "min_distance": min_distance, "max_distance": max_distance, "pair_with_min_distance": pair_with_min_distance, "pair_with_max_distance": pair_with_max_distance}

    def get_min_max_distances(
        self,
        lst_files: List[FilePathType],
        distance_metric: str = "hamming",
    ) -> dict:
        """Compute min and max distances between images in the given list."""
        # Validations
        assert distance_metric in ["hamming", "euclidean"], "Distance metric must be either 'hamming' or 'euclidean'"
        # Extract hashes from image paths
        hashes = self.extract_hashes_from_multiple_image_paths(lst_files)
        # Transform DictProxy from multiprocessing to dict
        dict_hashes = dict(hashes)
        return  self.get_all_distances_from_hashes(dict_hashes, distance_metric)

    def compute_image_hash_distance(
            self,
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray],
        hash_type: HashType = HashType.PERCEPTUAL
    ) -> Tuple[float, Tuple[str, str]]:
        """Compute similarity between two images using various hashing methods.

        Args:
            image1 (Union[str, Path, np.ndarray]): First image (path, string, or numpy array)
            image2 (Union[str, Path, np.ndarray]): Second image (path, string, or numpy array)
            hash_type (HashType): Type of hashing algorithm to use (default: PERCEPTUAL)

        Returns:
            float: Distance between images (0 = identical, 1 = completely different)
        """
        # Load images
        img1 = load_image(image1)
        img2 = load_image(image2)

        # Get hash function and compute hashes
        hash_function = HASH_FUNCTIONS[hash_type]
        hash1 = hash_function(img1)
        hash2 = hash_function(img2)

        # Calculate Hamming distance and normalize
        return np.count_nonzero(hash1 != hash2) / hash1.size

    def remove_duplicates(
        self,
        lst_files: List[FilePathType],
        distance_metric: str = "hamming",
        max_n_similar: Optional[int] = None,
        similarity_threshold: int = 0,
    ) -> List[FilePathType]:
        """
        Remove duplicates from the given list of images.

        Args:
            lst_files (List[TypeFilePath]): List of image paths.
            distance_metric (str): Distance metric to use.
            max_n_similar (Optional[int]): Maximum number of similar images to find.
            similarity_threshold (int): Similarity threshold to consider two images as similar. Default is 0 (identical).
        Returns:
            List[set]: List of images that are NOT considered duplicates.
        """

        duplicates = self.find_duplicates(
            lst_files,
            distance_metric=distance_metric,
            max_n_similar=max_n_similar,
            similarity_threshold=similarity_threshold,
        )

        copy_lst_images = lst_files.copy()

        # Remove duplicates
        for duplicate in duplicates:
            # Keep the first image and remove the rest
            to_remove = [d for d in list(duplicate)[1:]]
            for d in to_remove:
                if d in copy_lst_images:
                    copy_lst_images.remove(d)
        return copy_lst_images


    def is_similar(
        self, first: np.ndarray, second: np.ndarray, threshold: float = 0.0
    ) -> bool:
        hash1 = self.extract_hash_from_image(first)
        hash2 = self.extract_hash_from_image(second)
        distance = hash1 - hash2
        return distance <= threshold

    def compute_hamming_distance(
        self, first: np.ndarray, second: np.ndarray, normalize: bool = False
    ) -> float:
        img1 = load_image(first, output_type="pillow")
        img2 = load_image(second, output_type="pillow")

        hash1 = self.hash_func(img1)
        hash2 = self.hash_func(img2)
        distance = hash1 - hash2
        if normalize:
            return distance / hash1.hash.size
        return distance

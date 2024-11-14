# Hashing Similarity Tool 🖼️🔍

The `Hashing Similarity` tool is designed to compute image similarity using various hashing techniques. This file will guide you on how to use this class effectively with plenty of examples.

Image hashing is not suited to find class-similar items (e.g. comparing `dogs` vs `cats`, or distinguish a `husky` dog to another `husky dog`, or find an item of the same class among many images). Nevertheless, image hashing is an excellent tool to eliminate duplicated images from a dataset.

## Features ✨
- Supports multiple hashing techniques: `Average`, `Perceptual`, `Difference`, `Wavelet`, and `Color`.
- Easy-to-use interface for image similarity comparison.
- Customizable similarity threshold.

## Hashing Types 📝
- **AVERAGE**: Uses average hashing technique.
- **PERCEPTUAL**: Uses perceptual hashing technique.
- **DIFFERENCE**: Uses difference hashing technique.
- **WAVELET**: Uses wavelet hashing technique.
- **COLOR**: Uses color hashing technique.

## Usage 🚀

Image hashing is implemente via the `HashingSimilarity`. To initialize the HashingSimilarity class, you need to specify the hashing type.

```python
from HashingSimilarity import HashingSimilarity, HashType

# Initialize with Average Hashing
similarity_checker = HashingSimilarity(HashType.AVERAGE)
```

### Verify Similarity
To check if two images are similar, use the `is_similar` method. This method returns a boolean indicating whether the images are similar based on the specified threshold `[0, 1]`. If you do not specify the threshold, the default value is `threshold=0`, which means both images must be identical or almost identical.

Use the following example to compute the **similarity between two images**:

```python
# Check similarity between two images
are_similar = similarity_checker.is_similar('reference_image.jpg', 'image.jpg', threshold=0.5)
print(f"Are images similar? {are_similar}")
```

### Measure distance
If checking if two images are similar is not enough for you, and you want to retrive the distance between images, use the `compute_distances` method. This method returns a float value indicating the distance between your samples.

```python
# Check the distance between images
distance = similarity_checker.compute_distances('reference_image.jpg', 'image.jpg')
print(f"Distance between images: {distance}")
```

The distance parameter is not normalized. As hashing distance is computed with Hamming Distance, and the hashing distance represents the total number of different blocks in the hashing. If you want the distance to be normalized within the interval `[0, 1]`, you need to specify `normalize=True`:

```python
# Check the distance between images
distance = similarity_checker.compute_hamming_distance('reference_image.jpg', 'image.jpg', normalize=True)
print(f"Distance between images: {distance}")
```

### Finding duplicated images among a list of images

You can use the `remove_duplicates` method to find duplicated images among a list of images. This method returns a list of images that are NOT considered duplicates. In other words, this method removes the duplicates from the list.

```python
all_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = similarity_checker.remove_duplicates(all_images, distance_metric="hamming" similarity_threshold=0.05)
```

Alternatively, given a list of images, you can obtain the set of images that are considered duplicates by using the `find_duplicates` method.

```python
all_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
duplicates = similarity_checker.find_duplicates(all_images, distance_metric="hamming", similarity_threshold=0.05)
```

### Compute min and max distances

Consider the case where you have a list of images and you want to compute the min and max distances between all images. You can use the `get_min_max_distances` method to obtain the min and max distances.

```python
results = similarity_checker.get_min_max_distances(all_images, distance_metric="hamming", similarity_threshold=0.1)
```

## Scripts and tools 🛠️

### `remove_duplicates.py`

This script helps you identify and move duplicate images from a source folder to a destination folder using image hashing techniques.

#### Usage
```bash
python remove_duplicates.py --source_folder="path/to/images" --dest_folder="path/to/duplicates" [OPTIONS]
```

#### Arguments
- `source_folder`: Path to the folder containing your source images
- `dest_folder`: Path where duplicate images will be moved
- `similarity_threshold`: Threshold for image similarity (0-1), default is 0.1
- `hash_type`: Type of hashing to use ('perceptual', 'average', 'difference'), default is 'perceptual'
- `use_multiprocessing`: Whether to use multiprocessing for faster processing, default is True

#### Example
```bash
python remove_duplicates.py --source_folder="dataset/images" --dest_folder="dataset/duplicates" --similarity_threshold=0.15 --hash_type="average"
```

This will scan all images in the source folder, identify duplicates using the specified hashing method, and move them to the destination folder while preserving the original filenames.


## Additional Resources 📚
[ImageHash Documentation](https://github.com/JohannesBuchner/imagehash)

from typing import Union, Literal, TypeVar, overload
from pathlib import Path
import numpy as np
from PIL import Image
from .generic import ImageType

def load_pil_image(img: Union[str, Path, ImageType]) -> Image.Image:
    """
    Load an image as a PIL Image object.

    Args:
        img: Input image as path string, Path object, numpy array, or PIL Image

    Returns:
        PIL Image object

    Raises:
        ValueError: If input type is not supported
    """
    if isinstance(img, (str, Path)):
        img = Image.open(str(img))
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        raise ValueError("Image must be a path string, Path object, or image type")
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def load_np_image(img: Union[str, Path, ImageType]) -> np.ndarray:
    """
    Load an image as a numpy array.

    Args:
        img: Input image as path string, Path object, numpy array, or PIL Image

    Returns:
        numpy array

    Raises:
        ValueError: If input type is not supported
    """
    if isinstance(img, (str, Path)):
        return np.array(Image.open(str(img)))
    elif isinstance(img, Image.Image):
        return np.array(img)
    elif isinstance(img, np.ndarray):
        return img
    raise ValueError("Image must be a path string, Path object, or image type")

@overload
def load_image(img: Union[str, Path, ImageType], output_type: Literal['numpy']) -> np.ndarray: ...

@overload
def load_image(img: Union[str, Path, ImageType], output_type: Literal['pillow']) -> Image.Image: ...

def load_image(
    img: Union[str, Path, ImageType],
    output_type: Literal['numpy', 'pillow'] = 'numpy'
) -> Union[np.ndarray, Image.Image]:
    """
    Generic image loader that can output either numpy array or PIL Image.

    Args:
        img: Input image as path string, Path object, numpy array, or PIL Image
        output_type: Desired output type ('numpy' or 'pillow'), defaults to 'numpy'

    Returns:
        Image in the requested format (numpy array or PIL Image)

    Raises:
        ValueError: If input type is not supported or output_type is invalid
    """
    if output_type == 'numpy':
        return load_np_image(img)
    elif output_type == 'pillow':
        return load_pil_image(img)
    else:
        raise ValueError("output_type must be either 'numpy' or 'pillow'")
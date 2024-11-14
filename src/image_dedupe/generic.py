from enum import Enum
from pathlib import Path
from typing import Sequence, Union, TypeVar
import numpy as np
from PIL.Image import Image

ImageType = TypeVar("ImageType", np.ndarray, Image)
MultipleImagesType = Union[Sequence[np.ndarray], Sequence[Image]]  # Fixed definition
TextFileType = TypeVar("TextFileType", str, Path)
FilePathType = TypeVar("FilePathType", str, Path)
DirectoryType = TypeVar("DirectoryType", str, Path)


class ExplicitEnum(str, Enum):
    """
    Enumerator with explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

    @classmethod
    def exists(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def get_values(cls):
        return cls._value2member_map_

    @classmethod
    def get_enums(cls):
        return list(cls._member_map_.values())

import math
import pickle
from dataclasses import field, dataclass
from functools import partial
from pathlib import Path
from random import random
from typing import Union, List, Optional, Dict, Sequence, Any, Tuple
import xml.etree.ElementTree as ET
import numpy as np
import torch
import albumentations as A
import cv2 as cv

def pickle_load(file) -> Any:
    with open(file, "rb") as f:
        return pickle.load(f)

def pickle_save(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)

def read_xml(file: Union[Path, str]) -> ET.Element:
    tree = ET.parse(file)
    root = tree.getroot()

    return root

def find_child_by_tag(element: ET.Element, tag: str, value: str) -> Union[ET.Element, None]:
    for child in element:
        if child.get(tag) == value:
            return child
    return None

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class LabelEncoder:
    classes: Optional[List[str]]
    idx_to_cls: Optional[Dict[int, str]]
    cls_to_idx: Optional[Dict[str, int]]
    n_classes: Optional[int]

    def __init__(self):
        self.classes = None
        self.idx_to_cls = None
        self.cls_to_idx = None
        self.n_classes = None

    def transform(self, classes: Sequence[str]) -> List[int]:
        self.check_is_fitted()
        return [self.cls_to_idx[c] for c in classes]

    def inverse_transforms(self, indices: Sequence[int]) -> List[str]:
        self.check_is_fitted()
        return [self.idx_to_cls[i] for i in indices]

    def fit(self, classes: Sequence[str]):
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.idx_to_cls = dict(enumerate(classes))
        self.cls_to_idx = {cls: i for i, cls in self.idx_to_cls.items()}

        return self

    def add_classes(self, classes: List[str]):
        new_classes = self.classes + classes
        assert len(set(new_classes)) == len(new_classes), "Duplicates found in the new classes"

        return self.fit(new_classes)

    def read_encoding(self, filename: Union[str, Path]):
        if Path(filename).suffix == ".pkl":
            return self.read_sklearn_encoding(filename)
        else:
            classes = []
            saved_str = Path(filename).read_text()
            i = 0
            while i < len(saved_str):
                c = saved_str[i]
                i += 1
                while i < len(saved_str) and saved_str[i] != "n":
                    c += saved_str[i]
                    i += 1
                classes.append(c)
                i += 1
            return self.fit(classes)

    def read_sklearn_encoding(self, filename: Union[Path, str]):
        label_encoder = pickle_load(filename)
        classes = list(label_encoder.classes_)

        assert (
            list(label_encoder.inverse_transform(list(range(len(classes))))) == classes
        )
        self.fit(classes)
        self.dump(Path(filename).parent)
        return self

    def dump(self, outdir: Union[str, Path]):
        out = "\n".join(cls for cls in self.classes)
        (Path(outdir) / "label_encoding.txt").write_text(out)

    def check_is_fitted(self):
        if self.idx_to_cls is None or self.cls_to_idx is None:
            raise ValueError("Not fitted label encoder")


def dpi_adjusting(img: np.ndarray, scale: float, **kwargs) -> np.ndarray:
    height, width = img.shape[:2]
    new_height, new_width = math.ceil(height * scale), math.ceil(width * scale)
    return cv.resize(img, (new_width, new_height))

def randomly_displace_and_pad(
    img: np.ndarray,
    padded_size: Tuple[int, int],
    crop_if_necessary: bool = False,
    **kwargs,
) -> np.ndarray:
    
    frame_h, frame_w = padded_size
    img_h, img_w = img.shape
    if frame_h < img_h or frame_w < img_w:
        if crop_if_necessary:
            print(
                "WARNING (`randomly_displace_and_pad`): cropping input image before "
                "padding because it exceeds the size of the frame."
            )
            img_h, img_w = min(img_h, frame_h), min(img_w, frame_w)
            img = img[:img_h, :img_w]
        else:
            raise AssertionError(
                f"Frame is smaller than the image: ({frame_h}, {frame_w}) vs. ({img_h},"
                f" {img_w})"
            )

    res = np.zeros((frame_h, frame_w), dtype=img.dtype)

    pad_top = random.randrange(0, frame_h - img_h)
    pad_bottom = pad_top + img_h
    pad_left = random.randrange(0, frame_w - img_w)
    pad_right = pad_left + img_w

    res[pad_top:pad_bottom, pad_left:pad_right] = img
    return res
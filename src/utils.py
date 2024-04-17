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
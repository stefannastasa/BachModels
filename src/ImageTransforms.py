import math
from dataclasses import dataclass, field
from functools import partial
from random import randint

import albumentations as A
from typing import Tuple
import cv2 as cv
import numpy as np

class SafeRandomScale(A.RandomScale):
    def apply(self, img, scale=0, interpolation=cv.INTER_LINEAR, **params):
        height, width = img.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        if new_height <= 0 or new_width <= 0:
            return img
        return super().apply(img, scale, interpolation, **params)

def adjust_dpi(img: np.ndarray, scale: float, **kwargs):
    height, width = img.shape
    new_height, new_width = math.ceil(height * scale), math.ceil(width * scale)
    return cv.resize(img, (new_width, new_height))

def randomly_displace_and_pad(
    img: np.ndarray,
    padded_size: Tuple[int, int],
    crop_if_necessary: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Randomly displace an image within a frame, and pad zeros around the image.

    Args:
        img (np.ndarray): image to process
        padded_size (Tuple[int, int]): (height, width) tuple indicating the size of the frame
        crop_if_necessary (bool): whether to crop the image if its size exceeds that
            of the frame
    """
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

    pad_top =  randint(0, frame_h - img_h)
    pad_bottom = pad_top + img_h
    pad_left = randint(0, frame_w - img_w)
    pad_right = pad_left + img_w

    res[pad_top:pad_bottom, pad_left:pad_right] = img
    return res

@dataclass
class ImageTransforms:
    max_img_size: Tuple[int, int]  # (h, w)
    normalize_params: Tuple[float, float]  # (mean, std)
    scale: float = (
        0.5
    )
    random_scale_limit: float = 0.1
    random_rotate_limit: int = 10

    train_trnsf: A.Compose = field(init=False)
    test_trnsf: A.Compose = field(init=False)

    def __post_init__(self):
        scale, random_scale_limit, random_rotate_limit, normalize_params =(
            self.scale,
            self.random_scale_limit,
            self.random_rotate_limit,
            self.normalize_params
        )

        max_img_h, max_img_w = self.max_img_size
        max_scale = scale + scale * random_scale_limit
        padded_h, padded_w = math.ceil(max_scale * max_img_h), math.ceil(max_scale * max_img_w)

        self.train_trnsf = A.Compose([
            A.Lambda(partial(adjust_dpi, scale=scale)),
            SafeRandomScale(scale_limit=random_scale_limit, p=0.5),
            A.SafeRotate(
                limit = random_rotate_limit,
                border_mode = cv.BORDER_CONSTANT,
                value = 0
            ),
            A.RandomBrightnessContrast(),
            A.Perspective(scale=(0.01, 0.05)),
            A.GaussNoise(),
            A.Normalize(*normalize_params),
            A.Lambda(
                image=partial(
                    randomly_displace_and_pad,
                    padded_size=(padded_h, padded_w),
                    crop_if_necessary=False,
                )
            )
        ])

        self.test_trnsf = A.Compose([
            A.Lambda(partial(adjust_dpi, scale=scale)),
            A.Normalize(*normalize_params),
            A.PadIfNeeded(
                max_img_h, max_img_w, border_mode=cv.BORDER_CONSTANT, value=0
            )
        ])

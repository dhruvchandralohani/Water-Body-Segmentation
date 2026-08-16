"""
transforms.py

Albumentations pipelines for PatchDataset (train) and TileDataset (val/test).
Both operate on an already-cropped, already-resized (patch_size x patch_size)
image/mask pair -- geometric augmentation happens here at that fixed size,
not on the original full-resolution source image.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _binarize_mask(mask, **kwargs):
    """Convert a uint8 mask to a clean binary float mask.

    Args:
        mask: Input mask array with values in the 0-255 range.
        **kwargs: Additional keyword arguments accepted for compatibility with
            Albumentations transform callbacks.

    Returns:
        A float32 mask with values of 0.0 and 1.0.
    """
    return (mask > 127).astype("float32")


def get_train_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Create the training augmentation pipeline for image and mask pairs.

    Args:
        mean: Per-channel mean used for normalization.
        std: Per-channel standard deviation used for normalization.

    Returns:
        An Albumentations Compose pipeline for training-time augmentation.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.05, 0.12),
                hole_width_range=(0.05, 0.12),
                fill="random_uniform",
                p=0.2,
            ),
            A.Lambda(mask=_binarize_mask, p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Create the evaluation transform pipeline for image and mask pairs.

    Args:
        mean: Per-channel mean used for normalization.
        std: Per-channel standard deviation used for normalization.

    Returns:
        An Albumentations Compose pipeline for evaluation-time preprocessing.
    """
    return A.Compose(
        [
            A.Lambda(mask=_binarize_mask, p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

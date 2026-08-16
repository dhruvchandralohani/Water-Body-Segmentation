"""
data.py

Wraps PatchDataset (train) and TileDataset (val/test) into PyTorch
DataLoaders.
"""

from torch.utils.data import DataLoader

from data_pipeline.patch_dataset import PatchDataset
from data_pipeline.tile_dataset import TileDataset
from data_pipeline.transforms import get_eval_transform, get_train_transform


def get_train_loader(
    manifest,
    image_dir,
    mask_dir,
    batch_size=16,
    num_workers=4,
    window_range=(64, 2048),
    patch_size=256,
    fg_bias_ratio=0.7,
    num_patches_per_epoch=8000,
    seed=None,
):
    """Create a DataLoader for training patches from the dataset manifest.

    Args:
        manifest: DataFrame or manifest-like object describing the dataset.
        image_dir: Directory containing source images.
        mask_dir: Directory containing segmentation masks.
        batch_size: Number of samples per training batch.
        num_workers: Number of worker processes for data loading.
        window_range: Range of window sizes used by the patch sampler.
        patch_size: Size of each sampled training patch.
        fg_bias_ratio: Bias toward foreground patches during sampling.
        num_patches_per_epoch: Number of patches generated per epoch.
        seed: Optional random seed for reproducible sampling.

    Returns:
        A PyTorch DataLoader configured for training patches.
    """
    dataset = PatchDataset(
        manifest=manifest,
        image_dir=image_dir,
        mask_dir=mask_dir,
        window_range=window_range,
        patch_size=patch_size,
        fg_bias_ratio=fg_bias_ratio,
        num_patches_per_epoch=num_patches_per_epoch,
        transform=get_train_transform(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_eval_loader(
    manifest,
    image_dir,
    mask_dir,
    batch_size=16,
    num_workers=4,
    tile_size=256,
    overlap=0.25,
):
    """Create a DataLoader for evaluation tiles from the dataset manifest.

    Args:
        manifest: DataFrame or manifest-like object describing the dataset.
        image_dir: Directory containing source images.
        mask_dir: Directory containing segmentation masks.
        batch_size: Number of tiles per evaluation batch.
        num_workers: Number of worker processes for data loading.
        tile_size: Size of each evaluation tile.
        overlap: Fractional overlap between adjacent tiles.

    Returns:
        A PyTorch DataLoader configured for evaluation tiles.
    """

    dataset = TileDataset(
        manifest=manifest,
        image_dir=image_dir,
        mask_dir=mask_dir,
        tile_size=tile_size,
        overlap=overlap,
        transform=get_eval_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

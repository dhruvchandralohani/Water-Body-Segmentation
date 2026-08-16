"""
train.py

Training loop tying together data.py (loaders), model.py (DeepLabV3+),
loss.py (BCE+Dice), metrics.py (per-epoch validation), and tracking.py
(MLflow logging + model registry).

run_training(args, report_callback=None) holds the actual training loop
and returns the best validation IoU; main() is a thin CLI wrapper around
it. tune.py calls run_training() directly, once per trial, with a
callback that reports progress to Optuna and can prune the trial early.

Usage:
    python train.py --train-manifest splits/train.csv --val-manifest splits/val.csv \
        --image-dir images/ --mask-dir masks_corrected/ --checkpoint-dir checkpoints/
"""

import argparse
import time
from pathlib import Path

import torch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from data_pipeline.data import get_eval_loader, get_train_loader
from common.logging_setup import log_run_separator, setup_logger
from training.loss import BCEDiceLoss
from training.metrics import SegmentationMetrics
from training.model import build_model, freeze_encoder
from training.tracking import ExperimentTracker


def train_one_epoch(model, loader, criterion, optimizer, device, frozen_modules=()):
    """Train the model for one epoch over the provided training loader.

    Args:
        model: Segmentation model to train.
        loader: DataLoader yielding training image and mask batches.
        criterion: Loss function used to optimize the model.
        optimizer: Optimizer used for parameter updates.
        device: Target device for the training tensors.
        frozen_modules: Modules to hold in eval() mode. model.train() walks the
            whole tree and would otherwise put frozen BatchNorm layers back into
            training mode, letting their running statistics drift every epoch.

    Returns:
        The mean training loss across the epoch.
    """
    model.train()
    for module in frozen_modules:
        module.eval()
    losses = []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)


@torch.no_grad()
def validate(model, loader, device, criterion=None, threshold=0.5):
    """Evaluate the model on validation data and compute summary metrics.

    Args:
        model: Segmentation model to evaluate.
        loader: DataLoader yielding validation image, mask, and metadata batches.
        device: Target device for the validation tensors.
        criterion: Optional loss function used to compute validation loss.
        threshold: Probability threshold used for confusion-matrix metrics.

    Returns:
        A dictionary with validation metrics and optional validation loss.
    """
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)
    val_losses = []
    for images, masks, meta in loader:
        images = images.to(device)
        logits = model(images)
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        if criterion is not None:
            val_losses.append(criterion(logits, masks.to(device)).item())

        preds = torch.sigmoid(logits).cpu()
        metrics.update(preds, masks, meta["filename"])

    summary = metrics.summary()
    if criterion is not None:
        summary["val_loss"] = sum(val_losses) / len(val_losses)
    return summary


def build_argparser():
    """Create the CLI argument parser for the training script."""
    parser = argparse.ArgumentParser(description="Train the water body segmentation model.")

    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints")

    # Data pipeline
    parser.add_argument("--window-min", type=int, default=64)
    parser.add_argument("--window-max", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--fg-bias-ratio", type=float, default=0.7)
    parser.add_argument("--patches-per-epoch", type=int, default=8000)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument(
        "--arch",
        default="deeplabv3plus",
        help="Architecture family (deeplabv3plus, unet, fpn, ...). Decoder args "
        "the chosen arch does not accept are dropped with a warning.",
    )
    parser.add_argument("--encoder-name", default="mobilenet_v2")
    parser.add_argument(
        "--decoder-channels",
        type=int,
        default=None,
        help="Decoder width. Leave unset for smp's default. This is the "
        "capacity lever; --dropout is the regularization lever.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the pretrained encoder (params AND BatchNorm statistics). "
        "Reduces trainable capacity instead of regularizing it.",
    )
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument(
        "--decoder-atrous-rates", type=int, nargs=3, default=(2, 4, 6),
        help="Recalibrate if --patch-size changes.",
    )

    # Loss
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-alpha", type=float, default=0.5)
    parser.add_argument("--dice-beta", type=float, default=0.5)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5, help="ASPP dropout.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience.")

    # Tracking
    parser.add_argument("--experiment-name", default="water_body_segmentation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking backend. Resolved relative to the current working "
        "directory, so pass an explicit path (e.g. sqlite:///training/mlflow.db) "
        "when running from the repo root or under DVC.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu.")

    return parser


def run_training(args, report_callback=None):
    """Run the full training loop and return the best validation IoU.

    Args:
        args: Parsed CLI arguments controlling the training run.
        report_callback: Optional callback invoked after each epoch.

    Returns:
        The best validation IoU observed during training.
    """
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size > args.patches_per_epoch:
        raise ValueError(
            f"batch_size ({args.batch_size}) cannot exceed patches_per_epoch "
            f"({args.patches_per_epoch}) -- the train loader drops incomplete "
            f"batches, so this combination would silently produce zero batches "
            f"per epoch. Lower batch_size or raise patches_per_epoch."
        )

    logger = setup_logger("train", log_file="training.log")
    log_run_separator(logger, f"Training run: experiment={args.experiment_name} run={args.run_name}")
    logger.info(
        f"Config: arch={args.arch} encoder={args.encoder_name} weights={args.encoder_weights} "
        f"lr={args.lr} weight_decay={args.weight_decay} dropout={args.dropout} "
        f"batch_size={args.batch_size} patches_per_epoch={args.patches_per_epoch} "
        f"epochs={args.epochs} bce_weight={args.bce_weight} "
        f"dice_alpha={args.dice_alpha} dice_beta={args.dice_beta}"
    )

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights

    train_loader = get_train_loader(
        manifest=args.train_manifest,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_range=(args.window_min, args.window_max),
        patch_size=args.patch_size,
        fg_bias_ratio=args.fg_bias_ratio,
        num_patches_per_epoch=args.patches_per_epoch,
        seed=args.seed,
    )
    val_loader = get_eval_loader(
        manifest=args.val_manifest,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    model = build_model(
        arch=args.arch,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        decoder_atrous_rates=tuple(args.decoder_atrous_rates),
        decoder_aspp_dropout=args.dropout,
        decoder_channels=args.decoder_channels,
    ).to(device)

    frozen_modules = freeze_encoder(model) if args.freeze_encoder else ()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    val_size = len(val_loader.dataset)  # type: ignore[arg-type]
    logger.info(
        f"Data: {args.patches_per_epoch} patches/epoch (train), "
        f"{val_size} tiles (val), device={device}"
    )
    logger.info(
        f"Model: {model.__class__.__name__} ({args.encoder_name}), {total_params:,} parameters"
        + (f", {trainable_params:,} trainable (encoder frozen)" if args.freeze_encoder else "")
    )

    criterion = BCEDiceLoss(
        bce_weight=args.bce_weight, dice_alpha=args.dice_alpha, dice_beta=args.dice_beta
    )
    # Filtered, not model.parameters(): a frozen param has no grad, so AdamW
    # would skip it anyway, but passing it in still reports it as being
    # optimized and muddles what the run actually trained.
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.lr_patience, factor=0.5
    )

    with ExperimentTracker(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
    ) as tracker:
        tracker.log_config(
            data_config={
                "window_range": (args.window_min, args.window_max),
                "patch_size": args.patch_size,
                "fg_bias_ratio": args.fg_bias_ratio,
                "patches_per_epoch": args.patches_per_epoch,
                "tile_size": args.tile_size,
                "overlap": args.overlap,
            },
            model_config={
                "arch": args.arch,
                "encoder_name": args.encoder_name,
                "decoder_channels": args.decoder_channels,
                "freeze_encoder": args.freeze_encoder,
                "trainable_params": trainable_params,
                "encoder_weights": encoder_weights,
                "decoder_atrous_rates": tuple(args.decoder_atrous_rates),
            },
            loss_config={
                "bce_weight": args.bce_weight,
                "dice_alpha": args.dice_alpha,
                "dice_beta": args.dice_beta,
            },
            optim_config={
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "optimizer": "AdamW",
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
            train_manifest=args.train_manifest,
        )

        best_iou = -1.0
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            tracker.start_epoch()

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, frozen_modules=frozen_modules
            )
            val_summary = validate(model, val_loader, device, criterion=criterion)
            val_iou = val_summary["global"]["iou"]

            scheduler.step(val_iou)
            current_lr = optimizer.param_groups[0]["lr"]

            tracker.log_epoch(epoch, train_loss, val_summary, current_lr)

            is_best = val_iou > best_iou
            is_significant_improvement = val_iou > best_iou + args.min_delta

            if is_best:
                best_iou = val_iou

            if is_significant_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            tracker.log_checkpoint(
                model,
                checkpoint_dir / f"epoch_{epoch}.pt",
                is_best=is_best,
                registered_model_name=f"{args.experiment_name}_model" if is_best else None,
                input_shape=(3, args.patch_size, args.patch_size),
            )

            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_summary['val_loss']:.4f}  "
                f"val_iou={val_iou:.4f}  "
                f"val_dice={val_summary['global']['dice']:.4f}  "
                f"per_image_iou_min={val_summary['per_image_iou_min']:.4f}  "
                f"lr={current_lr:.2e}  best={is_best}  "
                f"patience={epochs_without_improvement}/{args.early_stopping_patience}"
            )

            if report_callback is not None:
                report_callback(epoch, val_iou)  # may raise (e.g. optuna.TrialPruned) -- let it propagate

            if epochs_without_improvement >= args.early_stopping_patience:
                logger.info(f"Early stopping: no val_iou improvement for {args.early_stopping_patience} epochs.")
                break

    logger.info(f"Training run complete. Best val_iou: {best_iou:.4f}")
    return best_iou


def main():
    """Parse CLI arguments and start the training workflow."""
    args = build_argparser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

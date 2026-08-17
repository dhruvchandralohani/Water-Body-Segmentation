"""
train.py

Training loop tying together data.py (loaders), model.py (selectable
architecture, DeepLabV3+ by default), loss.py (selectable pixel term plus a
Tversky region term), metrics.py (per-epoch validation), and tracking.py
(MLflow logging + model registry).

Argument defaults are read from params.yaml, the same file dvc.yaml resolves
its commands from, so a hand-run cannot silently diverge from a pipeline run.

run_training(args, report_callback=None) holds the actual training loop
and returns the best validation IoU; main() is a thin CLI wrapper around
it. tune.py calls run_training() directly, once per trial, with a
callback that reports progress to Optuna and can prune the trial early.

Usage:
    python -m training.train \
        --train-manifest data_pipeline/splits/train.csv \
        --val-manifest data_pipeline/splits/val.csv \
        --image-dir data/raw/images --mask-dir data/processed/masks_corrected \
        --checkpoint-dir training/checkpoints \
        --tracking-uri sqlite:///training/mlflow.db

    Normally you would not: `dvc repro train` runs exactly this with every
    value resolved from params.yaml.
"""

import argparse
import time
from pathlib import Path

import yaml

import torch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from data_pipeline.data import get_eval_loader, get_train_loader
from common.logging_setup import log_run_separator, setup_logger
from training.loss import CombinedLoss
from training.metrics import SegmentationMetrics
from training.model import build_model, freeze_encoder
from training.tracking import ExperimentTracker


def train_one_epoch(model, loader, criterion, optimizer, device, frozen_modules=(), accum_steps=1):
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
        accum_steps: Micro-batches to accumulate before each optimizer step.
            Effective batch size is loader batch size * accum_steps.

            This buys the OPTIMIZATION behaviour of a large batch, not all of it.
            BatchNorm normalises over each micro-batch independently, so
            accumulating 4 x 8 gives gradients averaged over 32 samples while the
            normalisation statistics remain 8-sample estimates. Where small-batch
            training suffers from noisy BN statistics, accumulation does not fix
            it -- only a larger real batch, or a batch-independent norm such as
            GroupNorm, does.

    Returns:
        The mean training loss across the epoch.
    """
    model.train()
    for module in frozen_modules:
        module.eval()

    losses = []
    optimizer.zero_grad(set_to_none=True)
    pending = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        # Divide so the accumulated gradient is the MEAN over the effective
        # batch rather than the sum. Without this the update is accum_steps
        # times too large and a tuned learning rate stops meaning what it meant.
        (loss / accum_steps).backward()
        losses.append(loss.item())  # unscaled, so the logged value stays comparable
        pending += 1

        if pending == accum_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

    # Flush a trailing partial group; otherwise its gradients are computed and
    # then thrown away. Such a group is scaled by accum_steps but holds fewer
    # micro-batches, so its update is proportionally smaller -- immaterial when
    # patches_per_epoch / batch_size divides evenly, as it does at 8000 / 8.
    if pending:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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


# Fallbacks used only when params.yaml cannot be found or read. They exist so
# the script still runs standalone, not as a second place to edit values.
_FALLBACK_DEFAULTS = {
    "window_min": 64, "window_max": 2048, "patch_size": 256, "fg_bias_ratio": 0.7,
    "patches_per_epoch": 8000, "tile_size": 384, "overlap": 0.25, "num_workers": 4,
    "encoder_name": "mobilenet_v2", "encoder_weights": "imagenet",
    "decoder_atrous_rates": [2, 4, 6],
    "bce_weight": 0.5, "dice_alpha": 0.5, "dice_beta": 0.5,
    "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.5, "batch_size": 8, "epochs": 50,
    "early_stopping_patience": 6, "min_delta": 0.001, "lr_patience": 3, "seed": 42,
    "experiment_name": "water_body_segmentation",
}


def load_param_defaults(path=None):
    """Read argparse defaults from params.yaml so there is one source of truth.

    The pipeline passes every value explicitly on the command line, so these
    defaults only bite when the script is run by hand. That is exactly when
    drift is dangerous: hardcoded defaults silently fell out of step with the
    tuned values in params.yaml (lr defaulted to 1e-3 against a tuned 9.18e-05,
    an 11x gap), so a manual run looked like a reproduction and was not.
    Reading the same file the pipeline reads makes that divergence impossible
    rather than merely discouraged.

    Args:
        path: Explicit params.yaml path. Defaults to the repo root beside this
            package.

    Returns:
        A flat mapping of parameter name to value, falling back to
        _FALLBACK_DEFAULTS for anything missing.
    """
    defaults = dict(_FALLBACK_DEFAULTS)

    path = Path(path) if path else Path(__file__).resolve().parents[1] / "params.yaml"
    try:
        with open(path) as f:
            params = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return defaults

    # Only sections describing a training run. paths/audit/evaluate and the
    # experiment grids are read by dvc.yaml, and pulling them in here would
    # collide on shared key names such as `seed` and `batch_size`.
    for section in ("data", "model", "loss", "train"):
        for key, value in (params.get(section) or {}).items():
            defaults[key] = value

    # nargs=3 flag, stored as "2 4 6" so it expands correctly inside a
    # dvc.yaml cmd string. Convert back to the list argparse would produce.
    rates = defaults.get("decoder_atrous_rates")
    if isinstance(rates, str):
        defaults["decoder_atrous_rates"] = [int(x) for x in rates.split()]

    return defaults


def optional_int(value):
    """Parse an int flag that may arrive as a null placeholder.

    DVC interpolates params.yaml values into command strings, so a YAML `null`
    reaches argparse as the text "None". type=int rejects that, which would
    crash every grid arm that leaves the value unset before training starts.

    Args:
        value: Raw command-line token.

    Returns:
        An int, or None for an empty/null placeholder.
    """
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return None
    return int(value)


def str2bool(value):
    """Parse a boolean flag that must also accept an explicit value.

    A plain store_true flag cannot be templated: `--freeze-encoder ${item.x}`
    has nowhere to put the value, so a foreach grid can only ever pass the flag
    or omit it, never vary it per arm. Accepting a value keeps the grid honest
    while `--freeze-encoder` on its own still works.

    Args:
        value: Raw command-line token.

    Returns:
        The parsed boolean.

    Raises:
        argparse.ArgumentTypeError: If the token is not a recognised boolean.
    """
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in ("true", "t", "yes", "y", "1"):
        return True
    if token in ("false", "f", "no", "n", "0", "none", "null", ""):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def build_argparser():
    """Create the CLI argument parser for the training script.

    Defaults come from params.yaml where available -- see load_param_defaults().
    """
    d = load_param_defaults()
    parser = argparse.ArgumentParser(description="Train the water body segmentation model.")

    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints")

    # Data pipeline
    parser.add_argument("--window-min", type=int, default=d["window_min"])
    parser.add_argument("--window-max", type=int, default=d["window_max"])
    parser.add_argument("--patch-size", type=int, default=d["patch_size"])
    parser.add_argument("--fg-bias-ratio", type=float, default=d["fg_bias_ratio"])
    parser.add_argument("--patches-per-epoch", type=int, default=d["patches_per_epoch"])
    parser.add_argument("--tile-size", type=int, default=d["tile_size"])
    parser.add_argument("--overlap", type=float, default=d["overlap"])
    parser.add_argument("--num-workers", type=int, default=d["num_workers"])

    # Model
    parser.add_argument(
        "--arch",
        default="deeplabv3plus",
        help="Architecture family (deeplabv3plus, unet, fpn, ...). Decoder args "
        "the chosen arch does not accept are dropped with a warning.",
    )
    parser.add_argument("--encoder-name", default=d["encoder_name"])
    parser.add_argument(
        "--decoder-channels",
        type=optional_int,
        default=None,
        help="Decoder width. Leave unset for smp's default. This is the "
        "capacity lever; --dropout is the regularization lever.",
    )
    parser.add_argument(
        "--freeze-encoder",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Freeze the pretrained encoder (params AND BatchNorm statistics). "
        "Reduces trainable capacity instead of regularizing it. Takes an optional "
        "explicit value so a foreach grid can vary it per arm.",
    )
    parser.add_argument("--encoder-weights", default=d["encoder_weights"])
    parser.add_argument(
        "--decoder-atrous-rates", type=int, nargs=3, default=d["decoder_atrous_rates"],
        help="Recalibrate if --patch-size changes.",
    )

    # Loss
    parser.add_argument(
        "--pixel-loss",
        default="bce",
        choices=["bce", "weighted_bce", "focal"],
        help="Pixel-wise term paired with the Tversky region term.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="Positive-class multiplier for --pixel-loss weighted_bce. Set to the "
        "measured background:foreground pixel ratio (see metrics/class_balance.json).",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.5,
        help="Positive-class weight for --pixel-loss focal. 0.5 is neutral; the "
        "paper's 0.25 down-weights foreground and is wrong for this imbalance.",
    )
    parser.add_argument("--bce-weight", type=float, default=d["bce_weight"],
                        help="Weight on the pixel-wise term; 1 - this goes to Tversky.")
    parser.add_argument("--dice-alpha", type=float, default=d["dice_alpha"])
    parser.add_argument("--dice-beta", type=float, default=d["dice_beta"])

    # Optimization
    parser.add_argument("--lr", type=float, default=d["lr"])
    parser.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    parser.add_argument("--dropout", type=float, default=d["dropout"], help="ASPP dropout.")
    parser.add_argument("--batch-size", type=int, default=d["batch_size"])
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Micro-batches accumulated per optimizer step. Effective batch is "
        "batch-size * this. Reaches batch sizes the card cannot hold, but leaves "
        "BatchNorm statistics at the micro-batch size.",
    )
    parser.add_argument("--epochs", type=int, default=d["epochs"])
    parser.add_argument("--early-stopping-patience", type=int, default=d["early_stopping_patience"])
    parser.add_argument("--min-delta", type=float, default=d["min_delta"])
    parser.add_argument("--lr-patience", type=int, default=d["lr_patience"], help="ReduceLROnPlateau patience.")

    # Tracking
    parser.add_argument("--experiment-name", default=d["experiment_name"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking backend. Resolved relative to the current working "
        "directory, so pass an explicit path (e.g. sqlite:///training/mlflow.db) "
        "when running from the repo root or under DVC.",
    )

    parser.add_argument("--seed", type=int, default=d["seed"])
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
        f"batch_size={args.batch_size} accum_steps={args.accum_steps} "
        f"effective_batch={args.batch_size * args.accum_steps} "
        f"patches_per_epoch={args.patches_per_epoch} "
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

    criterion = CombinedLoss(
        pixel_loss=args.pixel_loss,
        pixel_weight=args.bce_weight,
        tversky_alpha=args.dice_alpha,
        tversky_beta=args.dice_beta,
        pos_weight=args.pos_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
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
                "pixel_loss": args.pixel_loss,
                "pos_weight": args.pos_weight,
                "focal_gamma": args.focal_gamma,
                "tversky_alpha": args.dice_alpha,
                "tversky_beta": args.dice_beta,
                "encoder_name": args.encoder_name,
                "decoder_channels": args.decoder_channels,
                "freeze_encoder": args.freeze_encoder,
                "trainable_params": trainable_params,
                "accum_steps": args.accum_steps,
                "effective_batch_size": args.batch_size * args.accum_steps,
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
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                frozen_modules=frozen_modules,
                accum_steps=args.accum_steps,
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

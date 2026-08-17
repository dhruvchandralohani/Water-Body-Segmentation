"""
model.py

Model definition. Defaults to DeepLabV3+ with a pretrained CNN encoder,
chosen for this dataset specifically because ASPP's multi-scale dilated
convolutions directly target the same multi-scale problem the data pipeline
was built around, and a lightweight encoder (MobileNetV2 by default) fits a
4GB GPU.

decoder_atrous_rates is overridden from smp's default (12, 24, 36) --
those were calibrated for 513x513-class crops (~33x33 feature maps at
output_stride=16). At our patch_size=256, the deepest feature map is only
16x16, and (12, 24, 36) span 25-73px each -- only the center tap of every
3x3 dilated kernel lands inside a 16x16 feature map, so all three ASPP
branches degenerate to effectively the same single-point convolution.
(2, 4, 6) actually fits and differentiates at this feature map size.

`arch` opens this up to other families so that choice can be measured rather
than only argued. Architectures accept different decoder arguments -- U-Net
has no ASPP and so no atrous rates -- so requested decoder kwargs are filtered
against the constructor's real signature and anything dropped is reported.
Silently ignoring a hyperparameter for one variant of a benchmark would make
that benchmark unfair without it being visible anywhere; a logged warning and
the applied/dropped record returned by describe_variant() make it visible.

Outputs raw logits (activation=None) -- pair with BCEWithLogitsLoss and
metrics.py's evaluate(), both of which expect logits, not pre-sigmoided
probabilities.
"""

import gc
import inspect
import logging

import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

# Lowercase arch name -> smp class attribute. Explicit rather than reaching
# into smp internals, so a version bump can't silently change resolution.
ARCH_CLASSES = {
    "deeplabv3plus": "DeepLabV3Plus",
    "deeplabv3": "DeepLabV3",
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "manet": "MAnet",
    "linknet": "Linknet",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "pan": "PAN",
    "segformer": "Segformer",
}


def resolve_arch(arch):
    """Look up the smp model class for an architecture name.

    Args:
        arch: Architecture name, case-insensitive (e.g. "unet", "DeepLabV3Plus").

    Returns:
        The smp model class.

    Raises:
        ValueError: If the name is unknown or not present in the installed smp.
    """
    key = arch.lower().replace("-", "").replace("_", "").replace("+", "plus")
    if key not in ARCH_CLASSES:
        raise ValueError(f"Unknown arch {arch!r}. Supported: {', '.join(sorted(ARCH_CLASSES))}")

    cls = getattr(smp, ARCH_CLASSES[key], None)
    if cls is None:
        raise ValueError(
            f"arch {arch!r} maps to smp.{ARCH_CLASSES[key]}, which is not available in "
            f"segmentation_models_pytorch {smp.__version__}. Upgrade smp or pick another arch."
        )
    return cls


def split_supported_kwargs(cls, candidate_kwargs):
    """Partition requested kwargs into those the constructor accepts and those it does not.

    Args:
        cls: The model class whose __init__ signature is inspected.
        candidate_kwargs: Mapping of kwarg name to value to be filtered.

    Returns:
        A tuple of (applied, dropped) dictionaries.
    """
    params = inspect.signature(cls.__init__).parameters
    # A constructor taking **kwargs would swallow anything, so only trust
    # explicitly named parameters.
    accepted = {name for name, p in params.items() if p.kind is not inspect.Parameter.VAR_KEYWORD}

    applied = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    dropped = {k: v for k, v in candidate_kwargs.items() if k not in accepted}
    return applied, dropped


def build_model(
    arch="deeplabv3plus",
    encoder_name="mobilenet_v2",
    encoder_weights: "str | None" = "imagenet",
    in_channels=3,
    classes=1,
    decoder_atrous_rates=(2, 4, 6),
    decoder_aspp_dropout=0.5,
    decoder_channels=None,
):
    """Build a segmentation model with the requested architecture and encoder.

    Decoder arguments that the chosen architecture does not accept are dropped
    and logged at WARNING level rather than raising, so one config can drive a
    multi-architecture benchmark. Check the warnings before trusting a
    comparison: a dropped argument means that variant did not receive a setting
    the others did.

    Args:
        arch: Architecture family name (see ARCH_CLASSES).
        encoder_name: Name of the segmentation-models-pytorch encoder backbone.
        encoder_weights: Optional pretrained weights for the encoder.
        in_channels: Number of input channels in the image tensor.
        classes: Number of output segmentation classes.
        decoder_atrous_rates: ASPP dilation rates. ASPP-bearing archs only.
        decoder_aspp_dropout: Dropout probability in the ASPP block. ASPP-bearing
            archs only -- U-Net and friends have no equivalent knob, so they run
            without this regularization.
        decoder_channels: Decoder width. None leaves smp's default alone. Lowering
            it is the capacity-reduction lever, as opposed to the
            regularization lever that decoder_aspp_dropout provides.

    Returns:
        A configured model that returns raw logits.
    """
    cls = resolve_arch(arch)

    decoder_kwargs = {
        "decoder_atrous_rates": tuple(decoder_atrous_rates),
        "decoder_aspp_dropout": decoder_aspp_dropout,
    }
    if decoder_channels is not None:
        # Only sent when explicitly requested -- passing None through would
        # override smp's own default with an invalid value.
        decoder_kwargs["decoder_channels"] = decoder_channels
    applied, dropped = split_supported_kwargs(cls, decoder_kwargs)

    if dropped:
        logger.warning(
            "%s does not accept %s -- these settings are NOT applied to this variant. "
            "Any benchmark including it differs by more than architecture alone.",
            cls.__name__,
            ", ".join(sorted(dropped)),
        )

    return cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # raw logits -- required for BCEWithLogitsLoss / metrics.py's evaluate()
        **applied,
    )


def freeze_encoder(model):
    """Freeze the pretrained encoder and return the modules that must stay in eval mode.

    Two things have to happen for an encoder to actually be frozen, and only the
    first is obvious:

    1. requires_grad = False, so no gradients and no optimizer updates.
    2. BatchNorm layers held in eval() mode. requires_grad does NOT stop a BN
       layer from updating its running mean and variance -- those are buffers,
       not parameters. A "frozen" encoder left in train mode keeps drifting its
       normalization statistics toward the new data, which at batch_size 8 is
       both noisy and not what freezing was supposed to mean.

    Point 2 has to be re-applied after every model.train() call, because
    model.train() walks the whole module tree. That is why this returns the
    modules rather than just doing it once -- the caller re-applies each epoch.

    Args:
        model: An smp model exposing a .encoder attribute.

    Returns:
        A list containing the encoder module, to be put back into eval() mode
        after each model.train() call.

    Raises:
        AttributeError: If the model has no .encoder attribute.
    """
    if not hasattr(model, "encoder"):
        raise AttributeError(f"{type(model).__name__} has no .encoder to freeze")

    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()
    return [model.encoder]


def count_parameters(model):
    """Total and trainable parameter counts -- useful for checking the model fits your VRAM budget."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_peak_vram(model, batch_size=8, patch_size=256, in_channels=3, iterations=2):
    """Run real training steps on dummy data and report peak GPU memory.

    Runs forward, loss, backward AND optimizer.step(). The step matters: AdamW
    allocates its exp_avg and exp_avg_sq buffers lazily on the first step, which
    is another 2x the parameter memory. Measuring without it understates a
    22M-parameter model by ~180MB, and on a 4GB card that is the difference
    between fitting and not.

    Two iterations by default, because cuDNN autotuning can allocate extra
    workspace on the first call and the peak is what decides whether a run
    survives.

    Args:
        model: Model to probe. Moved to CUDA and left there.
        batch_size: Batch size to probe at -- use the real training value.
        patch_size: Spatial size of the dummy input.
        in_channels: Channel count of the dummy input.
        iterations: How many train steps to run.

    Returns:
        A dictionary with allocated_mb, reserved_mb and oom keys, or a dict with
        available=False if CUDA is not present.
    """
    import torch

    from training.loss import CombinedLoss

    if not torch.cuda.is_available():
        return {"available": False, "oom": False, "allocated_mb": 0.0, "reserved_mb": 0.0}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CombinedLoss()

    x = torch.randn(batch_size, in_channels, patch_size, patch_size, device="cuda")
    y = torch.randint(0, 2, (batch_size, 1, patch_size, patch_size), device="cuda").float()

    oom = False
    try:
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    except torch.cuda.OutOfMemoryError:
        oom = True

    allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
    reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    # Hand the memory back before the next variant is probed, otherwise
    # fragmentation from this one inflates the next one's numbers.
    del model, optimizer, criterion, x, y
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Built as one literal rather than key-by-key: a dict initialised with only
    # bool values narrows to dict[str, bool], and assigning a float into it
    # afterwards is a type error to static checkers even though it runs fine.
    return {
        "available": True,
        "oom": oom,
        "allocated_mb": allocated_mb,
        "reserved_mb": reserved_mb,
    }


def describe_variant(
    arch,
    encoder_name,
    decoder_atrous_rates=(2, 4, 6),
    decoder_aspp_dropout=0.5,
    probe_vram=False,
    batch_size=8,
    patch_size=256,
):
    """Report what a benchmark variant would actually be built with, without training it.

    Use this to confirm every variant received the settings you intended, and
    fits the card, before committing compute to a comparison.

    Args:
        arch: Architecture family name.
        encoder_name: Encoder backbone name.
        decoder_atrous_rates: Requested ASPP dilation rates.
        decoder_aspp_dropout: Requested ASPP dropout.
        probe_vram: If True, run real train steps and measure peak GPU memory.
        batch_size: Batch size used for the VRAM probe.
        patch_size: Patch size used for the VRAM probe.

    Returns:
        A dictionary describing the resolved class, applied and dropped decoder
        kwargs, parameter counts, and (if probed) peak memory.
    """
    cls = resolve_arch(arch)
    applied, dropped = split_supported_kwargs(
        cls,
        {
            "decoder_atrous_rates": tuple(decoder_atrous_rates),
            "decoder_aspp_dropout": decoder_aspp_dropout,
        },
    )
    model = build_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=None,  # skip the download; parameter count is identical
        decoder_atrous_rates=decoder_atrous_rates,
        decoder_aspp_dropout=decoder_aspp_dropout,
    )
    total, trainable = count_parameters(model)
    info = {
        "arch": arch,
        "class": cls.__name__,
        "encoder_name": encoder_name,
        "applied": applied,
        "dropped": sorted(dropped),
        "total_params": total,
        "trainable_params": trainable,
    }
    if probe_vram:
        info["vram"] = measure_peak_vram(model, batch_size=batch_size, patch_size=patch_size)
    return info


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Dry-run the benchmark grid: what each variant is actually built with, "
        "and optionally whether it fits the card."
    )
    parser.add_argument(
        "--probe-vram",
        action="store_true",
        help="Run two real train steps per variant on dummy data and report peak GPU memory.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the VRAM probe.")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size for the VRAM probe.")
    args = parser.parse_args()

    # Anything under "dropped" is a setting that variant will not receive.
    variants = [
        ("deeplabv3plus", "mobilenet_v2", (2, 4, 6)),
        ("deeplabv3plus", "mobilenet_v2", (12, 24, 36)),
        ("unet", "mobilenet_v2", (2, 4, 6)),
        ("deeplabv3plus", "resnet34", (2, 4, 6)),
    ]

    for arch, encoder, rates in variants:
        info = describe_variant(
            arch,
            encoder,
            decoder_atrous_rates=rates,
            probe_vram=args.probe_vram,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
        )
        line = (
            f"{info['class']:<16} {encoder:<14} rates={rates!s:<12} "
            f"{info['total_params']:>11,} params"
        )
        line += f"  dropped: {', '.join(info['dropped']) or 'none'}"

        vram = info.get("vram")
        if vram is None:
            pass
        elif not vram["available"]:
            line += "  |  vram: no CUDA device"
        elif vram["oom"]:
            line += f"  |  vram: OOM at batch {args.batch_size}"
        else:
            line += f"  |  vram: {vram['allocated_mb']:>6.0f} MB alloc / {vram['reserved_mb']:>6.0f} MB reserved"
        print(line)

    if args.probe_vram:
        print(
            "\nReported memory is allocator-tracked tensors only. The CUDA context itself "
            "costs roughly another 300-600 MB outside these numbers, so compare 'reserved' "
            "plus that margin against your card, not 'allocated'."
        )

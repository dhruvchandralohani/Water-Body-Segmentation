"""
test_model_arch.py

Named for the architecture rather than the module, because tests/test_model.py
collided with training/test_model.py -- the evaluation script -- and the two
were confusable enough that this file went missing in a copy.

Contracts the training loop assumes and never checks.

The one worth stating plainly: models must emit RAW LOGITS. BCEWithLogitsLoss
and metrics.evaluate() both apply their own sigmoid, so a model constructed with
activation="sigmoid" would be squashed twice. That does not raise -- it
compresses the effective range, flattens gradients, and produces a run that
merely underperforms. There is no symptom to notice.

Everything here builds models with encoder_weights=None to avoid a download in
CI. Architecture, shapes and parameter counts are identical either way.
"""

import pytest
import torch

from training.model import (
    ARCH_CLASSES,
    build_model,
    count_parameters,
    freeze_encoder,
    resolve_arch,
    split_supported_kwargs,
)

# The three families in the benchmark grid, with the encoder each requires.
BENCHMARK_ARCHS = [
    ("deeplabv3plus", "mobilenet_v2"),
    ("unet", "mobilenet_v2"),
    ("segformer", "mit_b0"),
]


def tiny_model(arch, encoder):
    """Build one benchmark variant without downloading pretrained weights."""
    return build_model(arch=arch, encoder_name=encoder, encoder_weights=None)


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch,encoder", BENCHMARK_ARCHS)
@pytest.mark.parametrize("size", [256, 320])
def test_output_matches_input_spatial_size(arch, encoder, size):
    """Every architecture must return a mask the same size as its input.

    Also covers the divisible-by-32 requirement all three share: 320 is a second
    valid size, so a model that silently downsampled would show up here rather
    than as a shape error deep inside the loss.
    """
    model = tiny_model(arch, encoder).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size))

    assert out.shape == (2, 1, size, size), f"{arch} returned {tuple(out.shape)}"


def final_conv(model):
    """The last convolution before the output, whose bias reaches every pixel."""
    head = getattr(model, "segmentation_head", None)
    assert head is not None, f"{type(model).__name__} has no segmentation_head to inspect"
    convs = [m for m in head.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs, "segmentation_head contains no Conv2d"
    return convs[-1]


@pytest.mark.parametrize("arch,encoder", BENCHMARK_ARCHS)
def test_segmentation_head_contains_no_activation(arch, encoder):
    """Structural half of the logits contract: nothing may squash the output.

    Checked by inspection rather than by observing the output range. An earlier
    version asserted that values escape [0, 1], which is only a proxy -- a
    randomly initialised DeepLabV3+ emitted 0.0341 to 0.0380, entirely inside
    the interval, and failed a model that was perfectly correct.
    """
    banned = (torch.nn.Sigmoid, torch.nn.Softmax, torch.nn.Softmax2d, torch.nn.Tanh)
    offenders = [
        type(m).__name__ for m in model_head(arch, encoder).modules() if isinstance(m, banned)
    ]
    assert not offenders, f"{arch} segmentation head applies {offenders}"


def model_head(arch, encoder):
    """The segmentation head of a freshly built variant."""
    return tiny_model(arch, encoder).segmentation_head


@pytest.mark.parametrize("arch,encoder", BENCHMARK_ARCHS)
def test_output_is_unsquashed(arch, encoder):
    """Behavioural half: a known bias must arrive at the output untouched.

    Zeroing the final conv's weights makes its output exactly its bias, so the
    model emits a constant. Set that to 5.0: raw logits give 5.0, while a
    sigmoid would give 0.9933. Deterministic, and it does not depend on
    initialisation luck the way an output-range check does.

    This matters because both BCEWithLogitsLoss and metrics.evaluate() apply
    their own sigmoid. A pre-activated model would be squashed twice -- training
    still runs and still converges, just worse, with no error to notice.
    """
    model = tiny_model(arch, encoder).eval()
    conv = final_conv(model)
    assert conv.bias is not None, "final conv has no bias; probe needs one"

    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.fill_(5.0)
        out = model(torch.randn(1, 3, 256, 256))

    assert out.max() > 1.0, f"{arch} output squashed into [0, 1]; it appears pre-activated"
    assert out.mean().item() == pytest.approx(5.0, abs=0.01)


@pytest.mark.parametrize("arch,encoder", BENCHMARK_ARCHS)
def test_gradients_reach_the_encoder(arch, encoder):
    """A full backward pass must produce finite gradients on encoder weights.

    Catches a detached graph or a frozen-by-accident backbone, either of which
    trains happily while learning nothing in the encoder.
    """
    model = tiny_model(arch, encoder).train()
    out = model(torch.randn(2, 3, 256, 256))
    out.mean().backward()

    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads, f"{arch}: no encoder parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


# ---------------------------------------------------------------------------
# Architecture resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("deeplabv3plus", "DeepLabV3Plus"),
        ("DeepLabV3Plus", "DeepLabV3Plus"),
        ("deeplab_v3_plus", "DeepLabV3Plus"),
        ("deeplabv3+", "DeepLabV3Plus"),
        ("unet", "Unet"),
        ("UNet", "Unet"),
        ("segformer", "Segformer"),
    ],
)
def test_arch_names_normalise(name, expected):
    """Spelling variants resolve to one class, so a grid arm cannot miss by a dash."""
    assert resolve_arch(name).__name__ == expected


def test_unknown_arch_raises_with_the_supported_list():
    """A typo must fail at construction and say what was expected.

    `resnet50` is the realistic mistake -- an encoder name in the arch slot.
    """
    with pytest.raises(ValueError, match="Unknown arch"):
        resolve_arch("resnet50")


def test_every_registered_arch_resolves():
    """ARCH_CLASSES must not name a class the installed smp does not provide."""
    for name in ARCH_CLASSES:
        assert resolve_arch(name) is not None


# ---------------------------------------------------------------------------
# Decoder kwarg filtering
# ---------------------------------------------------------------------------


def test_aspp_kwargs_are_dropped_for_architectures_without_aspp():
    """U-Net and SegFormer have no ASPP, so those settings must not be passed.

    Passing them would raise a TypeError and kill the arm; filtering them
    silently would hide a real asymmetry. The code does neither -- it drops them
    and warns, and this pins which side of that line each arch falls on.
    """
    requested = {"decoder_atrous_rates": (2, 4, 6), "decoder_aspp_dropout": 0.5}

    applied, dropped = split_supported_kwargs(resolve_arch("deeplabv3plus"), requested)
    assert set(applied) == set(requested) and not dropped

    for arch in ("unet", "segformer"):
        applied, dropped = split_supported_kwargs(resolve_arch(arch), requested)
        assert not applied and set(dropped) == set(requested), f"{arch}"


def test_kwarg_filtering_warns_when_it_drops_something(caplog):
    """The drop must be visible in the log, not silent.

    A benchmark arm that quietly did not receive a setting the others did would
    differ by more than architecture, and nothing would say so.
    """
    with caplog.at_level("WARNING"):
        build_model(arch="unet", encoder_name="mobilenet_v2", encoder_weights=None)

    assert any("does not accept" in record.message for record in caplog.records)


def test_decoder_channels_is_only_forwarded_when_set():
    """None must leave smp's own default alone rather than overriding it with None.

    The capacity grid leaves this unset on three of four arms, so the None path
    is the common one.
    """
    default = build_model(arch="unet", encoder_name="mobilenet_v2", encoder_weights=None)
    narrowed = build_model(
        arch="unet", encoder_name="mobilenet_v2", encoder_weights=None, decoder_channels=(64, 32, 16, 8, 4)
    )

    assert count_parameters(narrowed)[0] < count_parameters(default)[0]


# ---------------------------------------------------------------------------
# Encoder freezing
# ---------------------------------------------------------------------------


def test_freeze_encoder_stops_gradients_but_leaves_the_decoder_trainable():
    """Freezing must reduce trainable parameters without disabling learning."""
    model = build_model(arch="deeplabv3plus", encoder_name="mobilenet_v2", encoder_weights=None)
    total_before, trainable_before = count_parameters(model)

    freeze_encoder(model)
    total_after, trainable_after = count_parameters(model)

    assert total_after == total_before
    assert trainable_after < trainable_before
    assert trainable_after > 0, "the decoder must still train"
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_freeze_encoder_returns_modules_that_must_be_re_evaled_each_epoch():
    """requires_grad does not freeze BatchNorm; its running stats are buffers.

    model.train() walks the whole tree, so a one-time encoder.eval() is undone
    at the start of every epoch. The function returns the modules precisely so
    train_one_epoch can re-apply it -- this asserts that contract holds.
    """
    model = build_model(arch="deeplabv3plus", encoder_name="mobilenet_v2", encoder_weights=None)
    frozen = freeze_encoder(model)

    assert frozen, "freeze_encoder must return the modules to hold in eval mode"
    assert all(not module.training for module in frozen)

    model.train()
    assert model.encoder.training, "model.train() should have reverted the encoder, as documented"

    for module in frozen:
        module.eval()
    assert not model.encoder.training


def test_freeze_encoder_rejects_a_model_without_an_encoder():
    """A clear failure beats silently training everything."""
    with pytest.raises(AttributeError, match="encoder"):
        freeze_encoder(torch.nn.Linear(2, 2))


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def test_count_parameters_separates_total_from_trainable():
    """Both numbers are logged to MLflow and used to compare grid arms fairly."""
    model = build_model(arch="unet", encoder_name="mobilenet_v2", encoder_weights=None)
    total, trainable = count_parameters(model)

    assert total == trainable > 0
    for param in model.encoder.parameters():
        param.requires_grad = False
    assert count_parameters(model)[1] < total


def test_benchmark_arms_are_reported_with_their_sizes():
    """Parameter counts must be comparable across arms, since the encoder varies.

    SegFormer needs mit_b0 -- a MobileNetV2 encoder inside it would not be a
    transformer -- so the arms cannot be capacity-matched and the counts belong
    beside any reported score.
    """
    sizes = {arch: count_parameters(tiny_model(arch, enc))[0] for arch, enc in BENCHMARK_ARCHS}

    assert len(set(sizes.values())) == len(sizes), "arms should differ in size; report the counts"
    assert all(size > 0 for size in sizes.values())

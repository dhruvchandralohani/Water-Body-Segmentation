"""
tune.py

Optuna hyperparameter search over learning rate, weight decay, and
dropout. Calls run_training() from train.py directly
once per trial (not via subprocess, so no repeated Python startup cost
or data reloading overhead), with a callback that reports progress to
Optuna each epoch and prunes clearly-underperforming trials early.

Reuses train.py's argparser as a base, then adds tuning-specific flags.
Note: --lr, --weight-decay, and --dropout are inherited
from that base parser but are overridden per trial by Optuna regardless
of what's passed on the command line.

Usage:
    python tune.py --train-manifest splits/train.csv --val-manifest splits/val.csv \
        --image-dir images/ --mask-dir masks_corrected/ --epochs 8 --n-trials 20
"""

import copy
import json
import sys as _sys
from collections import Counter
from pathlib import Path as _Path

import optuna
import torch

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from common.logging_setup import log_run_separator, setup_logger
from training.model import resolve_arch, split_supported_kwargs
from training.train import build_argparser, run_training

logger = setup_logger("tune", log_file="tuning.log")


def suggest_args(trial, base_args):
    """Create a trial-specific argument set with Optuna-suggested hyperparameters.

    Args:
        trial: Optuna trial object that provides parameter suggestions.
        base_args: Base argument namespace from the training parser.

    Returns:
        A deep-copied argument namespace with tuning-specific values applied.
    """
    trial_args = copy.deepcopy(base_args)
    trial_args.lr = trial.suggest_float("lr", base_args.lr_min, base_args.lr_max, log=True)
    # Inherited from base_args, not pinned here. A hardcoded 8 silently
    # discarded whatever --batch-size the caller passed, which is how the
    # earlier batch-size mismatch between train.py and tune.py went unnoticed:
    # the value that actually ran was neither the CLI's nor train.py's default.
    trial_args.batch_size = base_args.batch_size
    trial_args.weight_decay = trial.suggest_float(
        "weight_decay", base_args.weight_decay_min, base_args.weight_decay_max, log=True
    )

    # Dropout is searched only where it exists. decoder_aspp_dropout is an ASPP
    # parameter, so U-Net and SegFormer drop it -- searching it there would spend
    # trials varying something the model never receives, and produce a study whose
    # reported best dropout is meaningless.
    if supports_aspp_dropout(base_args.arch):
        trial_args.dropout = trial.suggest_float("dropout", base_args.dropout_min, base_args.dropout_max)
    else:
        trial_args.dropout = 0.0

    trial_args.run_name = f"trial_{trial.number}"
    trial_args.checkpoint_dir = f"{base_args.checkpoint_dir}/trial_{trial.number}"

    return trial_args


def make_objective(base_args):
    """Create an Optuna objective function for the given base configuration."""

    def objective(trial):
        """Run one Optuna trial by training a model and reporting the validation IoU."""
        trial_args = suggest_args(trial, base_args)
        logger.info(
            f"Trial {trial.number} starting: lr={trial_args.lr:.2e} "
            f"batch_size={trial_args.batch_size} weight_decay={trial_args.weight_decay:.2e} "
            f"dropout={trial_args.dropout:.3f}"
        )

        # An enqueued trial is a configuration someone chose, not one the sampler
        # proposed. Pruning it against the median of a search it was not competing
        # in would cut short the very run that was meant to be measured.
        enqueued = "fixed_params" in trial.system_attrs

        def report_callback(epoch, val_iou):
            """Report to Optuna and prune only if the pruner says so."""
            trial.report(val_iou, epoch)
            if not enqueued and trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch} (val_iou={val_iou:.4f})")
                # `from None`: pruning is the expected outcome here, not an error
                # in the handler, so a chained traceback is noise.
                raise optuna.TrialPruned() from None

        try:
            val_iou = run_training(trial_args, report_callback=report_callback)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # Anything that is not a memory failure is a real bug: re-raise it.
            # Silently pruning genuine errors would turn a broken study into one
            # that merely looks unlucky.
            if not is_memory_error(exc):
                raise

            # Every GPU memory failure ends the process, not only the device-level
            # ones. Observed: a trial hit an allocator OOM, was pruned, and the
            # NEXT trial died inside torch.manual_seed -- before allocating a
            # single tensor. The context was already unusable. Continuing past one
            # of these does not resume the search, it just records a run of
            # failures that say nothing about their hyperparameters.
            free_gpu_memory()
            raise StudyAborted(
                f"Trial {trial.number} ran out of GPU memory ({type(exc).__name__}) "
                f"at batch_size={trial_args.batch_size}. The CUDA context cannot be "
                "trusted afterwards, so the study stops here. Completed trials are "
                "saved -- free GPU memory and rerun to continue."
            ) from None

        logger.info(f"Trial {trial.number} finished: val_iou={val_iou:.4f}")
        return val_iou

    return objective


# cuDNN and cuBLAS surface exhausted workspace as opaque status codes on a plain
# RuntimeError. Matching on the message is unpleasant but it is the only signal
# available -- torch does not classify these as OutOfMemoryError.
_MEMORY_ERROR_MARKERS = (
    "out of memory",
    "CUDNN_STATUS_EXECUTION_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "cuDNN error",
)


class StudyAborted(Exception):
    """Raised to end a study early for a reason that is not a trial's fault.

    Distinct from an ordinary failure: the completed trials are valid and
    already persisted, so the stage should exit cleanly rather than report an
    error and leave DVC unable to record the work.
    """


def free_gpu_memory():
    """Best-effort cache release that cannot itself break the caller.

    empty_cache() raises when the context is already broken, which turned a
    handled OOM into an unhandled one and killed the study during cleanup.
    """
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def is_memory_error(exc):
    """Whether an exception is a GPU memory failure, however it was reported.

    Args:
        exc: The caught exception.

    Returns:
        True if it should be treated as OOM and pruned rather than raised.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return any(marker in str(exc) for marker in _MEMORY_ERROR_MARKERS)


def supports_aspp_dropout(arch):
    """Whether this architecture actually accepts decoder_aspp_dropout.

    Asked of the model constructor's real signature rather than hardcoded, so a
    newly supported architecture needs no change here.

    Args:
        arch: Architecture name.

    Returns:
        True if the constructor takes decoder_aspp_dropout.
    """
    applied, _ = split_supported_kwargs(resolve_arch(arch), {"decoder_aspp_dropout": 0.5})
    return "decoder_aspp_dropout" in applied


def build_tune_argparser():
    """Build the CLI parser for hyperparameter tuning with shared training options."""
    parser = build_argparser()  # reuse every train.py argument as a base
    parser.description = "Hyperparameter search for the water body segmentation model."
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default="water_body_tuning")
    parser.add_argument(
        "--study-storage",
        default="sqlite:///optuna_study.db",
        help="Persistent storage so a study can resume across separate runs.",
    )
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-warmup-epochs", type=int, default=3)

    # Search bounds are arguments, not literals, because the same study drives
    # every architecture now. A range narrowed around one architecture's previous
    # answer would quietly prevent any other from finding its own -- transformers
    # in particular usually want a lower learning rate than CNNs.
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--weight-decay-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay-max", type=float, default=1e-1)
    parser.add_argument("--dropout-min", type=float, default=0.0)
    parser.add_argument("--dropout-max", type=float, default=0.5)
    parser.add_argument(
        "--enqueue-params",
        default="",
        help='JSON object of parameter values to run as the next trial instead of '
        'sampling, e.g. \'{"lr": 4.2e-04, "dropout": 0.04}\'. This is how a chosen '
        "configuration is trained: it becomes an ordinary trial in the same study, "
        "directly comparable with the sampled ones.",
    )
    return parser


def main():
    """Run the Optuna hyperparameter search from the command line."""
    args = build_tune_argparser().parse_args()

    log_run_separator(logger, f"Hyperparameter search: study={args.study_name} n_trials={args.n_trials}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.study_storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_epochs,
        ),
    )

    if args.enqueue_params:
        fixed = json.loads(args.enqueue_params)
        study.enqueue_trial(fixed)
        logger.info(f"Enqueued a fixed-parameter trial: {fixed} (will not be pruned)")

    try:
        study.optimize(
            make_objective(args),
            n_trials=args.n_trials,
            # Trial-level handling above already prunes memory failures; this is the
            # backstop for one raised outside run_training.
            catch=(torch.cuda.OutOfMemoryError,),
        )
    except StudyAborted as abort:
        # Not an error exit. Optuna has already persisted every completed trial,
        # so the run produced real work; reporting failure would only stop DVC
        # recording it and make the summary below unreachable.
        logger.warning(str(abort))

    # best_trial raises when no trial COMPLETED -- every one failed or was
    # pruned. That is a legitimate outcome of a short or unlucky study, so
    # report it rather than crashing the stage on the way out.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    states = Counter(t.state.name for t in study.trials)
    logger.info(f"Search complete. Trials by state: {dict(states)}")

    if not completed:
        logger.warning(
            "No trial completed, so there is no best trial. Every trial was pruned or "
            "failed -- check the pruner settings and n_trials, and note that the pruner "
            "cannot help until n_startup_trials trials have finished."
        )
        return

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best val_iou: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()

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
import sys as _sys
from pathlib import Path as _Path

import optuna
import torch

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from common.logging_setup import log_run_separator, setup_logger
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
    trial_args.lr = trial.suggest_float("lr", 5e-5, 1.2e-4, log=True)
    # Inherited from base_args, not pinned here. A hardcoded 8 silently
    # discarded whatever --batch-size the caller passed, which is how the
    # earlier batch-size mismatch between train.py and tune.py went unnoticed:
    # the value that actually ran was neither the CLI's nor train.py's default.
    trial_args.batch_size = base_args.batch_size
    trial_args.weight_decay = trial.suggest_float("weight_decay", 5e-5, 3e-3, log=True)
    trial_args.dropout = trial.suggest_float("dropout", 0.3, 0.5)

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

        def report_callback(epoch, val_iou):
            trial.report(val_iou, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch} (val_iou={val_iou:.4f})")
                # `from None`: OOM is the expected outcome being handled, not an
            # error in the handler, so the CUDA traceback is noise in the log.
            raise optuna.TrialPruned() from None

        try:
            val_iou = run_training(trial_args, report_callback=report_callback)
        except torch.cuda.OutOfMemoryError:
            logger.info(
                f"Trial {trial.number} hit CUDA OOM at batch_size={trial_args.batch_size} "
                f"-- pruning this trial, freeing GPU memory, continuing search."
            )
            torch.cuda.empty_cache()
            # `from None`: OOM is the expected outcome being handled, not an
            # error in the handler, so the CUDA traceback is noise in the log.
            raise optuna.TrialPruned() from None

        logger.info(f"Trial {trial.number} finished: val_iou={val_iou:.4f}")
        return val_iou

    return objective


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

    study.optimize(
        make_objective(args),
        n_trials=args.n_trials,
        catch=(torch.cuda.OutOfMemoryError,),
    )

    logger.info(f"Search complete. Best trial: {study.best_trial.number}")
    logger.info(f"Best val_iou: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()

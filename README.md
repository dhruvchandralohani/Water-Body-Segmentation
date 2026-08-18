# Water Body Segmentation

Segmentation of water bodies in satellite imagery, built as a reproducible DVC
pipeline: data audit, training with MLflow tracking, Optuna search, held-out
evaluation, and a Dockerised FastAPI inference service.

Every number the project reports is produced by a pipeline stage and written to
a versioned artifact. Nothing is derived by hand.

## Project structure

```
data_pipeline/   manifest building, dataset audit, patch/tile datasets, transforms
training/        model, loss, metrics, training loop, Optuna search, evaluation
deployment/      model export, FastAPI service, Dockerfile, static UI
common/          shared logging and MLflow helpers
dvc.yaml         the pipeline DAG
params.yaml      every tunable value, and the single source of truth for them
```

## Setup

Python 3.10+ (3.12 in the Dockerfile). A CUDA GPU is recommended for training;
inference runs on CPU.

```bash
pip install -r requirements.txt
```

Put the dataset where `params.yaml:paths` expects it, then let DVC track it:

```bash
dvc init
dvc add data/raw/images data/raw/masks
```

Copy the data into the repo rather than pointing at another drive. An external
path cannot be hashed or restored, which is most of what `dvc.lock` is for.

## Running the pipeline

```bash
dvc repro evaluate      # audit -> train -> evaluate
dvc repro export        # freeze the best run into deployment/exported_model
dvc dag                 # show the graph
dvc metrics show        # class balance, sampling balance, test metrics
```

Name a target rather than running a bare `dvc repro`. The experiment stages
below are frozen and would fail output verification before they have ever run.

### Pipeline stages

| stage | what it produces |
| --- | --- |
| `build_manifest` | `filename,width,height` for every source image |
| `audit` | corrected masks, train/val/test splits, `metrics/class_balance.json` |
| `sampling_balance` | `metrics/sampling_balance.json` -- effective foreground balance per `fg_bias_ratio` |
| `train` | checkpoints, MLflow run, registered model version |
| `evaluate` | `metrics/test_metrics.json`, prediction preview |
| `export` | standalone artifact the Dockerfile copies |

The audit runs four passes, cheapest filter first: size, all-foreground masks,
no-data mask correction, then the stratified split. The all-foreground pass
must precede correction -- correction repairs the black border of a
white-painted mask and hides the defect.

### Experiment stages (frozen)

`tune`, `benchmark`, `capacity`, `lossablation` and `batching` each cost hours
of GPU time and all depend on `training/train.py`, so a one-line edit to that
file would otherwise queue every one of them on the next `dvc repro`.
`frozen: true` prevents that.

To run one, delete the `frozen: true` line from its `do:` block, run it, and put
the line back. Editing the file is more reliable than `dvc freeze`/`dvc unfreeze`,
which do not expand a `foreach` group name. Note that freezing means "already
done", not "skip this": a frozen stage that has never run fails with
`missing data 'source'`.

| stage | question it answers |
| --- | --- |
| `tune` | Optuna search over lr, weight decay, dropout |
| `benchmark` | Does the ASPP dilation recalibration hold? Does a different architecture do better on thin features? |
| `capacity` | Is high dropout the right response to overfitting, versus freezing the encoder or shrinking the decoder? |
| `lossablation` | BCE+Dice against Tversky recall bias, weighted BCE, and focal loss |
| `batching` | Native batch 16 against gradient accumulation to the same effective batch -- isolates BatchNorm batch size |

## Configuration

`params.yaml` holds every value. `dvc.yaml` interpolates it into stage commands,
and `training/train.py` reads the same file for its argparse defaults, so a
hand-run cannot silently diverge from a pipeline run.

There is deliberately **no DVC remote**. The source dataset is public and every
downstream artifact is regenerable by `dvc repro`, so a remote would mostly be
ceremony -- `dvc.lock` still records content hashes, so a change to the raw data
still invalidates the pipeline. `dvc add` pins the dataset version in git; the
bytes are re-obtainable from the original source.

Two things live outside DVC's output tracking on purpose:

- `training/mlflow.db` and `mlruns/` -- DVC deletes a stage's outputs before
  running it, which would erase experiment history on every `dvc repro`.
- `metrics/*.json` are declared `cache: false`, so git versions them and
  `dvc metrics diff` works across commits.

## Serving

```bash
docker build -f deployment/Dockerfile -t water-seg .
docker run -p 8000:8000 water-seg
```

`GET /` serves the UI, `GET /health` reports model metadata including which
model version is live, `POST /predict` accepts an image and returns a mask
(`?format=png` for raw PNG bytes instead of the base64 wrapper), and
`GET /drift` compares recently served predictions against the training-time
water fraction in `metrics/class_balance.json`.

Every served prediction appends one line to `logs/predictions.jsonl`: input
dimensions, mean intensity, predicted water fraction, and duration. Summary
statistics only -- no image data, nothing reconstructable. A sustained gap
between the predicted water fraction and the training reference is the drift
signal; `/drift` reports it as a signed delta rather than a verdict, since what
counts as drift depends on deployment context.

### Promotion

`export_model.py --from-alias` resolves the version aliased `@production`
rather than searching for the best run, so what ships is what someone chose:

```bash
python -m deployment.promote_model --show   # what is live
python -m deployment.promote_model --best   # promote the best evaluated run
```

Two gates. A run with no `test_iou` cannot be promoted -- `val_iou` is the
signal used to *select* a model, so an unevaluated run has been chosen but never
scored on held-out data. And a candidate below the incumbent is refused without
`--force`, which exists because rollback is a legitimate reason to promote a
lower-scoring version.

The image still bakes in a fixed bundle, so a registry outage cannot stop the
service starting.

## Housekeeping

```powershell
.\scripts\clean_artifacts.ps1              # preview what a reset would remove
.\scripts\clean_artifacts.ps1 -Execute -CollectCache
```

Dry run by default, refuses to run outside the repo root, and never touches
`data/raw`. Do not reach for `git clean -xdf` instead -- DVC gitignores its
outputs, so that takes the raw data with it.

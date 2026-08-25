# Runbook

Every command needed to take this project from a fresh clone to a monitored
deployment, in order, with what each one actually does.

PowerShell syntax throughout, since that is where it was developed. On a POSIX
shell only the cleanup script and the environment-variable lines differ.

---

## 0. Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Installs training, evaluation and test dependencies. `deployment/requirements.txt`
is a separate, smaller set installed inside the Docker image — the serving
container has no need for torch's training path, Optuna or DVC.

The raw dataset is pinned by DVC but has no remote (a deliberate choice: the
source is public and everything downstream is regenerable). Place the Kaggle
Water Bodies dataset at `data/raw/images` and `data/raw/masks`, then:

```powershell
dvc status data/raw/images.dvc data/raw/masks.dvc
```

Reports whether the files on disk match the committed pins. `not in cache`
means the workspace files are fine but DVC's own copy is missing — restore it
with `dvc commit data/raw/images.dvc data/raw/masks.dvc`.

---

## 1. Data pipeline

```powershell
dvc repro sampling_balance
```

Runs three stages in dependency order.

**`build_manifest`** scans `data/raw/images` for `.jpg`/`.jpeg` and records each
file's dimensions. Raises if it finds nothing, rather than writing an empty CSV
and failing three stages later.

**`audit`** applies the exclusion rules in four passes and writes the splits.
It removes images smaller than 32px, masks that are entirely foreground, and
images whose masks are mostly no-data — the all-foreground check deliberately
runs *before* no-data correction, because correction repairs the black border
and destroys the evidence. Then it stratifies by image-size bucket into
train/val/test and writes `metrics/class_balance.json`.

Expect **2841 in, 2698 kept, 143 excluded, splits 1888/405/405**, pixel-level
water fraction **0.1843**. These stages are deterministic, so a rerun that
returns different numbers means something changed underneath.

**`sampling_balance`** measures what `fg_bias_ratio` actually does: it draws
patches at a range of settings and reports the resulting water fraction and
empty-patch share. No GPU, no training.

---

## 2. Training (Optuna study)

Tuning *is* training here. Each trial trains to convergence, the pruner kills
unpromising ones early, and `find_best_run` selects the best completed trial.
There is no separate fixed-configuration training stage.

```powershell
dvc repro -f -s tune
```

`-s` confines the run to this stage so the cached data stages are not
re-executed; `-f` forces it, because a study's declared inputs do not change
between runs and DVC would otherwise consider it up to date. Optuna resumes
from `training/optuna_study.db`, so trial numbers continue across invocations.

The stage is frozen in `dvc.yaml`, which is why `-f -s` is needed and why
`dvc repro evaluate` will not silently launch a multi-hour search.

To train one specific configuration instead of sampling, set
`tune.enqueue_params` in `params.yaml`:

```yaml
enqueue_params: '{"lr": 4.23e-04, "weight_decay": 1.69e-03, "dropout": 0.039}'
```

Optuna runs exactly those values as the next trial. Enqueued trials are never
pruned — a configuration you chose should not be cut off against a median it
was not competing in. Set it back to `""` to resume sampling.

To tune a different architecture, change three fields together:

```yaml
tune:
  arch: segformer
  encoder_name: mit_b0        # required: smp defaults Segformer to a resnet34
  study_name: water_body_tuning_segformer_mitb0
```

Studies share one storage file keyed by name, so earlier results survive.
Dropout is searched only where the architecture has ASPP; for U-Net and
SegFormer it is pinned to 0 rather than wasting trials on a parameter the model
never receives.

Watch progress:

```powershell
mlflow ui --backend-store-uri sqlite:///training/mlflow.db
```

---

## 3. Evaluation

```powershell
dvc repro evaluate
```

Selects the best **completed** run by `val_iou` — pruned trials are excluded,
since their peak is a partial result and would otherwise outrank a run that
trained to convergence — then scores it on the 405-image test split, tiled at
full resolution.

Writes `metrics/test_metrics.json` (global IoU, Dice, precision, recall,
accuracy, plus per-image IoU min/mean/max) and a qualitative preview PNG. This
is the only point at which the model touches test data.

Read three things, not just the IoU: the precision/recall gap (a positive gap
means the model under-predicts water), whether test exceeds val (expected —
the val split is less watery than train), and where the val curve peaked
relative to the epoch budget.

---

## 4. Promotion and export

```powershell
python -m deployment.promote_model --show
python -m deployment.promote_model --best
dvc repro export
```

**`--show`** reports which version currently holds the `production` alias.

**`--best`** moves the alias onto the best evaluated run, subject to two gates.
A run with no `test_iou` cannot be promoted — `val_iou` is the signal used to
*select* a model, so an unevaluated run has been chosen but never scored on
held-out data. And a candidate scoring below the incumbent is refused unless
`--force`, which exists because rollback is a legitimate reason to promote a
lower-scoring version.

**`export`** resolves `models:/water_body_segmentation_model@production` and
writes the bundle to `deployment/exported_model/best_model` — the path the
Dockerfile copies and `serve.py` defaults to. It **fails until something is
promoted**, which is intended: a deployment artifact should not exist before
the decision to deploy does.

---

## 5. Container

```powershell
docker build -f deployment/Dockerfile -t water-body-inference:onnx .
docker run -p 8000:8000 water-body-inference:onnx
```

The `onnx` tag distinguishes this from the older torch-based image locally.
Note that it is mutable in the same way `latest` is -- a deployment referencing
it has no record of which build it ran. The CI publish job tags by commit SHA
for that reason, and a real deployment should pin to a digest rather than either.

Builds a self-contained image: model bundle, `class_balance.json` reference,
and serving code. No tracking server or registry is needed at runtime, so a
registry outage cannot stop the service starting.

**The image ships onnxruntime, not torch.** `predict_image` is numpy-native and
both backends adapt to numpy, so nothing in the request path needs a tensor
library. To serve a PyTorch bundle instead, restore the torch install line in
the Dockerfile, add torch and `segmentation-models-pytorch` to
`deployment/requirements.txt`, and set `MODEL_BACKEND=pytorch` -- the
application code is identical either way.

`MODEL_BACKEND` defaults to `auto`, which prefers an ONNX graph when the bundle
carries one. Set it explicitly when comparing backends: asking for one and
silently getting the other would show up only as a latency difference.

Confirm which runtime is actually serving:

```powershell
curl.exe http://localhost:8000/metrics | Select-String model_info
```

The `backend` label is the answer, and the latency histogram carries the same
label -- so a Grafana panel splits by runtime and shows the difference rather
than asserting it.

```powershell
curl.exe http://localhost:8000/health
```

---

## 6. Kubernetes

```powershell
kind create cluster --name water-body
kind load docker-image water-body-inference:onnx --name water-body
kubectl create configmap grafana-dashboards --from-file=deployment/k8s/dashboards/
kubectl apply -f deployment/k8s/
kubectl rollout status deployment/water-body-inference
```

`kind load` copies the local image into the cluster's node, which has its own
image store. The ConfigMap must be created before `apply` — Grafana mounts it
by name and stays in `ContainerCreating` until it exists.

`apply -f` on the directory brings up the inference Deployment and Service,
Prometheus (with RBAC for pod discovery) and Grafana. The dashboard JSON lives
in `dashboards/` specifically so `apply -f` does not try to parse it as a
manifest.

If `kind` is unavailable, sideload manually:

```powershell
docker save water-body-inference:onnx -o wbi.tar
docker cp wbi.tar water-body-control-plane:/wbi.tar
docker exec water-body-control-plane ctr --namespace k8s.io images import /wbi.tar
```

---

## 7. Monitoring

```powershell
kubectl port-forward svc/water-body-inference 8000:8000
kubectl port-forward svc/prometheus 9090:9090
kubectl port-forward svc/grafana 3001:3000
```

Ports on the left are local and can be changed — useful on Windows, where
Hyper-V reserves ranges that make some ports unbindable.

Generate traffic first, or empty panels are indistinguishable from broken ones:

```powershell
Get-ChildItem data\raw\images\*.jpg | Select-Object -First 10 | ForEach-Object {
  curl.exe -s -X POST -F "file=@$($_.FullName)" http://localhost:8000/predict
}
```

Then check, in order: `localhost:9090/targets` shows the inference pod UP;
`localhost:9090/rules` lists one recording rule and four alerts;
`localhost:3001` renders the dashboard with the served water fraction plotted
against the training reference.

The signal that matters is `water_body_predicted_water_fraction` against
`water_body_training_water_fraction`. A sustained gap means the incoming
imagery is unlike what the model trained on — there are no labels at serving
time, so this is the available evidence that something changed.

To prove the alert fires rather than assume it: temporarily lower
`PredictedWaterFractionDrift` to `for: 1m` and `> 0.05` in
`deployment/k8s/prometheus.yaml`, re-apply, restart Prometheus, and watch
`localhost:9090/alerts` go Pending then Firing. Revert afterwards.

---

## 8. Tests and CI

```powershell
python -m pytest -m "not slow"    # ~50s, no GPU
python -m pytest -m slow          # ~2min, full chain on synthetic data
ruff check .
```

The fast suite covers config contracts, the data pipeline, metrics, losses,
transforms, model contracts, promotion gates and the serving API. The slow
suite runs `build_manifest → audit → train → evaluate → export` end to end on
44 synthetic images.

CI runs four jobs on push: **static** (ruff plus `kubeconform -strict` on the
manifests), **tests** (pyright, then both suites, then builds a minimal
servable bundle as an artifact), **docker** (builds the image and verifies
`/health` answers), and **publish** (pushes to GHCR, main only, tagged by
commit SHA as well as `latest`).

There is no deploy job. A GitHub runner has no route to a local cluster, and a
kubeconfig would not change that. GitOps — where the cluster pulls from the
repo — is the resolution that works behind NAT.

---

## 9. Cleanup

```powershell
.\scripts\clean_artifacts.ps1                      # dry run, shows sizes
.\scripts\clean_artifacts.ps1 -Execute -CollectCache
```

Removes generated artifacts: `dvc.lock`, processed data, splits, metrics,
checkpoints, the MLflow database and `mlruns`, the Optuna study, logs and the
image tarball. Keeps source data, the `.dvc` pins, and all code.

Two things to know before running it. MLflow's history and the model registry
go, including the `production` alias — `dvc repro export` will fail until you
promote again. And `dvc gc --workspace` also collects the raw-data cache
despite the `.dvc` pins referencing it, so run `dvc commit data/raw/*.dvc`
afterwards to restore it.

A fresh run reproduces the deterministic stages exactly; training will not.

For the cluster:

```powershell
kubectl delete -f deployment/k8s/
kind delete cluster --name water-body
```
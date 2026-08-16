# Water Body Segmentation

This project trains and deploys a segmentation model for identifying water bodies from aerial imagery. The workflow includes dataset preparation, training, evaluation, MLflow tracking, Optuna-based hyperparameter tuning, model export, and a FastAPI inference service.

## Project structure

- data_pipeline/: dataset manifest generation, mask auditing/correction, tiling, stitching, and data loading
- training/: model definition, training loop, evaluation metrics, MLflow tracking, Optuna tuning, and test-time visualization
- deployment/: model export, FastAPI serving, Docker packaging, and static UI assets
- common/: shared logging and MLflow helpers

## Requirements

- Python 3.10+ (3.12 is used in the provided Dockerfile)
- PyTorch and torchvision installed for your platform
- A CUDA-capable GPU is recommended for training, but CPU is supported for inference

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch is not already installed for your environment, install it separately for your platform before running the project commands below.

## Dataset layout

The training and evaluation scripts expect:

- a manifest CSV with columns `filename`, `width`, and `height`
- image files in an image directory
- mask files in a mask directory
- train/validation/test split CSV files

A typical layout is:

```text
data_pipeline/
  Data/
    manifest.csv
    images/
    masks/
  splits/
    train.csv
    val.csv
    test.csv
```

## 1. Prepare the dataset

If you need to rebuild the manifest from the current image directory:

```bash
python data_pipeline/build_manifest.py --image-dir "data_pipeline/Data/images" --output "data_pipeline/Data/manifest.csv"
```

To audit and correct masks, then create split CSV files:

```bash
python data_pipeline/audit.py \
  --manifest "data_pipeline/Data/manifest.csv" \
  --image-dir "data_pipeline/Data/images" \
  --mask-dir "data_pipeline/Data/masks" \
  --corrected-mask-dir "data_pipeline/Data/masks" \
  --output-dir "data_pipeline/splits"
```

> Adjust the paths to match your local dataset structure if it differs from the workspace layout.

## 2. Train a model

Run training with the provided split files:

```bash
python training/train.py \
  --train-manifest "data_pipeline/splits/train.csv" \
  --val-manifest "data_pipeline/splits/val.csv" \
  --image-dir "data_pipeline/Data/images" \
  --mask-dir "data_pipeline/Data/masks" \
  --checkpoint-dir "training/checkpoints"
```

Training logs and metrics are tracked through MLflow and stored in the local MLflow database under the project.

## 3. Run hyperparameter tuning

To search for better hyperparameters with Optuna:

```bash
python training/tune.py \
  --train-manifest "data_pipeline/splits/train.csv" \
  --val-manifest "data_pipeline/splits/val.csv" \
  --image-dir "data_pipeline/Data/images" \
  --mask-dir "data_pipeline/Data/masks" \
  --epochs 8 \
  --n-trials 20
```

## 4. Evaluate the trained model

Generate a preview of predictions for a handful of test images:

```bash
python training/test_model.py \
  --test-manifest "data_pipeline/splits/test.csv" \
  --image-dir "data_pipeline/Data/images" \
  --mask-dir "data_pipeline/Data/masks" \
  --num-images 4 \
  --output "predictions_preview.png"
```

Use `--run-id` if you want to evaluate a specific MLflow run instead of the best run discovered automatically.

## 5. Export the model

Export the best model bundle to a standalone directory for deployment:

```bash
python deployment/export_model.py --output-dir "deployment/exported_model"
```

## 6. Run the local inference service

Start the FastAPI app from the deployment directory:

```bash
cd deployment
uvicorn serve:app --host 0.0.0.0 --port 8000
```

The service exposes:

- `/health` for status checks
- `/predict` for uploading an image and receiving a segmentation result

You can also open the static UI from the service root.

## 7. Build and run with Docker

From the project root:

```bash
python deployment/export_model.py --output-dir "deployment/exported_model"
docker build -f deployment/Dockerfile -t water-body-inference .
docker run -p 8000:8000 water-body-inference
```

## Notes

- The project uses a local MLflow tracking database for experiment logging.
- The exported model is consumed by the deployment service without needing a live MLflow backend.
- If you change the patch size or data layout, review the training and inference configuration values carefully.

# IMPORTANT -- build context is the PROJECT ROOT, not this deployment/
# folder. This image needs files from data_pipeline/ and common/ too
# (stitch.py, tile_dataset.py, transforms.py, logging_setup.py), and
# Docker's COPY can never reach outside its build context to grab
# sibling-directory files -- there's no way around that, it's a Docker
# constraint, not a choice made here. Build from the project root:
#   cd water_body_segmentation
#   python deployment/export_model.py --output-dir deployment/exported_model
#   docker build -f deployment/Dockerfile -t water-body-inference .
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# No torch. The model is served through onnxruntime, and predict_image was
# rewritten in numpy so nothing in the request path needs a tensor library.
# This line previously installed CPU torch (~800 MB installed) and was the
# single largest thing in the image.
#
# To serve a PyTorch bundle instead, restore this line, add torch and
# segmentation-models-pytorch to deployment/requirements.txt, and set
# MODEL_BACKEND=pytorch. The application code is identical either way.
COPY deployment/requirements.txt ./deployment/requirements.txt

RUN pip install --no-cache-dir -r deployment/requirements.txt

COPY common/__init__.py ./common/__init__.py

COPY common/logging_setup.py ./common/logging_setup.py

COPY data_pipeline/__init__.py ./data_pipeline/__init__.py

# tiling.py, NOT tile_dataset.py: the latter imports torch.utils.data at module
# scope, and this image no longer ships torch. Copying it in would leave a
# module that fails on import the moment anything reaches for it.
COPY data_pipeline/stitch.py data_pipeline/tiling.py data_pipeline/transforms.py ./data_pipeline/

COPY deployment/__init__.py ./deployment/__init__.py

# prediction_log.py is imported by serve.py -- omitting it makes the container
# fail at import time, after a successful build.
COPY deployment/inference.py deployment/serve.py deployment/prediction_log.py \
     deployment/metrics_exporter.py ./deployment/

# The training-time water fraction /drift compares served predictions against.
# Small, versioned in git (cache: false), and without it /drift can report a
# mean but nothing to judge it by.
COPY metrics/class_balance.json ./metrics/class_balance.json

COPY deployment/static ./deployment/static

COPY deployment/exported_model/best_model ./deployment/exported_model/best_model

ENV MODEL_PATH=/app/deployment/exported_model/best_model

ENV DEVICE=cpu

ENV CLASS_BALANCE_PATH=/app/metrics/class_balance.json

ENV PREDICTION_LOG=/app/logs/predictions.jsonl

# Absolute, and matching the volume mount. The final WORKDIR is /app/deployment,
# so a relative "logs" would resolve to /app/deployment/logs -- somewhere the
# mounted volume is not.
ENV LOG_DIR=/app/logs

# Created and owned before dropping privileges. /app is root-owned, so a
# non-root process cannot mkdir inside it at runtime -- which is exactly how
# this failed the first time: PermissionError on 'logs' during import, before
# uvicorn had even started.
RUN mkdir -p /app/logs && chown -R 1000:1000 /app

USER 1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4).read()" || exit 1

WORKDIR /app/deployment

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
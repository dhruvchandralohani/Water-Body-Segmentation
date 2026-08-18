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

RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

COPY deployment/requirements.txt ./deployment/requirements.txt

RUN pip install --no-cache-dir -r deployment/requirements.txt

COPY common/__init__.py ./common/__init__.py

COPY common/logging_setup.py ./common/logging_setup.py

COPY data_pipeline/__init__.py ./data_pipeline/__init__.py

COPY data_pipeline/stitch.py data_pipeline/tile_dataset.py data_pipeline/transforms.py ./data_pipeline/

COPY deployment/__init__.py ./deployment/__init__.py

COPY deployment/inference.py deployment/prediction_log.py deployment/serve.py ./deployment/

COPY deployment/static ./deployment/static

COPY deployment/exported_model/best_model ./deployment/exported_model/best_model

ENV MODEL_PATH=/app/deployment/exported_model/best_model

ENV DEVICE=cpu

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4).read()" || exit 1

WORKDIR /app/deployment

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

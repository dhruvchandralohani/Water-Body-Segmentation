"""
serve.py

FastAPI inference service. Loads the exported model once at startup
(see export_model.py -- this never connects to a live MLflow tracking
store), and exposes /predict and /health. /predict returns the mask
plus full metadata (dimensions, tile count, water coverage, per-stage
timings) rather than just the mask, per the actual response-format
decision made for this project.

Usage:
    uvicorn serve:app --host 0.0.0.0 --port 8000

Environment variables (all optional, sensible defaults for local use):
    MODEL_PATH   -- path to an export_model.py bundle (default: exported_model/best_model)
    DEVICE       -- cuda / cpu (default: cuda if available, else cpu)
    TILE_SIZE, OVERLAP, PATCH_SIZE, THRESHOLD, BATCH_SIZE -- inference config, see inference.py
"""

import base64
import io
import os
import tempfile
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from deployment.inference import load_model, predict_image
from common.logging_setup import setup_logger

logger = setup_logger("serve", log_file="serve.log")

MODEL_PATH = os.environ.get("MODEL_PATH", "exported_model/best_model")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = int(os.environ.get("TILE_SIZE", 384))
OVERLAP = float(os.environ.get("OVERLAP", 0.25))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", 256))
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
USE_FP16 = os.environ.get("USE_FP16", "false").lower() == "true"
STATIC_DIR = _Path(__file__).resolve().parent / "static"

model_state: dict[str, Any] = {"model": None, "device": None, "model_name": "water-body-segmentation", "model_version": "unknown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the segmentation model when the FastAPI app starts up."""
    device = torch.device(DEVICE)
    logger.info(f"Loading model from {MODEL_PATH} onto {device}")
    t0 = time.time()
    model_state["model"] = load_model(MODEL_PATH, device)
    model_state["device"] = device
    try:
        mlmodel_path = _Path(MODEL_PATH) / "MLmodel"
        with open(mlmodel_path) as f:
            mlmodel = yaml.safe_load(f)
        model_state["model_version"] = mlmodel.get("run_id", "unknown")
    except Exception as e:
        logger.info(f"Could not read model version from MLmodel file: {e}")

    logger.info(f"Model loaded in {time.time() - t0:.2f}s, ready to serve")
    yield
    logger.info("Shutting down")
    model_state["model"] = None


app = FastAPI(title="Water Body Segmentation Inference Service", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """Serve the static web UI for the inference service."""
    return FileResponse(str(STATIC_DIR / "index.html"))


class TimingsResponse(BaseModel):
    load: float
    tile: float
    inference: float
    stitch: float
    total: float


class PredictResponse(BaseModel):
    width: int
    height: int
    num_tiles: int
    water_coverage_pct: float
    threshold: float
    device: str
    timings: TimingsResponse
    mask_png_base64: str
    original_png_base64: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str
    model_name: str
    model_version: str
    tile_size: int
    overlap: float
    patch_size: int
    threshold: float


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return the current model loading status and configuration."""
    return HealthResponse(
        status="ok" if model_state["model"] is not None else "model not loaded",
        model_loaded=model_state["model"] is not None,
        device=str(model_state["device"]) if model_state["device"] else "unknown",
        model_path=MODEL_PATH,
        model_name=model_state["model_name"],
        model_version=model_state["model_version"],
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        patch_size=PATCH_SIZE,
        threshold=THRESHOLD,
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    format: str = Query(
        "json",
        description="'json' (default, backward compatible) returns mask_png_base64 plus "
        "metadata. 'png' returns the raw PNG bytes directly, for consumers that don't need "
        "the ~33% base64 overhead or the metadata wrapper.",
    ),
):
    """Run inference on an uploaded image and return a segmentation mask response."""
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if format not in ("json", "png"):
        raise HTTPException(status_code=400, detail=f"format must be 'json' or 'png', got '{format}'")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image file, got content-type {file.content_type}")

    logger.info(f"Request received: filename={file.filename} content_type={file.content_type} format={format}")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "image.jpg").suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        mask, info = predict_image(
            model_state["model"],
            tmp_path,
            model_state["device"],
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            patch_size=PATCH_SIZE,
            threshold=THRESHOLD,
            batch_size=BATCH_SIZE,
            use_fp16=USE_FP16,
        )
    except Exception as e:
        logger.info(f"Request failed: filename={file.filename} error={e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        os.unlink(tmp_path)

    ok, encoded = cv2.imencode(".png", mask * 255)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output mask")
    png_bytes = encoded.tobytes()

    original_arr = np.frombuffer(contents, dtype=np.uint8)
    original_bgr = cv2.imdecode(original_arr, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to decode uploaded image for preview")
    ok, original_encoded = cv2.imencode(".png", original_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode original image preview")
    original_png_bytes = original_encoded.tobytes()

    logger.info(
        f"Request complete: filename={file.filename} "
        f"size={info['width']}x{info['height']} tiles={info['num_tiles']} "
        f"water_coverage={info['water_coverage_pct']:.2f}% "
        f"total_time={info['timings']['total']:.3f}s"
    )

    if format == "png":
        return Response(content=png_bytes, media_type="image/png")

    mask_b64 = base64.b64encode(png_bytes).decode("ascii")
    original_b64 = base64.b64encode(original_png_bytes).decode("ascii")
    return PredictResponse(
        width=info["width"],
        height=info["height"],
        num_tiles=info["num_tiles"],
        water_coverage_pct=info["water_coverage_pct"],
        threshold=THRESHOLD,
        device=str(model_state["device"]),
        timings=TimingsResponse(**info["timings"]),
        mask_png_base64=mask_b64,
        original_png_base64=original_b64,
    )

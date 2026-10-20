"""
FastAPI wrapper around triposr_helpers.py.

Run from python-server-part\ in the `triposer` env:
    uvicorn triposr_server:app --host 0.0.0.0 --port 8000

POST /reconstruct   multipart form field "file" (PNG/JPG)
    optional query params:
        mc_resolution  marching cubes grid (default: MC_RESOLUTION in triposr_helpers.py)
        remove_bg      true/false (default true)
    returns binary glTF (GLB, model/gltf-binary)
GET  /health
"""

import io
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from triposr_helpers import DEVICE, LoadModel, ProcessingTripoSR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("triposr-server")

# Holds the model and background remover once they are loaded
state = {"model": None, "rembg_session": None}

# The GPU handles one reconstruction at a time
gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at server startup
    t = time.time()
    log.info("Loading TripoSR on %s ...", DEVICE)
    state["model"], state["rembg_session"] = LoadModel()
    log.info("Ready after %.1f s", time.time() - t)
    yield
    # (nothing to clean up on shutdown)


app = FastAPI(title="TripoSR server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": state["model"] is not None}


@app.post("/reconstruct")
def reconstruct(
    file: UploadFile = File(...),
    mc_resolution: Optional[int] = Query(None, ge=32, le=320),
    remove_bg: bool = Query(True),
):
    t_start = time.time()

    # Step 1: decode the uploaded bytes into a PIL image
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image (send a real PNG/JPG)")

    # Step 2: build the optional arguments (None means "use the helper's default")
    kwargs = {"remove_bg": remove_bg}
    if mc_resolution is not None:
        kwargs["mc_resolution"] = mc_resolution

    # Step 3: run the reconstruction (one request on the GPU at a time)
    with gpu_lock:
        mesh = ProcessingTripoSR(image, state["model"], state["rembg_session"], **kwargs)

    # Step 4: serialize the mesh to binary glTF (GLB)
    glb_bytes = mesh.export(file_type="glb")

    log.info(
        "reconstruct: %d faces, %d KB, total %.2f s",
        len(mesh.faces), len(glb_bytes) // 1024, time.time() - t_start,
    )
    return Response(content=glb_bytes, media_type="model/gltf-binary")
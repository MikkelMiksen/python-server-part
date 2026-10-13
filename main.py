import sys
import time
from pathlib import Path

import numpy as np
import rembg
import torch
from PIL import Image

# Make the TripoSR submodule importable (folder sits next to this file)
sys.path.insert(0, str(Path(__file__).parent / "TripoSR"))
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FOREGROUND_RATIO = 0.85

# ---- Speed settings (tune these) -------------------------------------------
MC_RESOLUTION = 256        # marching cubes grid: 64 fastest, 128 balanced, 256 = TripoSR default
CHUNK_SIZE = 32768        # bigger chunks = fewer GPU passes (lower this if you run out of VRAM)
USE_FP16 = False           # half precision on the GPU (check that the mesh still looks right)
REMBG_MODEL = "u2net"     # "u2netp" is a smaller, faster (less accurate) background remover
# -----------------------------------------------------------------------------

use_fp16 = USE_FP16 and DEVICE.startswith("cuda")


def LoadModel():
    """Load TripoSR and the background remover ONCE and reuse them for every image."""
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(CHUNK_SIZE)
    model.to(DEVICE)

    rembg_session = rembg.new_session(REMBG_MODEL)

    # Warm-up: the first GPU run is slow (CUDA setup), so do it now on a dummy image
    dummy = Image.new("RGB", (512, 512), (128, 128, 128))
    ProcessingTripoSR(dummy, model, rembg_session, remove_bg=False, mc_resolution=32)
    return model, rembg_session


def LoadInputImage(filepath):
    img = Image.open(filepath)
    return img


def ProcessingTripoSR(image, model, rembg_session, mc_resolution=MC_RESOLUTION, remove_bg=True):
    # Step 1: remove the background (skip if the crop is already cut out - saves ~1.5 s)
    if remove_bg:
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, FOREGROUND_RATIO)
    else:
        image = image.convert("RGBA")

    # Step 2: composite the cut-out onto a grey background (what TripoSR expects)
    pixels = np.array(image).astype(np.float32) / 255.0
    rgb = pixels[:, :, :3]
    alpha = pixels[:, :, 3:4]
    pixels = rgb * alpha + (1 - alpha) * 0.5
    image = Image.fromarray((pixels * 255.0).astype(np.uint8))

    # Step 3 + 4: run the model, then extract the mesh (marching cubes)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
        scene_codes = model([image], device=DEVICE)
        meshes = model.extract_mesh(scene_codes, True, resolution=mc_resolution)

    mesh = meshes[0]
    return mesh


def SaveMesh(mesh, output_path):
    mesh.export(output_path)


if __name__ == "__main__":
    t = time.time()
    model, rembg_session = LoadModel()
    print(f"load + warm-up: {time.time() - t:.2f} s")

    input_image = LoadInputImage(r"C:\Users\mikke\Desktop\Projects\VR Object reconstruction\data_images\testers_from_www\bottle_fixed.png")

    t = time.time()
    mesh = ProcessingTripoSR(input_image, model, rembg_session)
    print(f"reconstruction: {time.time() - t:.2f} s, {len(mesh.faces)} faces")

    SaveMesh(mesh, "output/0/mesh.obj")
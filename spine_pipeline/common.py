"""Shared helpers for the SpineAI auto-rig pipeline."""
import json
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", "original")
PARTS_DIR = os.path.join(ASSETS, "parts")
REFERENCE = os.path.join(ASSETS, "reference", "assembled_character.png")
OUT = os.path.join(ROOT, "output", "spine_character")

ALPHA_NOISE = 16      # see-through leaves faint alpha everywhere; below this is noise
ALPHA_SOLID = 128     # "opaque" for matching / occlusion statistics


def load_rgba(path):
    return np.array(Image.open(path).convert("RGBA"))


def save_rgba(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), "RGBA").save(path)


def clean_alpha(rgba):
    """Zero out the faint alpha haze see-through leaves on every layer."""
    out = rgba.copy()
    out[out[:, :, 3] < ALPHA_NOISE] = 0
    return out


def reference_rgba():
    """Reference CG with its noisy background haze removed (alpha<128 -> 0)."""
    ref = load_rgba(REFERENCE)
    ref = ref.copy()
    ref[ref[:, :, 3] < ALPHA_SOLID] = 0
    return ref


def composite(layers, size, bg=None):
    """Alpha-composite (straight alpha) a list of (rgba_canvas) arrays, back to front."""
    h, w = size
    acc = np.zeros((h, w, 3), np.float32)
    acc_a = np.zeros((h, w), np.float32)
    if bg is not None:
        acc[:] = np.array(bg[:3], np.float32) / 255.0
        acc_a[:] = 1.0
    for layer in layers:
        a = layer[:, :, 3].astype(np.float32) / 255.0
        c = layer[:, :, :3].astype(np.float32) / 255.0
        out_a = a + acc_a * (1 - a)
        safe = np.where(out_a > 1e-6, out_a, 1)
        acc = (c * a[..., None] + acc * acc_a[..., None] * (1 - a[..., None])) / safe[..., None]
        acc_a = out_a
    out = np.zeros((h, w, 4), np.uint8)
    out[:, :, :3] = np.clip(acc * 255 + 0.5, 0, 255)
    out[:, :, 3] = np.clip(acc_a * 255 + 0.5, 0, 255)
    return out


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

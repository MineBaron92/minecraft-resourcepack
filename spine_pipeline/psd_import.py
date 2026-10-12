#!/usr/bin/env python3
"""
Stage 1 for PSD input (See-through PSD with layer offsets + optional depth PSD).

Replaces the search-based assembly of assemble.py: positions come straight from
the PSD.  What is still checked automatically:
  * empty layers are dropped,
  * draw order = PSD order, verified against the original image by pairwise
    occlusion votes and against the depth PSD (nearer = darker),
  * the composite is compared with the original image (metrics + diff image),
  * per-layer mean depth is stored for later use (parallax, ordering).

Writes the same stage1_assembly/ layout that rig.py consumes.

Usage: SPINE_OUT=output/char2 python3 spine_pipeline/psd_import.py PSD [--depth DEPTH_PSD] --ref ORIGINAL.png
"""
import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ALPHA_NOISE, ALPHA_SOLID, OUT, composite, save_rgba, write_json  # noqa: E402
import assemble  # noqa: E402  (hair split + render-error order search)
import cv2  # noqa: E402


def despeckle(rgba):
    """Drop isolated specks (generation haze survivors far from the part)."""
    a = rgba[:, :, 3]
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 24).astype(np.uint8), 8)
    if n <= 1:
        return rgba
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.isin(lab, 1 + np.where(areas >= min(25, 0.2 * areas.max()))[0]).astype(np.uint8)
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    out = rgba.copy()
    out[keep == 0] = 0
    return out

STAGE = os.path.join(OUT, "stage1_assembly")


def over_white(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgba[..., :3].astype(np.float32) * a + 255.0 * (1 - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("psd")
    ap.add_argument("--depth")
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()

    psd = PSDImage.open(args.psd)
    W, H = psd.size
    dpsd = PSDImage.open(args.depth) if args.depth else None
    dlayers = {l.name: l for l in dpsd.descendants()} if dpsd else {}
    ref = np.array(Image.open(args.ref).convert("RGBA"))
    assert ref.shape[:2] == (H, W), f"reference {ref.shape[:2]} != PSD {(H, W)}"

    layers, order, report = {}, [], {"canvas": [W, H], "dropped_empty_layers": [], "depth_mean": {}}
    for l in psd.descendants():                       # psd_tools: bottom -> top
        if l.kind != "pixel":
            continue
        img = np.array(l.composite(viewport=(0, 0, W, H)).convert("RGBA"))
        img[img[:, :, 3] < ALPHA_NOISE] = 0
        img = despeckle(img)
        name = l.name.replace(" ", "_")
        solid = int((img[:, :, 3] > ALPHA_SOLID).sum())
        if solid < 20:
            report["dropped_empty_layers"].append({"layer": l.name, "solid_px": solid})
            continue
        layers[name] = img
        order.append(name)
        if l.name in dlayers:
            d = np.array(dlayers[l.name].composite(viewport=(0, 0, W, H)).convert("L")).astype(np.float32)
            m = img[:, :, 3] > ALPHA_SOLID
            report["depth_mean"][name] = round(float(d[m].mean()), 2)   # 0 = near, 255 = far

    # ---- back hair split + order search on rendered error (same tools as assemble.py)
    assemble.W, assemble.H = W, H
    ref_clean = ref.copy()
    ref_clean[ref_clean[:, :, 3] < ALPHA_SOLID] = 0
    scene = assemble.Scene(ref_clean)
    for n in order:
        scene[n] = layers[n]
    psd_order = list(order)
    if "back_hair" in layers:
        hair = scene.pop("back_hair")
        # compare against EVERYTHING else in the stack (bangs included): where the
        # bangs already explain a hair-coloured reference pixel, back hair must not claim it
        back, front, split_stats = assemble.split_back_hair(scene, hair, [n for n in order if n != "back_hair"])
        report["back_hair_split"] = split_stats
        if split_stats["front_layer_px"] < 50:      # nothing is really in front: keep one layer
            scene["back_hair"] = hair
            report["back_hair_split"]["applied"] = False
        else:
            report["back_hair_split"]["applied"] = True
            scene["back_hair_back"], scene["back_hair_front"] = back, front
        if report["back_hair_split"]["applied"]:
            layers.pop("back_hair")
            layers["back_hair_back"], layers["back_hair_front"] = back, front
            i = order.index("back_hair")
            order[i] = "back_hair_back"
            order.insert(order.index("front_hair") if "front_hair" in order else len(order), "back_hair_front")
    scene.order = list(order)
    report["error_psd_order"] = round(scene.error(), 4)
    moves = []
    order, err = assemble.hill_climb_order(scene, order, log=moves)
    report["order_moves"] = moves
    report["error_final_order"] = round(err, 4)
    report["psd_order"] = psd_order

    # ---- order check 1: occlusion votes against the original image
    refc = ref[:, :, :3].astype(np.float32)
    refa = ref[:, :, 3] > ALPHA_SOLID
    conflicts = []
    solid = {n: layers[n][:, :, 3] > ALPHA_SOLID for n in order}
    covered_above = {}
    acc = np.zeros((H, W), bool)
    for n in reversed(order):                      # pixels covered by layers drawn above n
        covered_above[n] = acc.copy()
        acc |= solid[n]
    for i, a in enumerate(order):
        for b in order[i + 1:]:                        # b is drawn above a
            ma, mb = solid[a], solid[b]
            both = ma & mb & refa & ~covered_above[b]
            if both.sum() < 30:
                continue
            da = np.linalg.norm(refc - layers[a][:, :, :3], axis=2)
            db = np.linalg.norm(refc - layers[b][:, :, :3], axis=2)
            a_wins, b_wins = int((both & (da + 8 < db)).sum()), int((both & (db + 8 < da)).sum())
            if a_wins > 2 * b_wins and a_wins >= 40:
                conflicts.append({"below": a, "above": b, "original_says_below_is_on_top_px": a_wins,
                                  "agree_px": b_wins})
    report["order_conflicts_vs_original"] = conflicts
    # ---- order check 2: depth (a layer drawn above should be nearer where they overlap)
    if dpsd:
        def dname(n):
            return "back hair" if n.startswith("back_hair") else n.replace("_", " ")
        dmaps = {n: np.array(dlayers[dname(n)].composite(viewport=(0, 0, W, H)).convert("L")).astype(np.float32)
                 for n in order if dname(n) in dlayers}
        dconf = []
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                if a not in dmaps or b not in dmaps:
                    continue
                both = (layers[a][:, :, 3] > ALPHA_SOLID) & (layers[b][:, :, 3] > ALPHA_SOLID)
                if both.sum() < 30:
                    continue
                diff = float((dmaps[a][both] - dmaps[b][both]).mean())   # >0: a farther (correct)
                if diff < -3:
                    dconf.append({"below": a, "above": b, "depth_says_below_is_nearer_by": round(-diff, 1),
                                  "overlap_px": int(both.sum())})
        report["order_conflicts_vs_depth"] = dconf

    comp = composite([layers[n] for n in order], (H, W))
    diff = np.abs(over_white(comp) - over_white(ref))
    mse = float((diff ** 2).mean())
    report["metrics_vs_original"] = {
        "mean_abs_err_rgb": round(float(diff.mean()), 3),
        "mean_abs_err_in_silhouette": round(float(diff[refa].mean()), 3),
        "psnr_db": round(10 * np.log10(255 ** 2 / max(mse, 1e-9)), 2),
        "silhouette_iou": round(float((refa & (comp[:, :, 3] > ALPHA_SOLID)).sum() /
                                      max(1, (refa | (comp[:, :, 3] > ALPHA_SOLID)).sum())), 4)}

    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(os.path.join(STAGE, "placed_layers"))
    layout = {"canvas": [W, H], "draw_order": order, "parts": {}}
    for z, n in enumerate(order):
        save_rgba(layers[n], os.path.join(STAGE, "placed_layers", f"{z:02d}__{n}.png"))
        ys, xs = np.where(layers[n][:, :, 3] > 8)
        layout["parts"][n] = {"z": z, "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                              "pose": [0, 0, 1.0, 0.0], "source": "psd", "depth_mean": report["depth_mean"].get(n)}
    save_rgba(comp, os.path.join(STAGE, "final_composite.png"))
    heat = np.clip(diff.mean(axis=2) * 3, 0, 255).astype(np.uint8)
    side = np.concatenate([over_white(ref), np.full((H, 8, 3), 40), over_white(comp),
                           np.full((H, 8, 3), 40), np.dstack([heat] * 3)], 1).astype(np.uint8)
    Image.fromarray(side).save(os.path.join(STAGE, "final_comparison.png"))
    write_json(layout, os.path.join(STAGE, "layout_corrected.json"))
    write_json(report, os.path.join(STAGE, "assembly_report.json"))
    print("order:", order)
    print("metrics:", report["metrics_vs_original"])
    print("conflicts vs original:", conflicts)
    print("conflicts vs depth:", report.get("order_conflicts_vs_depth"))


if __name__ == "__main__":
    main()

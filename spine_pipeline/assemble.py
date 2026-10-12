#!/usr/bin/env python3
"""
Stage 1 — automatic character assembly from see-through layers.

What it does (no hand painting anywhere; every step is measured against the
reference CG `assembled_character.png`):

  1. Load the 29 part PNGs, strip see-through's alpha haze, drop empty layers.
  2. Place parts.  Full-canvas layers start at identity; cropped layers start
     from *constrained* template matching (search windows derived from other,
     already-placed layers, e.g. eyes inside the face, arms beside the torso).
     `layout.json` is only reported for comparison, never trusted.
  3. Split `back_hair` into a behind-body layer and an in-front-of-body layer
     using per-pixel occlusion evidence (which of hair/body matches the
     reference colour).  A single back-hair slot can't be both behind the
     face/torso and in front of the shoulders, which was the original bug.
  4. Refine every placement by local coordinate descent on the *rendered*
     composite error (so occlusion is accounted for).
  5. Find the draw order: occlusion voting -> render-error hill climbing.
  6. Visibility trim: pixels that are visible in the composite although the
     reference shows a lower layer / background there are made transparent.
  7. Write placed layers, layout, metrics and comparison images.

Usage:  python3 spine_pipeline/assemble.py
"""
import math
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ALPHA_SOLID, OUT, PARTS_DIR, clean_alpha, composite,  # noqa: E402
                    load_rgba, reference_rgba, save_rgba, write_json)

H = W = 768
STAGE = os.path.join(OUT, "stage1_assembly")

# semantic prior order, back -> front (only a starting point for the optimiser)
PRIOR = [
    "back_hair_back", "handwear-r", "handwear-l", "ears-r", "ears-l", "neck",
    "face", "eyewhite-r", "eyewhite-l", "irides-r", "irides-l", "eyelash-r",
    "eyelash-l", "eyebrow-r", "eyebrow-l", "nose", "mouth", "legwear",
    "topwear", "bottomwear", "back_hair_front", "front_hair", "headwear",
]
BODY_FOR_HAIR_SPLIT = ["handwear-r", "handwear-l", "ears-r", "ears-l", "neck", "face",
                       "topwear", "bottomwear", "legwear"]
# hard "below < above" constraints the order search may never violate
HARD = [("face", "eyewhite-r"), ("face", "eyewhite-l"), ("eyewhite-r", "irides-r"),
        ("eyewhite-l", "irides-l"), ("irides-r", "eyelash-r"), ("irides-l", "eyelash-l"),
        ("face", "nose"), ("face", "mouth"), ("face", "eyebrow-r"), ("face", "eyebrow-l"),
        ("eyelash-r", "front_hair"), ("eyelash-l", "front_hair"), ("eyebrow-r", "front_hair"),
        ("eyebrow-l", "front_hair"), ("neck", "face"), ("neck", "topwear"),
        ("ears-r", "front_hair"), ("ears-l", "front_hair"), ("back_hair_back", "face"),
        ("back_hair_back", "topwear"), ("front_hair", "headwear")]


def valid_order(order):
    pos = {n: i for i, n in enumerate(order)}
    return all(pos[a] < pos[b] for a, b in HARD if a in pos and b in pos)


# tiny face features must never be trimmed away by the visibility pass
NO_TRIM = {"eyewhite-r", "eyewhite-l", "irides-r", "irides-l", "eyelash-r", "eyelash-l",
           "eyebrow-r", "eyebrow-l", "nose", "mouth", "headwear", "front_hair"}


def short(fname):
    return fname.split("__", 1)[1].rsplit(".", 1)[0].replace(" ", "_")


# ─── geometry ────────────────────────────────────────────────────────────────

def warp(src, x, y, s, rot):
    """Place a cropped RGBA part on the 768 canvas: scale s, rotate rot (deg,
    about the part centre), top-left of the *unrotated scaled* box at (x, y)."""
    h, w = src.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -rot, s)
    M[0, 2] += x + cx * s - cx
    M[1, 2] += y + cy * s - cy
    out = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out, M


def bbox(alpha, thr=ALPHA_SOLID):
    ys, xs = np.where(alpha > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def over_white(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgba[..., :3].astype(np.float32) * a + 255.0 * (1 - a)


class Scene:
    """Layer stack with a cached premultiplied float representation so that
    hundreds of trial renders (placement search, order search) stay cheap."""

    def __init__(self, ref, small=3):
        self.ref = ref
        self.ref_w = over_white(ref)
        self.small = small
        self.ref_w_small = cv2.resize(self.ref_w, (W // small, H // small), interpolation=cv2.INTER_AREA)
        self.layers = {}       # name -> canvas RGBA (placed)
        self.src = {}          # name -> source RGBA (cropped parts)
        self.pose = {}         # name -> [x, y, s, rot]
        self.order = []
        self._pm = {}
        self._pm_small = {}

    def __setitem__(self, name, rgba):
        self.layers[name] = rgba
        self._pm.pop(name, None)
        self._pm_small.pop(name, None)

    def pop(self, name):
        self._pm.pop(name, None)
        self._pm_small.pop(name, None)
        return self.layers.pop(name)

    @staticmethod
    def premult(rgba):
        a = rgba[..., 3].astype(np.float32) / 255.0
        return rgba[..., :3].astype(np.float32) * a[..., None], a

    def pm(self, name, small=False):
        cache = self._pm_small if small else self._pm
        if name not in cache:
            lay = self.layers[name]
            if small:
                lay = cv2.resize(lay, (W // self.small, H // self.small), interpolation=cv2.INTER_AREA)
            cache[name] = self.premult(lay)
        return cache[name]

    def place(self, name, pose):
        self.pose[name] = list(pose)
        self[name], _ = warp(self.src[name], *pose)

    def _stack(self, order, window, override, small):
        x0, y0, x1, y1 = window
        c = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        a = np.zeros((y1 - y0, x1 - x0), np.float32)
        for n in order:
            if override and override[0] == n:
                lc, la = self.premult(override[1][y0:y1, x0:x1])
            else:
                pc, pa = self.pm(n, small)
                lc, la = pc[y0:y1, x0:x1], pa[y0:y1, x0:x1]
            inv = 1.0 - la
            c = lc + c * inv[..., None]
            a = la + a * inv
        return c, a

    def render(self, order=None, window=None, override=None):
        order = order or self.order
        window = window or (0, 0, W, H)
        c, a = self._stack(order, window, override, False)
        out = np.zeros(c.shape[:2] + (4,), np.uint8)
        safe = np.where(a > 1e-6, a, 1.0)
        out[..., :3] = np.clip(c / safe[..., None] + 0.5, 0, 255)
        out[..., 3] = np.clip(a * 255 + 0.5, 0, 255)
        return out

    def render_white(self, order=None, window=None, override=None, small=False):
        order = order or self.order
        if window is None:
            window = (0, 0, W // self.small, H // self.small) if small else (0, 0, W, H)
        c, a = self._stack(order, window, override, small)
        return c + 255.0 * (1.0 - a)[..., None]

    def error(self, order=None, window=None, override=None, small=False):
        refw = self.ref_w_small if small else self.ref_w
        if window is None:
            window = (0, 0, refw.shape[1], refw.shape[0])
        x0, y0, x1, y1 = window
        img = self.render_white(order, window, override, small)
        return float(np.abs(img - refw[y0:y1, x0:x1]).mean())


# ─── initial placement for cropped parts ─────────────────────────────────────

def template_match(ref_rgb, part, scales, region):
    x0, y0, x1, y1 = region
    R = ref_rgb[y0:y1, x0:x1]
    best = None
    for s in scales:
        w, h = max(2, round(part.shape[1] * s)), max(2, round(part.shape[0] * s))
        if w >= R.shape[1] or h >= R.shape[0]:
            continue
        q = cv2.resize(part, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        m = (q[:, :, 3] > ALPHA_SOLID).astype(np.float32)
        if m.sum() < 10:
            continue
        res = cv2.matchTemplate(R, q[:, :, :3].astype(np.float32), cv2.TM_SQDIFF,
                                mask=np.dstack([m] * 3)) / (m.sum() * 3)
        mn, _, loc, _ = cv2.minMaxLoc(res)
        if best is None or mn < best[0]:
            best = (float(mn), float(s), loc[0] + x0, loc[1] + y0)
    return best


def centre_pose(src, cx, cy, s=1.0, rot=0.0):
    h, w = src.shape[:2]
    return [cx - w * s / 2.0, cy - h * s / 2.0, s, rot]


def pose_centre(src, pose):
    h, w = src.shape[:2]
    x, y, s, _ = pose
    return x + w * s / 2.0, y + h * s / 2.0


# ─── local refinement on rendered error ──────────────────────────────────────

def refine(scene, name, steps=(8, 4, 2, 1), s_range=(0.6, 1.4), rot_range=(-25, 25),
           allow_rot=False, margin=24, rounds=3, log=None):
    pose = list(scene.pose[name])
    src = scene.src[name]

    def window_for(p):
        lay, _ = warp(src, *p)
        b = bbox(lay[:, :, 3], 8)
        if b is None:
            return (0, 0, W, H)
        return (max(0, b[0] - margin), max(0, b[1] - margin), min(W, b[2] + margin), min(H, b[3] + margin))

    def cost(p):
        lay, _ = warp(src, *p)
        win = window_for(pose)            # fixed window around current pose (comparable costs)
        b = bbox(lay[:, :, 3], 8)
        if b is not None:                 # union with candidate box
            win = (min(win[0], max(0, b[0] - margin)), min(win[1], max(0, b[1] - margin)),
                   max(win[2], min(W, b[2] + margin)), max(win[3], min(H, b[3] + margin)))
        e_new = scene.error(window=win, override=(name, lay))
        e_old = scene.error(window=win)
        return e_new - e_old              # relative to current -> window-size independent

    start = scene.error(window=window_for(pose))
    for step in steps:
        for _ in range(rounds):
            improved = False
            deltas = [(0, step, 0, 0), (0, -step, 0, 0), (1, step, 0, 0), (1, -step, 0, 0),
                      (2, 0.02 * step, 0, 0), (2, -0.02 * step, 0, 0)]
            if allow_rot:
                deltas += [(3, 1.5 * step, 0, 0), (3, -1.5 * step, 0, 0)]
            for idx, d, _, _ in deltas:
                cand = list(pose)
                cand[idx] += d
                if idx == 2 and not (s_range[0] <= cand[2] <= s_range[1]):
                    continue
                if idx == 3 and not (rot_range[0] <= cand[3] <= rot_range[1]):
                    continue
                if idx == 2:  # scale about centre
                    cx, cy = pose_centre(src, pose)
                    cand = centre_pose(src, cx, cy, cand[2], cand[3])
                c = cost(cand)
                if c < -1e-3:
                    pose = cand
                    scene.place(name, pose)
                    improved = True
            if not improved:
                break
    end = scene.error(window=window_for(pose))
    if log is not None:
        log.append({"part": name, "pose": [round(v, 2) for v in pose],
                    "window_err_before": round(start, 3), "window_err_after": round(end, 3)})
    return pose


# ─── back-hair split ─────────────────────────────────────────────────────────

def split_back_hair(scene, hair, body_names):
    ref = scene.ref[:, :, :3].astype(np.float32)
    body = composite([scene.layers[n] for n in body_names if n in scene.layers], (H, W))
    ha, ba = hair[:, :, 3] > ALPHA_SOLID, body[:, :, 3] > ALPHA_SOLID
    refa = scene.ref[:, :, 3] > 0
    dh = np.linalg.norm(ref - hair[:, :, :3].astype(np.float32), axis=2)
    db = np.linalg.norm(ref - body[:, :, :3].astype(np.float32), axis=2)
    both = ha & ba & refa
    hair_wins = both & (dh + 12 < db)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(hair_wins.astype(np.uint8), cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k9)
    # keep only sizeable connected strands
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 60:
            keep[lab == i] = 1
    # soften edge so the two halves blend seamlessly
    soft = cv2.GaussianBlur(keep.astype(np.float32), (5, 5), 0)
    front = hair.copy()
    back = hair.copy()
    front[:, :, 3] = (hair[:, :, 3] * soft).astype(np.uint8)
    back[:, :, 3] = (hair[:, :, 3] * (1 - soft)).astype(np.uint8)
    stats_out = {"overlap_px": int(both.sum()), "hair_in_front_px": int(hair_wins.sum()),
                 "body_in_front_px": int((both & (db + 12 < dh)).sum()),
                 "front_layer_px": int((front[:, :, 3] > ALPHA_SOLID).sum())}
    return back, front, stats_out


# ─── draw order ──────────────────────────────────────────────────────────────

def occlusion_votes(scene, names):
    ref = scene.ref[:, :, :3].astype(np.float32)
    refa = scene.ref[:, :, 3] > 0
    votes = {}
    for i, a in enumerate(names):
        la = scene.layers[a]
        ma = la[:, :, 3] > ALPHA_SOLID
        if not ma.any():
            continue
        da = np.linalg.norm(ref - la[:, :, :3].astype(np.float32), axis=2)
        for b in names[i + 1:]:
            lb = scene.layers[b]
            mb = lb[:, :, 3] > ALPHA_SOLID
            both = ma & mb & refa
            if both.sum() < 6:
                continue
            db = np.linalg.norm(ref - lb[:, :, :3].astype(np.float32), axis=2)
            aw = int((both & (da + 8 < db)).sum())
            bw = int((both & (db + 8 < da)).sum())
            votes[(a, b)] = (aw, bw, int(both.sum()))
    return votes


def order_from_votes(names, votes, prior_rank, min_wins=20, ratio=2.0):
    """Topological sort of strong 'X is above Y' evidence; the semantic prior
    breaks ties and orders unrelated layers. Cycles drop their weakest edge."""
    edges = {}  # (below, above) -> weight
    for (a, b), (aw, bw, _) in votes.items():
        if aw >= min_wins and aw > ratio * bw:
            edges[(b, a)] = aw
        elif bw >= min_wins and bw > ratio * aw:
            edges[(a, b)] = bw
    while True:
        indeg = {n: 0 for n in names}
        for (lo, hi) in edges:
            indeg[hi] += 1
        out, ready = [], sorted([n for n in names if indeg[n] == 0], key=prior_rank.get)
        rem = dict(edges)
        while ready:
            n = ready.pop(0)
            out.append(n)
            for (lo, hi) in list(rem):
                if lo == n:
                    del rem[(lo, hi)]
                    indeg[hi] -= 1
                    if indeg[hi] == 0:
                        ready.append(hi)
            ready.sort(key=prior_rank.get)
        if len(out) == len(names):
            if valid_order(out):
                return out
            # evidence contradicts a hard constraint: fall back to the prior
            return sorted(names, key=prior_rank.get)
        # cycle: drop weakest remaining edge and retry
        weakest = min(rem, key=rem.get)
        del edges[weakest]


def hill_climb_order(scene, order, max_sweeps=4, log=None):
    best = list(order)
    best_e = scene.error(order=best, small=True)
    for sweep in range(max_sweeps):
        improved = False
        for name in list(best):
            i = best.index(name)
            for j in range(len(best)):
                if j == i:
                    continue
                cand = best[:i] + best[i + 1:]
                cand.insert(j, name)
                if not valid_order(cand):
                    continue
                e = scene.error(order=cand, small=True)
                if e < best_e - 1e-4:
                    if log is not None:
                        log.append({"sweep": sweep, "move": name, "from": i, "to": j,
                                    "err": round(e, 4)})
                    best, best_e, improved = cand, e, True
                    break
        if not improved:
            break
    return best, scene.error(order=best)


# ─── visibility trim ─────────────────────────────────────────────────────────

def visibility_trim(scene, iters=3, thr=45.0):
    """Remove part pixels that are visible in the composite where the
    reference clearly shows whatever lies underneath (a lower layer or the
    background). Purely subtractive: nothing is painted."""
    refw = scene.ref_w
    report = {}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(iters):
        changed = 0
        order = scene.order
        # per-pixel index of the top-most solid layer
        top = np.full((H, W), -1, np.int32)
        for idx, n in enumerate(order):
            top[scene.layers[n][:, :, 3] > ALPHA_SOLID] = idx
        for idx, n in enumerate(order):
            if n in NO_TRIM:
                continue
            mine = top == idx
            if not mine.any():
                continue
            with_me = scene.render_white()
            without = scene.render_white(order=order[:idx] + order[idx + 1:])
            e_with = np.abs(with_me - refw).mean(axis=2)
            e_without = np.abs(without - refw).mean(axis=2)
            bad = mine & (e_with > thr) & (e_without + 15 < e_with)
            bad = cv2.morphologyEx(bad.astype(np.uint8), cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            bad = cv2.morphologyEx(bad, cv2.MORPH_CLOSE, k).astype(bool) & mine
            if bad.sum() < 20:
                continue
            lay = scene.layers[n].copy()
            lay[bad, 3] = 0
            scene[n] = lay
            report[n] = report.get(n, 0) + int(bad.sum())
            changed += int(bad.sum())
        if changed == 0:
            break
    return report


# ─── metrics / debug images ──────────────────────────────────────────────────

def metrics(scene, comp=None):
    comp = scene.render() if comp is None else comp
    refa = scene.ref[:, :, 3] > 0
    ca = comp[:, :, 3] > ALPHA_SOLID
    diff = np.abs(over_white(comp) - scene.ref_w)
    mse = float((diff ** 2).mean())
    return {
        "mean_abs_err_rgb": round(float(diff.mean()), 3),
        "mean_abs_err_in_silhouette": round(float(diff[refa].mean()), 3),
        "psnr_db": round(10 * math.log10(255 ** 2 / max(mse, 1e-9)), 2),
        "silhouette_iou": round(float((refa & ca).sum() / max(1, (refa | ca).sum())), 4),
        "pixels_err_gt_40": round(float((diff.mean(axis=2) > 40)[refa | ca].mean()), 4),
    }


def save_debug(scene, comp, path_prefix):
    ref = scene.ref_w.astype(np.uint8)
    cw = over_white(comp).astype(np.uint8)
    diff = np.abs(cw.astype(np.int16) - ref.astype(np.int16)).mean(axis=2)
    heat = cv2.applyColorMap(np.clip(diff * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    gap = np.full((H, 8, 3), 40, np.uint8)
    side = np.concatenate([ref, gap, cw, gap, heat], axis=1)
    cv2.imwrite(path_prefix + "_comparison.png", side[:, :, ::-1])
    save_rgba(comp, path_prefix + "_composite.png")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(STAGE, exist_ok=True)
    ref = reference_rgba()
    scene = Scene(ref)
    ref_rgb = ref[:, :, :3].astype(np.float32)
    report = {"dropped_empty_layers": [], "initial": {}, "refine": [], "order_moves": []}

    files = sorted(f for f in os.listdir(PARTS_DIR) if f.endswith(".png"))
    for f in files:
        name = short(f)
        img = clean_alpha(load_rgba(os.path.join(PARTS_DIR, f)))
        solid = int((img[:, :, 3] > ALPHA_SOLID).sum())
        if solid < 20:
            report["dropped_empty_layers"].append({"file": f, "solid_px": solid})
            continue
        scene.src[name] = img
        if img.shape[:2] == (H, W):
            scene.place(name, [0, 0, 1.0, 0.0])

    # ---- 1. initial placement of cropped parts (constrained matching) -----
    face_b = bbox(scene.layers["face"][:, :, 3])
    top_b = bbox(scene.layers["topwear"][:, :, 3])
    fx0, fy0, fx1, fy1 = face_b
    fmid = (fx0 + fx1) // 2
    eye_band = (fy0 + int(0.35 * (fy1 - fy0)), fy0 + int(0.85 * (fy1 - fy0)))
    side_reg = {"-r": (fx0 - 10, eye_band[0], fmid + 5, eye_band[1]),   # character right = viewer left
                "-l": (fmid - 5, eye_band[0] - 15, fx1 + 25, eye_band[1])}
    scales_fine = np.round(np.arange(0.70, 1.16, 0.02), 3)
    for side in ("-r", "-l"):
        lash = "eyelash" + side
        m = template_match(ref_rgb, scene.src[lash], scales_fine, side_reg[side])
        scene.place(lash, [m[2], m[3], m[1], 0.0])
        report["initial"][lash] = {"method": "template(face-band)", "rmse": round(math.sqrt(m[0]), 2),
                                   "region": side_reg[side]}
        cx, cy = pose_centre(scene.src[lash], scene.pose[lash])
        s = m[1]
        # eyewhite / iris hang under the lash line, brows above it
        scene.place("eyewhite" + side, centre_pose(scene.src["eyewhite" + side], cx, cy + 5 * s, s))
        ewx, ewy = pose_centre(scene.src["eyewhite" + side], scene.pose["eyewhite" + side])
        reg = (int(cx - 40), int(cy - 20), int(cx + 40), int(cy + 30))
        mi = template_match(ref_rgb, scene.src["irides" + side], [s * k for k in (0.9, 0.95, 1.0, 1.05, 1.1)], reg)
        scene.place("irides" + side, [mi[2], mi[3], mi[1], 0.0])
        report["initial"]["irides" + side] = {"method": "template(near lash)", "rmse": round(math.sqrt(mi[0]), 2)}
        scene.place("eyebrow" + side, centre_pose(scene.src["eyebrow" + side], cx, cy - 22 * s, s))
        report["initial"]["eyewhite" + side] = {"method": "relative to eyelash"}
        report["initial"]["eyebrow" + side] = {"method": "relative to eyelash (hidden under bangs)"}
    # ears: beside the face at eye height (mostly hidden under hair)
    ey = sum(pose_centre(scene.src["eyelash" + s_], scene.pose["eyelash" + s_])[1] for s_ in ("-r", "-l")) / 2
    es = scene.pose["eyelash-r"][2]
    scene.place("ears-r", centre_pose(scene.src["ears-r"], fx0 + 4, ey + 22, es))
    scene.place("ears-l", centre_pose(scene.src["ears-l"], fx1 - 4, ey + 22, es))
    report["initial"]["ears"] = {"method": "face silhouette sides at eye height"}
    # arms: hang from the shoulders, beside the torso
    tx0, ty0, tx1, ty1 = top_b
    for side, ax in (("-r", tx0 - 5), ("-l", tx1 + 5)):
        n = "handwear" + side
        src = scene.src[n]
        upper = src[: int(src.shape[0] * 0.42)]          # forearm is bent behind the back in the CG
        reg = (max(0, ax - 110), ty0 - 20, min(W, ax + 110), ty0 + 320)
        best = None
        for rot in range(-14, 15, 4):
            hh, ww = upper.shape[:2]
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), -rot, 1.0)
            ru = cv2.warpAffine(upper, M, (ww, hh))
            m = template_match(ref_rgb, ru, [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], reg)
            if m and (best is None or m[0] < best[0][0]):
                best = (m, rot)
        (err, sc, mx, my), rot = best
        # convert the upper-crop placement to a pose of the full part (rotation about part centre)
        hh, ww = upper.shape[:2]
        ucx, ucy = mx + ww * sc / 2, my + hh * sc / 2              # upper-crop centre on canvas
        H_, W_ = src.shape[:2]
        off = np.array([0, (H_ - hh) / 2 * sc])                    # full centre relative to upper centre
        th = math.radians(rot)
        off = np.array([off[0] * math.cos(th) - off[1] * math.sin(th),
                        off[0] * math.sin(th) + off[1] * math.cos(th)])
        scene.place(n, centre_pose(src, ucx + off[0], ucy + off[1], sc, rot))
        report["initial"][n] = {"method": "template(upper arm, beside torso, +-14deg)",
                                "rmse": round(math.sqrt(err), 2), "rot": rot}

    # ---- 2. split back hair -----------------------------------------------
    hair = scene.pop("back_hair")
    scene.src.pop("back_hair")
    back, front, split_stats = split_back_hair(scene, hair, BODY_FOR_HAIR_SPLIT)
    scene["back_hair_back"], scene["back_hair_front"] = back, front
    report["back_hair_split"] = split_stats

    names = [n for n in PRIOR if n in scene.layers]
    extra = [n for n in scene.layers if n not in names]
    names += extra
    prior_rank = {n: i for i, n in enumerate(names)}
    scene.order = list(names)
    report["error_prior_order"] = round(scene.error(), 4)

    # ---- 3. refine placements on rendered error ---------------------------
    cropped = [n for n in scene.order if n in scene.src and scene.src[n].shape[:2] != (H, W)]
    for rnd in range(2):
        for n in cropped:
            is_arm = n.startswith("handwear")
            refine(scene, n, s_range=(0.6, 1.4), allow_rot=is_arm,
                   steps=(8, 4, 2, 1) if is_arm or n.startswith("ears") else (6, 3, 2, 1),
                   margin=30 if is_arm else 14, log=report["refine"])
    # full-canvas layers: verify identity with a +-3px nudge search
    for n in [n for n in scene.order if n in scene.src and scene.src[n].shape[:2] == (H, W)]:
        refine(scene, n, steps=(2, 1), s_range=(0.98, 1.02), margin=4, rounds=2, log=report["refine"])

    # ---- 4. draw order ------------------------------------------------------
    votes = occlusion_votes(scene, scene.order)
    report["occlusion_votes"] = {f"{a}|{b}": v for (a, b), v in votes.items() if v[2] >= 30}
    voted = order_from_votes(scene.order, votes, prior_rank)
    scene.order, err = hill_climb_order(scene, voted, log=report["order_moves"])
    report["error_after_order"] = round(err, 4)

    # second refinement pass with the final order
    for n in cropped:
        is_arm = n.startswith("handwear")
        refine(scene, n, allow_rot=is_arm, steps=(2, 1), margin=30 if is_arm else 14,
               log=report["refine"])

    comp_before_trim = scene.render()
    report["metrics_before_trim"] = metrics(scene, comp_before_trim)
    save_debug(scene, comp_before_trim, os.path.join(STAGE, "before_trim"))

    # ---- 5. visibility trim --------------------------------------------------
    report["visibility_trim_px"] = visibility_trim(scene)
    comp = scene.render()
    report["metrics_final"] = metrics(scene, comp)
    save_debug(scene, comp, os.path.join(STAGE, "final"))

    # ---- 6. outputs ----------------------------------------------------------
    layer_dir = os.path.join(STAGE, "placed_layers")
    shutil.rmtree(layer_dir, ignore_errors=True)
    os.makedirs(layer_dir, exist_ok=True)
    layout = {"canvas": [W, H], "draw_order": scene.order, "parts": {}}
    for z, n in enumerate(scene.order):
        save_rgba(scene.layers[n], os.path.join(layer_dir, f"{z:02d}__{n}.png"))
        layout["parts"][n] = {"z": z, "bbox": bbox(scene.layers[n][:, :, 3], 8),
                              "pose": [round(v, 2) for v in scene.pose.get(n, [0, 0, 1, 0])],
                              "source": "split of back_hair" if n.startswith("back_hair_") else
                              next(f for f in files if short(f) == n)}
    write_json(layout, os.path.join(STAGE, "layout_corrected.json"))
    write_json(report, os.path.join(STAGE, "assembly_report.json"))
    print("order:", scene.order)
    print("before trim:", report["metrics_before_trim"])
    print("final:", report["metrics_final"])
    print("trim:", report["visibility_trim_px"])


if __name__ == "__main__":
    main()

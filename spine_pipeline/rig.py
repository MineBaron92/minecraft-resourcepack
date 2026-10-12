#!/usr/bin/env python3
"""
Stages 2-3 — automatic rig + idle animation -> Spine 4.2 JSON + atlas.

Input : output/spine_character/stage1_assembly/{layout_corrected.json, placed_layers/}
Output: output/spine_character/spine/{character.json, character.atlas, character.png}

  * Landmarks are measured from the assembled layers (layer bboxes, shirt
    blush clusters -> breast centres, hair-strand centrelines, bang columns,
    arm poses from stage 1).
  * Bones: root/hip/spine/chest(+breath leaf)/neck/head, independent breast
    bones, eye bones (blink), 3 bang bones, 2x4 side-hair chains, 2 back-hair
    bones, shirt-hem cloth bones, upper/lower arms, thighs.
  * Large layers become weighted grid meshes; weights are Gaussian falloffs
    of the distance to each allowed bone segment (top-4, normalised).
  * Idle (8 s loop): breathing (2 breaths), body sway, head counter-motion,
    blinks. Secondary motion (hair, bangs, breasts, shirt) is simulated with
    damped springs driven by the primary motion and baked into keys, so it
    plays in any runtime. Spine 4.2 physics constraints are also emitted;
    `idle_physics` keeps only the primary keys and lets the runtime simulate.

The JSON is assembled with the official skill's build_spine_json.py (curve
conversion to the 4.2 wire format) and the atlas with make_atlas.py.
"""
import glob
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor"))
from common import OUT, load_rgba, save_rgba, write_json  # noqa: E402
import build_spine_json as bsj  # noqa: E402  (official skill script)

H = W = 768
STAGE1 = os.path.join(OUT, "stage1_assembly")
SPINE_DIR = os.path.join(OUT, "spine")
IMG_DIR = os.path.join(OUT, "work", "images")
NAME = "character"
DUR = 8.0          # idle loop length (s)
FPS_BAKE = 30      # key density for baked curves


def to_spine(px, py):
    return float(px - W / 2), float(H - py)


# ─── landmarks ───────────────────────────────────────────────────────────────

def bbox(a, thr=128):
    ys, xs = np.where(a > thr)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def load_layers():
    layout = json.load(open(os.path.join(STAGE1, "layout_corrected.json")))
    layers = {}
    for f in glob.glob(os.path.join(STAGE1, "placed_layers", "*.png")):
        layers[os.path.basename(f).split("__", 1)[1][:-4]] = load_rgba(f)
    return layout, layers


def landmarks(layout, L):
    lm = {}
    T, B, LG = bbox(L["topwear"][:, :, 3]), bbox(L["bottomwear"][:, :, 3]), bbox(L["legwear"][:, :, 3])
    F, N, FH = bbox(L["face"][:, :, 3]), bbox(L["neck"][:, :, 3]), bbox(L["front_hair"][:, :, 3])
    lm.update(topwear=T, bottomwear=B, legwear=LG, face=F, neck=N, front_hair=FH)
    # breast centres: the two largest blush (pink) clusters on the shirt
    t = L["topwear"].astype(int)
    pink = (t[:, :, 3] > 128) & (t[:, :, 0] - t[:, :, 1] > 18) & (t[:, :, 0] > 180)
    n, lab, st, cen = cv2.connectedComponentsWithStats(pink.astype(np.uint8))
    big = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:2]
    cs = sorted([tuple(cen[i]) for i in big])
    lm["breast_r"], lm["breast_l"] = cs  # character right = viewer left
    lm["breast_radius"] = float(np.mean([max(st[i, 2], st[i, 3]) for i in big]) * 0.75)
    # side hair centre lines
    hair = np.maximum(L["back_hair_back"][:, :, 3], L["back_hair_front"][:, :, 3]) > 128
    for side, sel in (("r", lambda xs: xs[xs < F[0] + 5]), ("l", lambda xs: xs[xs > F[2] - 5])):
        rows = [y for y in range(F[1] + 80, H) if len(sel(np.where(hair[y])[0])) > 6]
        y0, y1 = rows[0], rows[-1]
        pts = []
        for y in np.linspace(y0, y1 - 8, 5):
            band = hair[int(y) - 6:int(y) + 6]
            xs = sel(np.where(band.any(0))[0])
            pts.append((float(np.median(xs)), float(y)))
        lm["hair_" + side] = pts
    hb = hair[:, F[0] + 20:F[2] - 20]
    lm["hair_back_bottom"] = int(np.where(hb.any(1))[0].max())
    # bang columns
    fh = L["front_hair"][:, :, 3] > 128
    cols = np.linspace(FH[0], FH[2], 4)
    lm["bangs"] = []
    for a, b in zip(cols[:-1], cols[1:]):
        ys = np.where(fh[:, int(a):int(b)].any(1))[0]
        lm["bangs"].append(((a + b) / 2, float(ys.min() + 20), float(ys.max() - 12)))
    # eyes (centre of lash layers)
    for side in ("r", "l"):
        x0, y0, x1, y1 = bbox(L["eyelash-" + side][:, :, 3], 60)
        lm["eye_" + side] = ((x0 + x1) / 2, (y0 + y1) / 2 + 3)
    # arms from stage-1 poses (source part 75x474 / 90x475, rot about centre)
    for side, src in (("r", (75, 474)), ("l", (90, 475))):
        x, y, s, rot = layout["parts"]["handwear-" + side]["pose"]
        w, h = src
        cx, cy = x + w * s / 2, y + h * s / 2
        th = math.radians(rot)
        def P(u, v):  # point in source coords -> canvas
            dx, dy = (u - w / 2) * s, (v - h / 2) * s
            return (cx + dx * math.cos(th) - dy * math.sin(th), cy + dx * math.sin(th) + dy * math.cos(th))
        lm["arm_" + side] = (P(w / 2, 10), P(w / 2, h * 0.45), P(w / 2, h * 0.97))
    return lm


# ─── skeleton ────────────────────────────────────────────────────────────────

class Bone:
    def __init__(self, name, parent, head_px, tip_px, inherit=None):
        self.name, self.parent, self.inherit = name, parent, inherit
        self.head = np.array(to_spine(*head_px))
        tip = np.array(to_spine(*tip_px))
        d = tip - self.head
        self.length = float(np.hypot(*d))
        self.angle = math.degrees(math.atan2(d[1], d[0]))
        self.tip = tip


def bust_R(lm):
    """Bust radius: the blush cluster marks only the lower front of each breast."""
    return 1.15 * lm["breast_radius"]


def bust_centre(lm, s):
    bx, by = lm["breast_" + s]
    return bx, by - 0.2 * bust_R(lm)


def build_skeleton(lm):
    T, B, LG, F, N = lm["topwear"], lm["bottomwear"], lm["legwear"], lm["face"], lm["neck"]
    cx = (T[0] + T[2]) / 2
    ncx = (N[0] + N[2]) / 2
    hip_y = B[1] + 0.55 * (B[3] - B[1])
    waist_y = T[3] - 30
    chest_y = T[1] + 0.45 * (T[3] - T[1])
    neck_y = N[3] - 15
    head_y = F[3] - 22
    bones = [Bone("root", None, (W / 2, H), (W / 2, H - 1))]
    bones[0].length, bones[0].angle = 0.0, 0.0
    add = lambda *a, **k: bones.append(Bone(*a, **k))
    add("hip", "root", (cx, hip_y), (cx, waist_y))
    add("spine", "hip", (cx, waist_y), (cx, chest_y))
    add("chest", "spine", (cx, chest_y), (ncx, neck_y))
    add("breath", "chest", (cx, chest_y), (cx, chest_y - 60))
    R = bust_R(lm)
    for s in ("r", "l"):
        bx, by = bust_centre(lm, s)
        # root: upper attachment (swings), tip: mass centre (bounces, squashes/stretches)
        add("breast_" + s, "chest", (bx, by - 1.0 * R), (bx, by))
        add("breast_" + s + "_tip", "breast_" + s, (bx, by), (bx, by + 0.8 * R))
    add("neck", "chest", (ncx, neck_y), (ncx, head_y))
    add("head", "neck", (ncx, head_y), (ncx, F[1]))
    for s in ("r", "l"):
        ex, ey = lm["eye_" + s]
        add("eye_" + s, "head", (ex, ey), (ex + 10, ey))
    for i, (bx, y0, y1) in enumerate(lm["bangs"]):
        add(["bang_r", "bang_c", "bang_l"][i], "head", (bx, y0), (bx, y1))
    for s in ("r", "l"):
        pts = lm["hair_" + s]
        parent = "head"
        for i in range(4):
            name = f"hair_{s}{i + 1}"
            add(name, parent, pts[i], pts[i + 1])
            parent = name
    hb0 = (ncx, F[1] + 70)
    hb1 = (ncx, (F[1] + 70 + lm["hair_back_bottom"]) / 2)
    add("hair_b1", "head", hb0, hb1)
    add("hair_b2", "hair_b1", hb1, (ncx, lm["hair_back_bottom"] - 10))
    for s, x in (("r", T[0] + 22), ("l", T[2] - 22)):
        add("shirt_" + s, "spine", (x, T[3] - 70), (x + (-8 if s == "r" else 8), T[3] - 5))
    for s in ("r", "l"):
        a0, a1, a2 = lm["arm_" + s]
        add(f"arm_{s}_up", "chest", a0, a1)
        add(f"arm_{s}_low", f"arm_{s}_up", a1, a2)
    lw = LG[2] - LG[0]
    for s, x in (("r", LG[0] + 0.27 * lw), ("l", LG[2] - 0.27 * lw)):
        add("thigh_" + s, "hip", (x, LG[1] + 15), (x, H - 5))
    return bones


def bone_json(bones):
    by = {b.name: b for b in bones}
    out = []
    for b in bones:
        j = {"name": b.name}
        if b.parent:
            p = by[b.parent]
            th = math.radians(-p.angle)
            d = b.head - p.head
            j["parent"] = b.parent
            j["x"] = round(d[0] * math.cos(th) - d[1] * math.sin(th), 2)
            j["y"] = round(d[0] * math.sin(th) + d[1] * math.cos(th), 2)
            rot = b.angle - p.angle
            rot = (rot + 180) % 360 - 180
            j["rotation"] = round(rot, 2)
        j["length"] = round(b.length, 2)
        if b.inherit:
            j["inherit"] = b.inherit
        out.append(j)
    return out


# ─── attachments & weights ───────────────────────────────────────────────────

# slot -> allowed bones with (strength, sigma); sigma None -> from bone length
WEIGHTS = {
    "back_hair_back": {"head": (1.0, 70), "hair_r1": (1, 30), "hair_r2": (1, 32), "hair_r3": (1, 34),
                       "hair_r4": (1, 36), "hair_l1": (1, 30), "hair_l2": (1, 32), "hair_l3": (1, 34),
                       "hair_l4": (1, 36), "hair_b1": (0.8, 45), "hair_b2": (0.8, 50)},
    "back_hair_front": {"head": (1.0, 70), "hair_r1": (1, 30), "hair_r2": (1, 32), "hair_r3": (1, 34),
                        "hair_r4": (1, 36), "hair_l1": (1, 30), "hair_l2": (1, 32), "hair_l3": (1, 34),
                        "hair_l4": (1, 36)},
    "front_hair": {"head": (1.0, 45), "bang_r": (1, 26), "bang_c": (1, 26), "bang_l": (1, 26)},
    "topwear": {"spine": (1.0, 45), "breath": (1.3, 110),
                "breast_r": (1.5, 26), "breast_l": (1.5, 26),            # sigma rescaled by bust_R at runtime
                "breast_r_tip": (2.8, 30), "breast_l_tip": (2.8, 30),
                "shirt_r": (1.2, 18), "shirt_l": (1.2, 18), "neck": (0.8, 14),
                "arm_r_up": (0.6, 16), "arm_l_up": (0.6, 16)},
    "bottomwear": {"spine": (1.0, 45), "hip": (1.0, 70), "thigh_r": (1.0, 45), "thigh_l": (1.0, 45)},
    "legwear": {"hip": (1.0, 50), "thigh_r": (1.0, 45), "thigh_l": (1.0, 45)},
    "handwear-r": {"chest": (0.6, 25), "arm_r_up": (1, 30), "arm_r_low": (1, 30)},
    "handwear-l": {"chest": (0.6, 25), "arm_l_up": (1, 30), "arm_l_low": (1, 30)},
    "neck": {"chest": (1.0, 18), "neck": (1.0, 25), "head": (1.0, 18)},
}
RIGID = {"face": "head", "ears-r": "head", "ears-l": "head", "nose": "head", "mouth": "head",
         "eyebrow-r": "head", "eyebrow-l": "head", "headwear": "head",
         "eyewhite-r": "eye_r", "irides-r": "eye_r", "eyelash-r": "eye_r",
         "eyewhite-l": "eye_l", "irides-l": "eye_l", "eyelash-l": "eye_l"}
GRID = {"back_hair_back": 28, "back_hair_front": 24, "front_hair": 18, "topwear": 9,
        "bottomwear": 26, "legwear": 30, "handwear-r": 24, "handwear-l": 24, "neck": 14}


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def grid_mesh(alpha, step):
    h, w = alpha.shape
    nx, ny = max(1, math.ceil(w / step)), max(1, math.ceil(h / step))
    xs, ys = np.linspace(0, w, nx + 1), np.linspace(0, h, ny + 1)
    idx = {}
    verts = []
    def add(i, j):
        if (i, j) not in idx:
            idx[(i, j)] = len(verts)
            verts.append((xs[i], ys[j]))
        return idx[(i, j)]
    # hull first: clockwise perimeter
    for i in range(nx + 1): add(i, 0)
    for j in range(1, ny + 1): add(nx, j)
    for i in range(nx - 1, -1, -1): add(i, ny)
    for j in range(ny - 1, 0, -1): add(0, j)
    hull = len(verts)
    for j in range(1, ny):
        for i in range(1, nx): add(i, j)
    solid = cv2.dilate((alpha > 8).astype(np.uint8), np.ones((5, 5), np.uint8))
    tris = []
    for j in range(ny):
        for i in range(nx):
            cell = solid[int(ys[j]):int(math.ceil(ys[j + 1])), int(xs[i]):int(math.ceil(xs[i + 1]))]
            if cell.any():
                a, b, c, d = idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]
                tris += [a, b, c, a, c, d]
    return np.array(verts, float), hull, tris


def despeckle(rgba):
    """Drop isolated specks (see-through haze survivors) so crops are tight."""
    a = rgba[:, :, 3]
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 24).astype(np.uint8), 8)
    if n <= 1:
        return rgba
    areas = st[1:, cv2.CC_STAT_AREA]
    keep_ids = 1 + np.where(areas >= min(25, 0.2 * areas.max()))[0]
    keep = np.isin(lab, keep_ids).astype(np.uint8)
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    out = rgba.copy()
    out[keep == 0] = 0
    return out


def make_attachments(layout, layers, bones):
    by = {b.name: b for b in bones}
    bidx = {b.name: i for i, b in enumerate(bones)}
    shutil.rmtree(IMG_DIR, ignore_errors=True)
    os.makedirs(IMG_DIR)
    atts, slots, stats = {}, [], {}
    for slot in layout["draw_order"]:
        lay = despeckle(layers[slot])
        x0, y0, x1, y1 = bbox(lay[:, :, 3], 4)
        x0, y0, x1, y1 = max(0, x0 - 2), max(0, y0 - 2), min(W, x1 + 2), min(H, y1 + 2)
        crop = lay[y0:y1, x0:x1]
        save_rgba(crop, os.path.join(IMG_DIR, slot + ".png"))
        h, w = crop.shape[:2]
        if slot in RIGID:
            b = by[RIGID[slot]]
            c = np.array(to_spine(x0 + w / 2, y0 + h / 2)) - b.head
            th = math.radians(-b.angle)
            atts[slot] = {"x": round(c[0] * math.cos(th) - c[1] * math.sin(th), 2),
                          "y": round(c[0] * math.sin(th) + c[1] * math.cos(th), 2),
                          "rotation": round(-b.angle, 2), "width": w, "height": h}
            slots.append({"name": slot, "bone": RIGID[slot], "attachment": slot})
            stats[slot] = {"type": "region", "bone": RIGID[slot]}
            continue
        verts, hull, tris = grid_mesh(crop[:, :, 3], GRID[slot])
        world = np.array([to_spine(x0 + vx, y0 + vy) for vx, vy in verts])
        allowed = WEIGHTS[slot]
        names = list(allowed)
        logw = np.stack([math.log(allowed[n][0]) - 0.5 * (seg_dist(world, by[n].head, by[n].tip) / allowed[n][1]) ** 2
                         for n in names], 1)
        logw -= logw.max(1, keepdims=True)
        wts = np.exp(logw)
        vdata = []
        nb_hist = [0] * 5
        for vi in range(len(world)):
            order = np.argsort(-wts[vi])[:4]
            sel = [(names[k], wts[vi, k]) for k in order if wts[vi, k] > 0.02 * wts[vi, order[0]]]
            tot = sum(v for _, v in sel)
            vdata.append(len(sel))
            nb_hist[len(sel)] += 1
            for n, v in sel:
                b = by[n]
                th = math.radians(-b.angle)
                d = world[vi] - b.head
                vdata += [bidx[n], round(d[0] * math.cos(th) - d[1] * math.sin(th), 2),
                          round(d[0] * math.sin(th) + d[1] * math.cos(th), 2), round(v / tot, 4)]
        uvs = []
        for vx, vy in verts:
            uvs += [round(vx / w, 5), round(vy / h, 5)]
        atts[slot] = {"type": "mesh", "uvs": uvs, "triangles": tris, "vertices": vdata,
                      "hull": hull, "width": w, "height": h}
        root_bone = max(names, key=lambda n: wts[:, names.index(n)].sum())
        slots.append({"name": slot, "bone": root_bone, "attachment": slot})
        stats[slot] = {"type": "weighted_mesh", "vertices": len(verts), "triangles": len(tris) // 3,
                       "bones": names, "influences_histogram": nb_hist[1:]}
    return atts, slots, stats


# ─── animation: primary motion programs ──────────────────────────────────────

TAU = 2 * math.pi


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def blink_value(t, times):
    v = 1.0
    for tb in times:
        d = t - tb
        if 0 <= d <= 0.16:
            v = min(v, 1 - 0.88 * d / 0.07 if d <= 0.07 else 0.12 + 0.88 * (d - 0.07) / 0.09)
    return max(0.12, min(1.0, v))


def breath_wave(t, period):
    ph = (t % period) / period
    return 0.5 - 0.5 * math.cos(TAU * (ph + 0.06 * math.sin(TAU * ph)))


def idle_primary(t, dur=8.0):
    """Calm idle: 2 breaths, slow sway, head counter-motion, blinks."""
    b = breath_wave(t, dur / 2)
    sw = lambda lag=0.0: math.sin(TAU * t / dur - lag)
    m = {"hip": (1.6 * sw(), 0.9 * b, 0.0, 1, 1),            # hip local axes = world axes
         "spine": (0, 0, 0.7 * sw(0.5), 1, 1),
         "chest": (1.1 * b, 0, 0.9 * sw(0.9) - 0.35 * b, 1, 1),   # chest local x = world up
         "breath": (0, 0, 0, 1 + 0.022 * b, 1 + 0.012 * b),
         "neck": (0, 0, -0.5 * sw(1.3) + 0.3 * b, 1, 1),
         "head": (0, 0, -1.1 * sw(1.7) + 0.6 * math.sin(TAU * t / (dur / 2) - 2.0), 1, 1)}
    for s, sg in (("r", 1), ("l", -1)):
        m[f"arm_{s}_up"] = (0, 0, sg * 0.8 * b + 0.6 * sw(1.1), 1, 1)
        m[f"arm_{s}_low"] = (0, 0, sg * 0.5 * b, 1, 1)
        m["thigh_" + s] = (0, 0, -0.4 * sw(0.2), 1, 1)
    bl = blink_value(t, (2.7, 6.1, 6.45))
    for s in ("r", "l"):
        m["eye_" + s] = (0, 0, 0, 1, bl)
    return m


HEAVE_KEYS = [  # (u, value, slope du) -- velocity-continuous (C1) Hermite spline
    (0.00, 0.00, 0.0), (0.52, 1.00, 0.0), (0.60, 1.00, 0.0),
    (0.69, 0.05, -14.0),          # fast, gravity-like fall ...
    (0.74, -0.15, 0.0),           # ... decelerated over ~0.16 s (no infinite stop)
    (0.86, 0.00, 0.0), (1.00, 0.00, 0.0)]


def heave(u):
    """One bounce cycle (u in [0,1)): slow lift, short hang, fast drop, small
    undershoot, settle. Shaped after the reference's body motion. C1 so the
    driven springs never receive an instantaneous velocity change."""
    u %= 1.0
    for (u0, v0, m0), (u1, v1, m1) in zip(HEAVE_KEYS[:-1], HEAVE_KEYS[1:]):
        if u0 <= u <= u1:
            h = u1 - u0
            x = (u - u0) / h
            h00, h10 = 2 * x ** 3 - 3 * x ** 2 + 1, x ** 3 - 2 * x ** 2 + x
            h01, h11 = -2 * x ** 3 + 3 * x ** 2, x ** 3 - x ** 2
            return h00 * v0 + h10 * h * m0 + h01 * v1 + h11 * h * m1
    return 0.0


def bump(u, a, b, c):
    """0 -> 1 between a and b, back to 0 between b and c (smooth both ways)."""
    return smooth((u - a) / (b - a)) * (1 - smooth((u - b) / (c - b)))


def bust_primary(t, dur=6.4):
    """Bust showcase loop: two heave/drop cycles of 3.2 s plus a lateral sway
    at half that rate so left and right respond differently."""
    cyc = dur / 2
    h = heave((t % cyc) / cyc)
    hd = heave(((t - 0.12) % cyc) / cyc)          # head/neck follow a little late
    sw = lambda lag=0.0: math.sin(TAU * t / dur - lag)
    b = breath_wave(t, cyc)
    side = 1.0 if (t % dur) < cyc else -1.0                  # alternate the drop's lateral kick
    kick = side * 2.2 * bump((t % cyc) / cyc, 0.58, 0.70, 0.95)   # returns to 0 before the cycle ends
    m = {"hip": (2.8 * sw() + kick, 6.0 * h + 0.6 * b, 0.0, 1, 1),
         "spine": (0, 0, 1.1 * sw(0.4), 1, 1),
         "chest": (3.5 * h + 0.8 * b, 0, 1.4 * sw(0.8) - 0.8 * h, 1, 1),
         "breath": (0, 0, 0, 1 + 0.03 * h + 0.012 * b, 1 + 0.015 * h),
         "neck": (0, 0, -0.8 * sw(1.2) + 0.9 * hd, 1, 1),
         "head": (0, 0, -1.6 * sw(1.6) - 1.4 * hd, 1, 1)}
    for s, sg in (("r", 1), ("l", -1)):
        m[f"arm_{s}_up"] = (0, 0, sg * 2.2 * h + 0.8 * sw(1.0), 1, 1)
        m[f"arm_{s}_low"] = (0, 0, sg * 1.2 * hd, 1, 1)
        m["thigh_" + s] = (0, 0, -0.6 * sw(0.2), 1, 1)
    bl = blink_value(t, (1.15, 4.6))
    for s in ("r", "l"):
        m["eye_" + s] = (0, 0, 0, 1, bl)
    return m


PROGRAMS = {"idle": (idle_primary, 8.0), "bust_bounce": (bust_primary, 6.4)}


# ─── forward kinematics for the simulations ──────────────────────────────────

def fk(bones, bj, deltas):
    """World (pos, angle) of every bone given local deltas. Scale ignored (tiny)."""
    world = {}
    for b, j in zip(bones, bj):
        dx, dy, dr, _, _ = deltas.get(b.name, (0, 0, 0, 1, 1))
        lx, ly, lr = j.get("x", 0) + dx, j.get("y", 0) + dy, j.get("rotation", 0) + dr
        if b.parent is None:
            world[b.name] = (np.array([lx, ly]), lr)
            continue
        pp, pa = world[b.parent]
        th = math.radians(pa)
        pos = pp + np.array([lx * math.cos(th) - ly * math.sin(th), lx * math.sin(th) + ly * math.cos(th)])
        world[b.name] = (pos, pa + lr)
    return world


def rot2(v, deg):
    th = math.radians(deg)
    return np.array([v[0] * math.cos(th) - v[1] * math.sin(th), v[0] * math.sin(th) + v[1] * math.cos(th)])


# ─── secondary motion: hair / bangs / shirt (linear angular springs) ─────────

SECONDARY = {}
for s in ("r", "l"):
    for i, f in enumerate((1.6, 1.3, 1.1, 0.95)):
        SECONDARY[f"hair_{s}{i + 1}"] = ("rot", f, 0.22, "hair_side")
    SECONDARY["shirt_" + s] = ("rot", 2.0, 0.3, "shirt")
for n in ("bang_r", "bang_c", "bang_l"):
    SECONDARY[n] = ("rot", 1.9, 0.25, "bangs")
SECONDARY["hair_b1"] = ("rot", 1.2, 0.25, "hair_back")
SECONDARY["hair_b2"] = ("rot", 1.0, 0.25, "hair_back")
TARGET_IDLE = {"hair_side": 2.4, "bangs": 1.6, "hair_back": 1.2, "shirt": 1.8}   # deg peak in idle
CAP = {"hair_side": 7.0, "bangs": 4.0, "hair_back": 3.5, "shirt": 5.0}          # deg peak, any program


def simulate_secondary(bones, bj, prim, dur, dt=1 / 120, cycles=3):
    names = [b.name for b in bones if b.name in SECONDARY]
    st = {n: [0.0, 0.0] for n in names}          # angle (deg), angular velocity (rad/s)
    hist = {}
    samples = {n: [] for n in names}
    steps = int(round(dur / dt))
    by = {b.name: b for b in bones}
    for k in range(steps * cycles + 1):
        t = k * dt
        d = prim(t % dur, dur)
        for n in names:
            d[n] = (0, 0, st[n][0], 1, 1)
        world = fk(bones, bj, d)
        for n in names:
            _, f, z, _ = SECONDARY[n]
            kk, c = (TAU * f) ** 2, 2 * z * TAU * f
            pos, ang = world[n]
            _, pang = world[by[n].parent]
            h = hist.setdefault(n, [pos, pos, pang, pang])
            acc = (pos - 2 * h[1] + h[0]) / dt ** 2 if k > 1 else np.zeros(2)
            alpha = math.radians(pang - 2 * h[3] + h[2]) / dt ** 2 if k > 1 else 0.0
            a = math.radians(ang)
            torque = -(acc @ np.array([-math.sin(a), math.cos(a)])) / max(by[n].length, 10.0) - alpha
            st[n][1] += (-kk * math.radians(st[n][0]) - c * st[n][1] + torque) * dt
            st[n][0] += math.degrees(st[n][1] * dt)
            hist[n] = [h[1], pos, h[3], pang]
            if k >= steps * (cycles - 1):
                samples[n].append((t - steps * (cycles - 1) * dt, st[n][0]))
    return samples


# ─── bust soft-body simulation ───────────────────────────────────────────────
#
# Each side is a mass on a nonlinear 2-D spring hanging from the chest:
#   * driven by the inertial (negative) acceleration of its attachment point,
#   * anisotropic: lifting is stiffer than sagging (supported from below by
#     the garment), plus cubic stiffening so big drops overshoot snappily but
#     stay bounded,
#   * low damping (zeta ~0.13) -> several visible rebounds per drop, as in the
#     reference (secondary bounce 1.0-1.4 Hz on top of a ~0.3 Hz body heave),
#   * left/right differ by a few % in frequency -> organic phase drift,
#   * inner-side collision: the two sides can't push through each other.
# Output per side: root swing (rotate), tip translate, tip squash/stretch
# (volume-preserving-ish scale along / across the bone).

BUST = {"fy": 1.05, "fx": 0.9, "zeta": 0.12, "lift_stiff": 1.35, "cubic": 2.5,
        "asym": 0.045, "collide": 0.08, "swing_deg_per_R": 22.0,
        "stretch_per_R": 0.85, "stretch_vel": 0.010}
BUST_TARGET = {"bust_bounce": 0.22, "idle": 0.05}   # peak tip offset / R per program


def simulate_bust(bones, bj, prim, dur, R, gain, dt=1 / 240, cycles=4):
    by = {b.name: b for b in bones}
    res = {}
    for side, sgn_in, asym in (("r", +1, -BUST["asym"]), ("l", -1, +BUST["asym"])):
        root = "breast_" + side
        wy, wx = TAU * BUST["fy"] * (1 + asym), TAU * BUST["fx"] * (1 + asym)
        p, v = np.zeros(2), np.zeros(2)
        hist = []
        out = []
        steps = int(round(dur / dt))
        for k in range(steps * cycles + 1):
            t = k * dt
            world = fk(bones, bj, prim(t % dur, dur))
            pos, ang = world[root]
            tip_rest = pos + rot2(np.array([by[root].length, 0.0]), ang)   # where the mass sits at rest
            hist.append(tip_rest)
            if len(hist) > 3:
                hist.pop(0)
            acc = (hist[-1] - 2 * hist[-2] + hist[-3]) / dt ** 2 if len(hist) == 3 else np.zeros(2)
            q = p / R
            ky = wy * wy * (BUST["lift_stiff"] if p[1] > 0 else 1.0)
            stiff = 1 + BUST["cubic"] * float(q @ q)
            f = np.array([-wx * wx * p[0], -ky * p[1]]) * stiff - 2 * BUST["zeta"] * np.array([wx, wy]) * v
            f -= gain * acc
            inner = sgn_in * p[0] - BUST["collide"] * R            # penetration toward the other side
            if inner > 0:
                f[0] -= sgn_in * (6 * wx * wx) * inner
            v += f * dt
            p += v * dt
            if k >= steps * (cycles - 1) and k % 4 == 0:
                out.append((t - steps * (cycles - 1) * dt, p[0], p[1], v[0], v[1], ang))
        res[side] = np.array(out)
    return res


def bust_channels(series, R):
    """Turn simulated mass offsets into Spine channels for root + tip bones."""
    t, px, py, vx, vy, ang = series.T
    swing = BUST["swing_deg_per_R"] * px / R                      # sideways swing of the whole shape
    # tip translate expressed in the root bone frame (root world angle + swing)
    loc = np.array([rot2((x, y), -(a + s)) for x, y, a, s in zip(px, py, ang, swing)])
    # world (px, py) after removing the part the swing already produced
    stretch = 1 - BUST["stretch_per_R"] * py / R - BUST["stretch_vel"] * vy / R * R / 10.0
    stretch = np.clip(stretch, 0.82, 1.22)
    across = stretch ** -0.6
    return t, swing, loc[:, 0] * 0.65, loc[:, 1] * 0.65, stretch, across


def resample(t_src, cols, fps, dur):
    """Values at key times; last key blended to the first (seamless loop)."""
    ts = np.linspace(0, dur, int(round(dur * fps)) + 1)
    out = []
    for c in cols:
        o = np.interp(ts, t_src, c)
        nb = max(2, int(0.3 * fps))
        for k in range(nb):
            a = (k + 1) / nb
            o[-nb + k] = o[-nb + k] * (1 - a) + o[0] * a
        out.append(o)
    return ts, out


def keys_rot(ts, vals):
    return [{"time": round(float(t), 4), "value": round(float(v), 3)} for t, v in zip(ts, vals)]


def keys_xy(ts, xs, ys):
    return [{"time": round(float(t), 4), "x": round(float(x), 3), "y": round(float(y), 3)}
            for t, x, y in zip(ts, xs, ys)]


def primary_timelines(prim, dur, fps):
    ts = np.linspace(0, dur, int(round(dur * fps)) + 1)
    P = [prim(t, dur) for t in ts]
    out = {}
    for name in P[0]:
        tl = {}
        vals = np.array([p[name] for p in P])
        if np.abs(vals[:, 2]).max() > 1e-4:
            tl["rotate"] = keys_rot(ts, vals[:, 2])
        if np.abs(vals[:, :2]).max() > 1e-4:
            tl["translate"] = keys_xy(ts, vals[:, 0], vals[:, 1])
        if np.abs(vals[:, 3:] - 1).max() > 1e-4:
            tl["scale"] = keys_xy(ts, vals[:, 3], vals[:, 4])
        if tl:
            out[name] = tl
    return out


def build_animations(bones, bj, lm):
    R = bust_R(lm)
    # ---- calibrate bust gain per program (same spring model, different drive level)
    bust_gain = {}
    for name, (prim, dur) in PROGRAMS.items():
        probe = simulate_bust(bones, bj, prim, dur, R, 1.0, cycles=3)
        peak = max(np.abs(probe[s][:, 1:3]).max() for s in probe) / R
        bust_gain[name] = BUST_TARGET[name] / max(peak, 1e-9)
    gain = bust_gain
    sec_idle = simulate_secondary(bones, bj, *PROGRAMS["idle"])
    gains = {}
    for grp in set(v[3] for v in SECONDARY.values()):
        pk = max(max(abs(x[1]) for x in sec_idle[n]) for n in sec_idle if SECONDARY[n][3] == grp)
        gains[grp] = TARGET_IDLE[grp] / max(pk, 1e-9)
    anims, report = {}, {"bust_gain": {k: round(v, 4) for k, v in gain.items()}, "bust_R_px": round(R, 1), "secondary_gains": gains}
    for name, (prim, dur) in PROGRAMS.items():
        prim_tl = primary_timelines(prim, dur, FPS_BAKE)
        sec = sec_idle if name == "idle" else simulate_secondary(bones, bj, prim, dur)
        baked = dict(prim_tl)
        caps = {}
        for grp in gains:
            pk = max(max(abs(x[1]) for x in sec[n]) for n in sec if SECONDARY[n][3] == grp) * gains[grp]
            caps[grp] = min(1.0, CAP[grp] / max(pk, 1e-9))
        for n, ser in sec.items():
            grp = SECONDARY[n][3]
            arr = np.array(ser)
            ts, (th,) = resample(arr[:, 0], [arr[:, 1] * gains[grp] * caps[grp]], FPS_BAKE, dur)
            baked[n] = {"rotate": keys_rot(ts, th)}
        bust = simulate_bust(bones, bj, prim, dur, R, gain[name])
        stats = {}
        for side in ("r", "l"):
            t, swing, tx, ty, sv, sh = bust_channels(bust[side], R)
            ts, (swing, tx, ty, sv, sh) = resample(t, [swing, tx, ty, sv, sh], 30, dur)
            baked["breast_" + side] = {"rotate": keys_rot(ts, swing)}
            baked["breast_" + side + "_tip"] = {"translate": keys_xy(ts, tx, ty), "scale": keys_xy(ts, sv, sh)}
            stats[side] = {"tip_disp_px_max": round(float(np.hypot(tx, ty).max()), 2),
                           "swing_deg_range": [round(float(swing.min()), 2), round(float(swing.max()), 2)],
                           "stretch_range": [round(float(sv.min()), 3), round(float(sv.max()), 3)]}
        anims[name] = baked
        anims[name + "_physics"] = dict(prim_tl)
        report[name] = {"duration": dur, "bust": stats, "hair_caps": caps}
    return anims, report


# ─── physics constraints (Spine 4.2) ─────────────────────────────────────────

def physics_constraints(bones):
    out = []
    for b in bones:
        n = b.name
        if n in SECONDARY:
            _, f, _, grp = SECONDARY[n]
            c = {"name": "phys_" + n, "bone": n, "rotate": 1, "inertia": 0.5,
                 "strength": round(40 * f * f, 1), "damping": 0.85, "mass": 1.0}
        elif n in ("breast_r", "breast_l"):
            c = {"name": "phys_" + n, "bone": n, "rotate": 0.35, "inertia": 0.6,
                 "strength": 110, "damping": 0.82, "mass": 1.2}
        elif n.endswith("_tip") and n.startswith("breast_"):
            c = {"name": "phys_" + n, "bone": n, "x": 0.9, "y": 1.0, "scaleX": 0.5, "inertia": 0.75,
                 "strength": 140, "damping": 0.8, "mass": 1.4}
        else:
            continue
        c.update({"order": len(out), "limit": 5000})
        out.append(c)
    return out


def main():
    layout, layers = load_layers()
    lm = landmarks(layout, layers)
    R = bust_R(lm)
    tw = WEIGHTS["topwear"]
    for s in ("r", "l"):
        tw["breast_" + s] = (tw["breast_" + s][0], 0.34 * R)
        tw["breast_" + s + "_tip"] = (tw["breast_" + s + "_tip"][0], 0.50 * R)
    bones = build_skeleton(lm)
    bj = bone_json(bones)
    atts, slots, att_stats = make_attachments(layout, layers, bones)
    anims, anim_report = build_animations(bones, bj, lm)
    phys = physics_constraints(bones)
    # physics on by default (setup mix 1); baked animations switch it off so the
    # secondary motion is never applied twice, *_physics animations switch it on.
    for name, a in anims.items():
        a.setdefault("bones", {})
        val = 1 if name.endswith("_physics") else 0
        anims[name] = {"bones": {k: v for k, v in a.items() if k != "bones"},
                       "physics": {p["name"]: {"mix": [{"time": 0, "value": val}]} for p in phys}}

    config = {"skeleton": {"name": NAME, "width": W, "height": H}, "bones": bj, "slots": slots,
              "attachments": atts, "animations": [], "custom_animations": anims}
    spine = bsj.build_spine_json(config)          # official skill builder (4.2 wire format)
    spine["skeleton"]["fps"] = 30
    spine["skeleton"]["referenceScale"] = 100
    spine["skeleton"]["images"] = "./images/"
    spine["physics"] = phys

    shutil.rmtree(SPINE_DIR, ignore_errors=True)
    os.makedirs(SPINE_DIR)
    write_json(spine, os.path.join(SPINE_DIR, NAME + ".json"))
    rc = os.system(f'{sys.executable} "{os.path.join(HERE, "vendor", "make_atlas.py")}" '
                   f'--parts "{IMG_DIR}" --output "{SPINE_DIR}" --name {NAME} --padding 2 > /dev/null')
    assert rc == 0, "make_atlas failed"
    shutil.copytree(IMG_DIR, os.path.join(SPINE_DIR, "images"))

    weight_map(layout, layers, bones, atts, os.path.join(OUT, "verification", "bust_weight_map.png"))
    rig_report = {"landmarks_px": dict(lm),
                  "bones": [{"name": b.name, "parent": b.parent, "head_spine": [round(v, 1) for v in b.head],
                             "angle": round(b.angle, 1), "length": round(b.length, 1)} for b in bones],
                  "attachments": att_stats, "physics_constraints": [p["name"] for p in phys],
                  "bust_params": BUST, "animations": anim_report}
    write_json(json.loads(json.dumps(rig_report, default=float)), os.path.join(OUT, "work", "rig_report.json"))
    print(f"bones={len(bones)} slots={len(slots)} physics={len(phys)} anims={list(anims)}")
    print(json.dumps(json.loads(json.dumps(anim_report, default=float)), indent=1))


def weight_map(layout, layers, bones, atts, path):
    """Debug image: topwear mesh coloured by bust influence (red=right root,
    yellow=right tip, blue=left root, cyan=left tip)."""
    names = [b.name for b in bones]
    a = atts["topwear"]
    v = a["vertices"]
    nverts = len(a["uvs"]) // 2
    cols = {"breast_r": (230, 60, 60), "breast_r_tip": (255, 210, 40),
            "breast_l": (60, 90, 230), "breast_l_tip": (40, 220, 230)}
    lay = layers["topwear"]
    x0, y0, x1, y1 = bbox(despeckle(lay)[:, :, 3], 4)
    x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
    img = (lay[:, :, :3].astype(np.float32) * 0.45 + 255 * 0.55 * (lay[:, :, 3:] < 10)).astype(np.uint8).copy()
    i = 0
    for vi in range(nverts):
        n = v[i]; i += 1
        c = np.zeros(3)
        for _ in range(n):
            bi, _, _, w = v[i:i + 4]; i += 4
            c += np.array(cols.get(names[bi], (200, 200, 200))) * w
        u, vv = a["uvs"][2 * vi], a["uvs"][2 * vi + 1]
        px, py = int(x0 + u * a["width"]), int(y0 + vv * a["height"])
        cv2.circle(img, (px, py), 2, tuple(int(x) for x in c), -1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img[220:510, 230:530], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()

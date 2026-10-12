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
FPS_BAKE = 15      # key density for baked curves


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
    r = lm["breast_radius"]
    for s in ("r", "l"):
        bx, by = lm["breast_" + s]
        add("breast_" + s, "chest", (bx, by - 0.9 * r), (bx, by + 0.4 * r))
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
    "topwear": {"spine": (1.0, 45), "breath": (1.0, 110), "breast_r": (2.2, 32), "breast_l": (2.2, 32),
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
GRID = {"back_hair_back": 28, "back_hair_front": 24, "front_hair": 18, "topwear": 20,
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


# ─── animation: primary motion ───────────────────────────────────────────────

TAU = 2 * math.pi


def breath(t):          # 0 exhaled .. 1 inhaled, two breaths per loop, slightly faster inhale
    ph = (t % (DUR / 2)) / (DUR / 2)
    return 0.5 - 0.5 * math.cos(TAU * (ph + 0.06 * math.sin(TAU * ph)))


def sway(t, lag=0.0):
    return math.sin(TAU * t / DUR - lag)


def primary(t):
    """Local deltas per bone at time t: {bone: (dx, dy, drot, sx, sy)} (dx/dy in parent space)."""
    b = breath(t)
    m = {}
    # hip bone points up (parent root unrotated): local x = world x
    m["hip"] = (1.6 * sway(t), 0.9 * b, 0.0, 1, 1)
    m["spine"] = (0, 0, 0.7 * sway(t, 0.5), 1, 1)
    m["chest"] = (1.1 * b, 0, 0.9 * sway(t, 0.9) - 0.35 * b, 1, 1)   # chest local x = world up
    m["breath"] = (0, 0, 0, 1 + 0.022 * b, 1 + 0.012 * b)               # local x = up (length), y = width
    m["neck"] = (0, 0, -0.5 * sway(t, 1.3) + 0.3 * b, 1, 1)
    m["head"] = (0, 0, -1.1 * sway(t, 1.7) + 0.6 * math.sin(TAU * t / (DUR / 2) - 2.0), 1, 1)
    for s, sg in (("r", 1), ("l", -1)):
        m[f"arm_{s}_up"] = (0, 0, sg * 0.8 * b + 0.6 * sway(t, 1.1), 1, 1)
        m[f"arm_{s}_low"] = (0, 0, sg * 0.5 * b, 1, 1)
        m["thigh_" + s] = (0, 0, -0.4 * sway(t, 0.2), 1, 1)
    # blinks (eye bone points +x world; local y = world y)
    blink = 1.0
    for tb in (2.7, 6.1, 6.45):
        d = t - tb
        if 0 <= d <= 0.16:
            blink = min(blink, 0.12 + 0.88 * abs(d - 0.07) / 0.09 if d > 0.07 else 1 - 0.88 * d / 0.07)
    for s in ("r", "l"):
        m["eye_" + s] = (0, 0, 0, 1, max(0.12, min(1.0, blink)))
    return m


# ─── forward kinematics for the spring simulation ────────────────────────────

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


# secondary bones: kind, natural freq (Hz), damping ratio, group (amplitude normalisation)
SECONDARY = {}
for s in ("r", "l"):
    for i, f in enumerate((1.6, 1.3, 1.1, 0.95)):
        SECONDARY[f"hair_{s}{i + 1}"] = ("rot", f, 0.22, "hair_side")
    SECONDARY["breast_" + s] = ("trans+rot", 2.2, 0.18, "breast")
    SECONDARY["shirt_" + s] = ("rot", 2.0, 0.3, "shirt")
for n in ("bang_r", "bang_c", "bang_l"):
    SECONDARY[n] = ("rot", 1.9, 0.25, "bangs")
SECONDARY["hair_b1"] = ("rot", 1.2, 0.25, "hair_back")
SECONDARY["hair_b2"] = ("rot", 1.0, 0.25, "hair_back")
TARGET = {"hair_side": 2.4, "bangs": 1.6, "hair_back": 1.2, "shirt": 1.8,
          "breast": 2.6, "breast_rot": 2.0}   # deg (rot) / px (trans) peak


def simulate(bones, bj, dt=1 / 120, cycles=3):
    names = [b.name for b in bones if b.name in SECONDARY]
    st = {n: dict(th=0.0, w=0.0, x=0.0, vx=0.0, y=0.0, vy=0.0) for n in names}
    hist_pos, hist_ang = {}, {}
    samples = {n: [] for n in names}
    steps = int(round(DUR / dt))
    by = {b.name: b for b in bones}
    for k in range(steps * cycles + 1):
        t = k * dt
        d = primary(t % DUR)
        for n in names:
            s = st[n]
            d[n] = (s["x"], s["y"], s["th"], 1, 1)
        world = fk(bones, bj, d)
        for n in names:
            kind, f, z, _ = SECONDARY[n]
            kk, c = (TAU * f) ** 2, 2 * z * TAU * f
            pos, ang = world[n]
            ppos, pang = world[by[n].parent]
            if kind == "trans+rot":
                pos = pos - np.array([st[n].get("wx", 0.0), st[n].get("wy", 0.0)])
            hp = hist_pos.setdefault(n, [pos, pos])
            ha = hist_ang.setdefault(n, [pang, pang])
            acc = (pos - 2 * hp[1] + hp[0]) / dt ** 2 if k > 1 else np.zeros(2)
            alpha = math.radians(pang - 2 * ha[1] + ha[0]) / dt ** 2 if k > 1 else 0.0
            s = st[n]
            L = max(by[n].length, 10.0)
            a_rad = math.radians(ang)
            nrm = np.array([-math.sin(a_rad), math.cos(a_rad)])
            torque = -(acc @ nrm) / L - alpha
            s["w"] += (-kk * math.radians(s["th"]) * 1 - c * s["w"] + torque) * dt
            s["th"] += math.degrees(s["w"] * dt)
            if kind == "trans+rot":
                # translation lags the head acceleration (in world), stored in world units
                s["vx"] += (-kk * s.get("wx", 0) - c * s["vx"] - acc[0]) * dt
                s["vy"] += (-kk * s.get("wy", 0) - c * s["vy"] - acc[1]) * dt
                s["wx"] = s.get("wx", 0) + s["vx"] * dt
                s["wy"] = s.get("wy", 0) + s["vy"] * dt
                th = math.radians(-pang)   # world -> parent-local
                s["x"] = s["wx"] * math.cos(th) - s["wy"] * math.sin(th)
                s["y"] = s["wx"] * math.sin(th) + s["wy"] * math.cos(th)
            hp[0], hp[1] = hp[1], pos
            ha[0], ha[1] = ha[1], pang
            if k >= steps * (cycles - 1):
                samples[n].append((t - steps * (cycles - 1) * dt, s["th"], s["x"], s["y"]))
    # amplitude normalisation per group (linear system -> scaling the response is exact)
    gains = {}
    for grp in set(v[3] for v in SECONDARY.values()):
        members = [n for n in names if SECONDARY[n][3] == grp]
        if grp == "breast":
            peak = max(max(abs(v[3]) for v in samples[n]) for n in members)
            peak_r = max(max(abs(v[1]) for v in samples[n]) for n in members)
            gains["breast"] = TARGET["breast"] / max(peak, 1e-6)
            gains["breast_rot"] = TARGET["breast_rot"] / max(peak_r, 1e-6)
        else:
            peak = max(max(abs(v[1]) for v in samples[n]) for n in members)
            gains[grp] = TARGET[grp] / max(peak, 1e-6)
    return samples, gains


def resample(series, fps):
    """Pick values at key times; force the last key to equal the first (seamless loop)."""
    arr = np.array(series)
    ts = np.linspace(0, DUR, int(DUR * fps) + 1)
    out = [np.interp(ts, arr[:, 0], arr[:, i]) for i in range(1, arr.shape[1])]
    for o in out:  # blend the final 0.4 s toward the start value to absorb sim drift
        nb = int(0.4 * fps)
        for k in range(nb):
            a = (k + 1) / nb
            o[-nb + k] = o[-nb + k] * (1 - a) + o[0] * a
    return ts, out


def keys_rot(ts, vals):
    return [{"time": round(float(t), 4), "value": round(float(v), 3)} for t, v in zip(ts, vals)]


def keys_xy(ts, xs, ys):
    return [{"time": round(float(t), 4), "x": round(float(x), 3), "y": round(float(y), 3)}
            for t, x, y in zip(ts, xs, ys)]


def build_animations(bones, bj):
    ts = np.linspace(0, DUR, int(DUR * FPS_BAKE) + 1)
    P = [primary(t) for t in ts]
    prim = {}
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
            prim[name] = tl
    samples, gains = simulate(bones, bj)
    sec = {}
    for n, ser in samples.items():
        kind, _, _, grp = SECONDARY[n]
        t2, (th, x, y) = resample(ser, FPS_BAKE)
        if kind == "trans+rot":
            sec[n] = {"rotate": keys_rot(t2, th * gains["breast_rot"]),
                      "translate": keys_xy(t2, x * gains["breast"], y * gains["breast"])}
        else:
            sec[n] = {"rotate": keys_rot(t2, th * gains[grp])}
    idle = {"bones": {**prim, **sec}}
    idle_phys = {"bones": dict(prim)}
    return idle, idle_phys, gains


# ─── physics constraints (Spine 4.2) ─────────────────────────────────────────

def physics_constraints(bones):
    out = []
    order = 0
    for b in bones:
        if b.name not in SECONDARY:
            continue
        kind, f, z, grp = SECONDARY[b.name]
        c = {"name": "phys_" + b.name, "order": order, "bone": b.name, "rotate": 1,
             "inertia": 0.5, "strength": round(40 * f * f, 1), "damping": 0.85, "mass": 1.0,
             "limit": 5000}
        if kind == "trans+rot":
            c.update({"x": 0.6, "y": 0.8, "rotate": 0.3, "inertia": 0.6, "damping": 0.8})
        if grp.startswith("hair"):
            c["wind"], c["gravity"] = 0, 0
        out.append(c)
        order += 1
    return out


def main():
    layout, layers = load_layers()
    lm = landmarks(layout, layers)
    bones = build_skeleton(lm)
    bj = bone_json(bones)
    atts, slots, att_stats = make_attachments(layout, layers, bones)
    idle, idle_phys, gains = build_animations(bones, bj)
    phys = physics_constraints(bones)
    # physics on by default (setup mix 1); the baked idle switches it off so
    # secondary motion is not applied twice, idle_physics switches it on.
    idle["physics"] = {p["name"]: {"mix": [{"time": 0, "value": 0}]} for p in phys}
    idle_phys["physics"] = {p["name"]: {"mix": [{"time": 0, "value": 1}]} for p in phys}

    config = {"skeleton": {"name": NAME, "width": W, "height": H}, "bones": bj, "slots": slots,
              "attachments": atts, "animations": [],
              "custom_animations": {"idle": idle, "idle_physics": idle_phys}}
    spine = bsj.build_spine_json(config)          # official skill builder (4.2 wire format)
    spine["skeleton"]["fps"] = 30
    spine["skeleton"]["referenceScale"] = 100
    spine["skeleton"]["images"] = "./images/"
    spine["physics"] = phys

    shutil.rmtree(SPINE_DIR, ignore_errors=True)
    os.makedirs(SPINE_DIR)
    write_json(spine, os.path.join(SPINE_DIR, NAME + ".json"))
    # atlas via the official skill script
    rc = os.system(f'{sys.executable} "{os.path.join(HERE, "vendor", "make_atlas.py")}" '
                   f'--parts "{IMG_DIR}" --output "{SPINE_DIR}" --name {NAME} --padding 2 > /dev/null')
    assert rc == 0, "make_atlas failed"
    shutil.copytree(IMG_DIR, os.path.join(SPINE_DIR, "images"))  # loose images for Spine editor import

    rig_report = {"landmarks_px": {k: v for k, v in lm.items()},
                  "bones": [{"name": b.name, "parent": b.parent, "head_spine": [round(v, 1) for v in b.head],
                             "angle": round(b.angle, 1), "length": round(b.length, 1)} for b in bones],
                  "attachments": att_stats, "secondary_gains": gains,
                  "physics_constraints": [p["name"] for p in phys],
                  "animations": {"idle": {"duration": DUR, "bones_keyed": len(idle["bones"])},
                                 "idle_physics": {"duration": DUR, "bones_keyed": len(idle_phys["bones"])}}}
    write_json(json.loads(json.dumps(rig_report, default=float)), os.path.join(OUT, "work", "rig_report.json"))
    print(f"bones={len(bones)} slots={len(slots)} meshes={sum(1 for a in atts.values() if a.get('type') == 'mesh')} "
          f"physics={len(phys)} gains={ {k: round(v, 2) for k, v in gains.items()} }")


if __name__ == "__main__":
    main()

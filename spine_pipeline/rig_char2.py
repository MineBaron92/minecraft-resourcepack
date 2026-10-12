#!/usr/bin/env python3
"""
Rig + animation for character 2 (full body, A-pose, PSD input).

Reuses rig.py's machinery (weighted grid meshes, secondary springs, bust
soft-body, physics constraints, Spine 4.2 export via the official builder)
and only swaps what is character-specific:

  * landmarks measured from this body: ponytail chain from the back-hair
    silhouette, bangs/side locks, cardigan front hems, pleated-skirt panels,
    arm axes (PCA of each sleeve layer), legs, bust from the white top region
  * skeleton: feet stay planted (shins hang from root, thighs from hip, the
    leg mesh blends between them), so body bounce never slides the shoes
  * motion programs: light idle and bust_bounce scaled to this figure

Usage: SPINE_OUT=output/char2 python3 spine_pipeline/rig_char2.py
"""
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig as R  # noqa: E402  (shared machinery; module globals are re-pointed below)
from common import OUT, write_json  # noqa: E402

layout = json.load(open(os.path.join(R.STAGE1, "layout_corrected.json")))
R.W, R.H = layout["canvas"]
W, H = R.W, R.H


# ─── landmarks ───────────────────────────────────────────────────────────────

def pca_axis(alpha):
    ys, xs = np.where(alpha > 128)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    proj = (pts - c) @ d
    return c + d * np.percentile(proj, 2), c + d * np.percentile(proj, 98)


def landmarks(L):
    bb = lambda n: R.bbox(L[n][:, :, 3])
    lm = {k: bb(k) for k in ("back_hair", "front_hair", "face", "neck", "topwear", "bottomwear",
                             "legwear", "footwear", "handwear-r", "handwear-l")}
    F, T = lm["face"], lm["topwear"]
    # bust: the white ribbed top (bright, low saturation) inside the top layer, minus the cardigan
    t = L["topwear"].astype(int)
    rgb = t[:, :, :3]
    white = (t[:, :, 3] > 128) & (rgb.min(2) > 185) & (rgb.max(2) - rgb.min(2) < 30)
    # the ribbing lines cut the top into strips: close them, then take the union of big pieces
    white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(white)
    areas = st[1:, cv2.CC_STAT_AREA]
    big = 1 + np.where(areas >= 0.15 * areas.max())[0]
    white = np.isin(lab, big).astype(np.uint8)
    yy, xx = np.where(white)
    wx0, wy0, ww, wh = xx.min(), yy.min(), xx.max() - xx.min() + 1, yy.max() - yy.min() + 1
    lm["white_top"] = [int(wx0), int(wy0), int(wx0 + ww), int(wy0 + wh)]
    R_ = 0.20 * ww                     # the merged region also catches light cardigan edges
    by = wy0 + 0.45 * wh
    lm["bust_R"] = float(R_)
    lm["bust_r"] = (float(wx0 + 0.28 * ww), float(by))       # character right = viewer left
    lm["bust_l"] = (float(wx0 + 0.72 * ww), float(by))
    # ponytail: back-hair pixels right of the face, centre line top->bottom
    bh = L["back_hair"][:, :, 3] > 128
    side = bh.copy()
    side[:, :F[2] - 15] = False
    rows = [y for y in range(H) if side[y].sum() > 5]
    ys = np.linspace(rows[0] + 15, rows[-1] - 6, 4)
    lm["ponytail"] = [(float(np.median(np.where(side[int(y) - 4:int(y) + 4].any(0))[0])), float(y)) for y in ys]
    top = np.argwhere(bh)
    tip = top[np.argmin(top[:, 0])]                          # highest back-hair pixel = tuft tip
    lm["tuft"] = (float(tip[1]), float(tip[0]))
    # bangs / side locks: 3 columns of the front hair
    fh = L["front_hair"][:, :, 3] > 128
    FH = lm["front_hair"]
    cols = np.linspace(FH[0], FH[2], 4)
    lm["bangs"] = []
    for a, b in zip(cols[:-1], cols[1:]):
        yy = np.where(fh[:, int(a):int(b)].any(1))[0]
        lm["bangs"].append(((a + b) / 2, float(yy.min() + 25), float(yy.max() - 10)))
    # eyes
    for s in ("r", "l"):
        x0, y0, x1, y1 = R.bbox(L["eyelash-" + s][:, :, 3], 60)
        lm["eye_" + s] = ((x0 + x1) / 2, (y0 + y1) / 2 + 3)
    # arms: PCA axis of each sleeve layer, shoulder = end nearer the torso centre
    tcx = (T[0] + T[2]) / 2
    for s in ("r", "l"):
        p0, p1 = pca_axis(L["handwear-" + s][:, :, 3])
        sh, hand = (p0, p1) if abs(p0[0] - tcx) < abs(p1[0] - tcx) else (p1, p0)
        # The PCA end is the TOP EDGE of the sleeve. Rotating about that edge swings the arm
        # root out over the shoulder ("dislocated"). Put the joint inside the torso
        # silhouette instead: down the arm axis and in toward the body, so the seam stays
        # hidden under the cardigan shoulder when the arm rises.
        a_len = float(np.hypot(*(hand - sh)))
        a_wid = float((L["handwear-" + s][:, :, 3] > 128).sum()) / a_len
        axis = (hand - sh) / a_len
        inward = np.array([np.sign(tcx - sh[0]), 0.0])
        sh = sh + axis * 0.18 * a_wid + inward * 0.45 * a_wid
        elbow = sh + 0.48 * (hand - sh)
        wrist = sh + 0.80 * (hand - sh)
        lm["arm_width_" + s] = a_wid
        lm["arm_" + s] = (tuple(sh), tuple(elbow), tuple(wrist), tuple(hand))
    # legs: split the leg layer at the body centre line
    lg = L["legwear"][:, :, 3] > 128
    LG = lm["legwear"]
    for s, sel in (("r", slice(0, int(tcx))), ("l", slice(int(tcx), W))):
        m = np.zeros_like(lg)
        m[:, sel] = lg[:, sel]
        yy, xx = np.where(m)
        top_y, bot_y = yy.min(), yy.max()
        def cx_at(y):
            r = np.where(m[int(y)])[0]
            return float(r.mean())
        knee_y = top_y + 0.48 * (bot_y - top_y)
        lm["leg_" + s] = ((cx_at(top_y + 20), float(top_y + 20)), (cx_at(knee_y), float(knee_y)),
                          (cx_at(bot_y - 10), float(bot_y - 10)))
    # skirt panels and cardigan hems
    B = lm["bottomwear"]
    lm["skirt"] = [(B[0] + f * (B[2] - B[0]), B[1] + 0.25 * (B[3] - B[1]), B[3] - 6) for f in (0.2, 0.5, 0.8)]
    cg = (t[:, :, 3] > 128) & ~white.astype(bool)
    lm["cardigan"] = {}
    for s, sel in (("r", slice(0, int(tcx))), ("l", slice(int(tcx), W))):
        m = np.zeros_like(cg)
        m[:, sel] = cg[:, sel]
        yy, xx = np.where(m)
        low = yy > np.percentile(yy, 80)
        lm["cardigan"][s] = (float(xx[low].mean()), float(np.percentile(yy, 55)), float(yy.max()))
    return lm


# ─── skeleton ────────────────────────────────────────────────────────────────

def build_skeleton(lm):
    B, T, F, N = lm["bottomwear"], lm["topwear"], lm["face"], lm["neck"]
    Bone = R.Bone
    cx = (T[0] + T[2]) / 2
    ncx = (N[0] + N[2]) / 2
    hip_y = B[1] + 0.35 * (B[3] - B[1])
    waist_y = B[1] + 5
    chest_y = T[1] + 0.40 * (T[3] - T[1])
    neck_y = N[3] - 12
    head_y = F[3] - 18
    bones = [Bone("root", None, (W / 2, H), (W / 2, H - 1))]
    bones[0].length, bones[0].angle = 0.0, 0.0
    add = lambda *a, **k: bones.append(Bone(*a, **k))
    add("hip", "root", (cx, hip_y), (cx, waist_y))
    add("spine", "hip", (cx, waist_y), (cx, chest_y))
    add("chest", "spine", (cx, chest_y), (ncx, neck_y))
    add("breath", "chest", (cx, chest_y), (cx, chest_y - 50))
    Rb = lm["bust_R"]
    for s in ("r", "l"):
        bx, by = lm["bust_" + s]
        add("breast_" + s, "chest", (bx, by - 1.0 * Rb), (bx, by))
        add("breast_" + s + "_tip", "breast_" + s, (bx, by), (bx, by + 0.8 * Rb))
    add("neck", "chest", (ncx, neck_y), (ncx, head_y))
    add("head", "neck", (ncx, head_y), (ncx, F[1]))
    for s in ("r", "l"):
        ex, ey = lm["eye_" + s]
        add("eye_" + s, "head", (ex, ey), (ex + 10, ey))
    for i, (bx, y0, y1) in enumerate(lm["bangs"]):
        add(["bang_r", "bang_c", "bang_l"][i], "head", (bx, y0), (bx, y1))
    p = lm["ponytail"]
    add("pony1", "head", p[0], p[1])
    add("pony2", "pony1", p[1], p[2])
    add("pony3", "pony2", p[2], p[3])
    add("tuft", "head", p[0], lm["tuft"])
    for s in ("r", "l"):
        x, y0, y1 = lm["cardigan"][s]
        add("cardigan_" + s, "chest", (x, y0), (x, y1))
    for i, (x, y0, y1) in enumerate(lm["skirt"]):
        add(["skirt_r", "skirt_c", "skirt_l"][i], "hip", (x, y0), (x, y1))
    for s in ("r", "l"):
        a0, a1, a2, a3 = lm["arm_" + s]
        # clavicle: lets the shoulder rise with the arm instead of a pin joint at the armpit
        sx = a0[0] * 0.55 + ncx * 0.45
        add("shoulder_" + s, "chest", (sx, a0[1] + 4), a0)
        add(f"arm_{s}_up", "shoulder_" + s, a0, a1)
        # deltoid: a short helper on the upper arm right at the joint; only the cardigan's
        # shoulder cap is weighted to it, so the fabric stretches over the rising sleeve root
        # instead of the root popping out above the shoulder line
        ax = np.subtract(a1, a0) / np.hypot(*np.subtract(a1, a0))
        aw = lm["arm_width_" + s]
        # segment sits on the shoulder CAP (outward from the inner pivot), away from the top
        add("deltoid_" + s, f"arm_{s}_up", tuple(np.add(a0, ax * 0.45 * aw)), tuple(np.add(a0, ax * 1.05 * aw)))
        add(f"arm_{s}_low", f"arm_{s}_up", a1, a2)
        add("hand_" + s, f"arm_{s}_low", a2, a3)
    for s in ("r", "l"):
        l0, l1, l2 = lm["leg_" + s]
        add("thigh_" + s, "hip", l0, l1)
        # front view: a crouch reads as the legs FORESHORTENING (knees toward the camera),
        # so legs compress along their length instead of bending sideways. noScale keeps
        # the shin's own scale independent of the thigh's (its position still follows).
        add("shin_" + s, "thigh_" + s, l1, l2, inherit="noScale")
    LEG["len"] = float(np.mean([np.hypot(*(np.subtract(lm["leg_" + s][2], lm["leg_" + s][0]))) for s in "rl"]))
    return bones


LEG = {"len": 400.0}


HAIR_BACK = {"head": (1.0, 55), "pony1": (1.2, 22), "pony2": (1.2, 22), "pony3": (1.2, 24), "tuft": (1.2, 16)}
WEIGHTS = {
    "back_hair": HAIR_BACK,
    "front_hair": {"head": (1.0, 40), "bang_r": (1, 22), "bang_c": (1, 22), "bang_l": (1, 22)},
    "topwear": {"spine": (1.0, 35), "breath": (1.3, 80), "breast_r": (1.5, 12), "breast_l": (1.5, 12),
                "breast_r_tip": (2.8, 14), "breast_l_tip": (2.8, 14),
                # no arm influence: the upper-arm line runs along the cardigan side, and
                # raised arms (jump_cheer) dragged those vertices into a sawtooth edge
                "cardigan_r": (1.4, 14), "cardigan_l": (1.4, 14), "neck": (0.8, 10),
                # shoulders only near the joint (small rotations: shrug), not along the arm line
                "shoulder_r": (0.9, 10), "shoulder_l": (0.9, 10),
                "deltoid_r": (1.3, 22), "deltoid_l": (1.3, 22)},     # sigma rescaled to arm width
    "bottomwear": {"hip": (1.0, 40), "spine": (0.6, 20), "skirt_r": (1.2, 26), "skirt_c": (1.2, 26),
                   "skirt_l": (1.2, 26)},
    "legwear": {"hip": (0.8, 35), "thigh_r": (1.0, 40), "thigh_l": (1.0, 40), "shin_r": (1.0, 40),
                "shin_l": (1.0, 40)},
    "footwear": {"shin_r": (1.0, 40), "shin_l": (1.0, 40)},
    "handwear-r": {"shoulder_r": (0.5, 12), "arm_r_up": (1, 24), "arm_r_low": (1, 22), "hand_r": (1.3, 12)},
    "handwear-l": {"shoulder_l": (0.5, 12), "arm_l_up": (1, 24), "arm_l_low": (1, 22), "hand_l": (1.3, 12)},
    "neck": {"chest": (1.0, 14), "neck": (1.0, 20), "head": (1.0, 14)},
}
RIGID = {"face": "head", "ears-r": "head", "ears-l": "head", "nose": "head", "mouth": "head",
         "eyebrow-r": "head", "eyebrow-l": "head",
         "eyewhite-r": "eye_r", "irides-r": "eye_r", "eyelash-r": "eye_r",
         "eyewhite-l": "eye_l", "irides-l": "eye_l", "eyelash-l": "eye_l"}
GRID = {"back_hair": 16, "front_hair": 14, "topwear": 7, "bottomwear": 14, "legwear": 22, "footwear": 24,
        "handwear-r": 18, "handwear-l": 18, "neck": 12}

SECONDARY = {"pony1": ("rot+len", 1.5, 0.2, "ponytail"), "pony2": ("rot+len", 1.3, 0.2, "ponytail"),
             "pony3": ("rot+len", 1.15, 0.2, "ponytail"), "tuft": ("rot+len", 2.2, 0.25, "tuft"),
             "bang_r": ("rot+len", 1.9, 0.25, "bangs"), "bang_c": ("rot+len", 1.9, 0.25, "bangs"),
             "bang_l": ("rot+len", 1.9, 0.25, "bangs"),
             "cardigan_r": ("rot+len", 1.6, 0.28, "cardigan"), "cardigan_l": ("rot+len", 1.6, 0.28, "cardigan"),
             "skirt_r": ("rot+len", 1.4, 0.24, "skirt"), "skirt_c": ("rot+len", 1.5, 0.24, "skirt"),
             "skirt_l": ("rot+len", 1.4, 0.24, "skirt")}
# forearm and hand trail the keyed upper arm (overlapping action) -- springs on top of the keys
for _s in ("r", "l"):
    SECONDARY[f"arm_{_s}_low"] = ("rot", 2.0 * (1.03 if _s == "l" else 1.0), 0.32, "forearm")
    SECONDARY["hand_" + _s] = ("rot", 2.6 * (1.03 if _s == "l" else 1.0), 0.30, "hand")
FIXED_GAIN = {"forearm": 0.55, "hand": 0.8}
TARGET_IDLE = {"ponytail": 2.5, "tuft": 2.0, "bangs": 1.4, "cardigan": 1.6, "skirt": 1.4}
_CAP_CALM = {"ponytail": 7.0, "tuft": 5.0, "bangs": 3.5, "cardigan": 4.5, "skirt": 4.5, "forearm": 6.0, "hand": 8.0}
CAP = {"idle": _CAP_CALM, "bust_bounce": _CAP_CALM,
       "jump_cheer": {"ponytail": 22.0, "tuft": 14.0, "bangs": 7.0, "cardigan": 12.0, "skirt": 9.0,
                      "forearm": 22.0, "hand": 28.0}}
LEN_TARGET = {"jump_cheer": {"ponytail": 0.12, "tuft": 0.15, "bangs": 0.06, "cardigan": 0.08, "skirt": 0.10}}


# ─── motion programs (scaled to this figure; feet stay planted) ──────────────

def idle_primary(t, dur=8.0):
    b = R.breath_wave(t, dur / 2)
    sw = lambda lag=0.0: math.sin(R.TAU * t / dur - lag)
    m = {"hip": (0.8 * sw(), 0.6 * b, 0.4 * sw(0.2), 1, 1),
         "spine": (0, 0, 0.6 * sw(0.5), 1, 1),
         "chest": (0.9 * b, 0, 0.7 * sw(0.9) - 0.3 * b, 1, 1),
         "breath": (0, 0, 0, 1 + 0.02 * b, 1 + 0.012 * b),
         "neck": (0, 0, -0.4 * sw(1.3) + 0.3 * b, 1, 1),
         "head": (0, 0, -1.2 * sw(1.7) + 0.6 * math.sin(R.TAU * t / (dur / 2) - 2.0), 1, 1)}
    for s, sg in (("r", 1), ("l", -1)):
        m[f"arm_{s}_up"] = (0, 0, sg * 1.2 * b + 1.0 * sw(1.1), 1, 1)
        m[f"arm_{s}_low"] = (0, 0, sg * 1.0 * b + 0.8 * sw(1.5), 1, 1)
        m["thigh_" + s] = (0, 0, -0.4 * sw(0.2), 1, 1)
    bl = R.blink_value(t, (2.7, 6.1, 6.45))
    for s in ("r", "l"):
        m["eye_" + s] = (0, 0, 0, 1, bl)
    return m


def bust_primary(t, dur=6.4):
    cyc = dur / 2
    u = (t % cyc) / cyc
    h, hd = R.heave(u), R.heave(((t - 0.12) % cyc) / cyc)
    sw = lambda lag=0.0: math.sin(R.TAU * t / dur - lag)
    b = R.breath_wave(t, cyc)
    side = 1.0 if (t % dur) < cyc else -1.0
    kick = side * 1.2 * R.bump(u, 0.58, 0.70, 0.95)
    m = {"hip": (1.2 * sw() + kick, 3.5 * h + 0.4 * b, 0.5 * sw(0.2), 1, 1),   # a small knee-dip bounce
         "spine": (0, 0, 0.9 * sw(0.4), 1, 1),
         "chest": (2.6 * h + 0.6 * b, 0, 1.1 * sw(0.8) - 0.6 * h, 1, 1),
         "breath": (0, 0, 0, 1 + 0.03 * h + 0.012 * b, 1 + 0.015 * h),
         "neck": (0, 0, -0.7 * sw(1.2) + 0.8 * hd, 1, 1),
         "head": (0, 0, -1.4 * sw(1.6) - 1.3 * hd, 1, 1)}
    for s, sg in (("r", 1), ("l", -1)):
        m[f"arm_{s}_up"] = (0, 0, sg * 2.5 * h + 1.0 * sw(1.0), 1, 1)
        m[f"arm_{s}_low"] = (0, 0, sg * 2.0 * hd, 1, 1)
        m["thigh_" + s] = (0, 0, -0.5 * sw(0.2), 1, 1)
    bl = R.blink_value(t, (1.15, 4.6))
    for s in ("r", "l"):
        m["eye_" + s] = (0, 0, 0, 1, bl)
    return m


def herm(t, keys):
    """C1 Hermite through (t, value, slope) keys; clamps outside."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0, m0), (t1, v1, m1) in zip(keys[:-1], keys[1:]):
        if t0 <= t <= t1:
            h = t1 - t0
            x = (t - t0) / h
            return ((2 * x ** 3 - 3 * x ** 2 + 1) * v0 + (x ** 3 - 2 * x ** 2 + x) * h * m0 +
                    (-2 * x ** 3 + 3 * x ** 2) * v1 + (x ** 3 - x ** 2) * h * m1)
    return keys[-1][1]


# Jump timing (s): anticipation crouch -> take-off -> ballistic arc -> landing absorb -> settle -> hold
T_OFF, T_APEX, T_LAND, JUMP_H = 0.74, 1.05, 1.36, 72.0
V0 = 2 * JUMP_H / (T_LAND - T_OFF)                        # take-off speed of a symmetric parabola
HIP_Y = [(0.0, 0, 0), (0.30, 0, 0), (0.62, -26, 0), (T_OFF, 0, V0)]
LAND_Y = [(T_LAND, 0, -V0), (1.52, -22, 0), (1.80, 3, 0), (2.05, 0, 0), (2.8, 0, 0)]
# peak ~76 deg: above ~80 deg the root of a 2-D cutout arm shows above the shoulder line
# (a hand-animated rig would swap in a "raised shoulder" drawing there)
ARM_UP = [(0.0, 0, 0), (0.30, 0, 0), (0.62, 14, 0), (0.82, -52, -300), (1.00, -76, 0), (1.30, -73, 0),
          (1.52, -53, 90), (1.85, -14, 90), (2.2, 3, 0), (2.45, 0, 0), (2.8, 0, 0)]
ARM_LOW = [(0.0, 0, 0), (0.30, 0, 0), (0.62, 10, 0), (0.85, -15, 0), (1.05, -28, 0), (1.35, -22, 0),
           (1.60, -6, 0), (2.0, 4, 0), (2.4, 0, 0), (2.8, 0, 0)]


def jump_primary(t, dur=2.8):
    if t <= T_OFF:
        y = herm(t, HIP_Y)
    elif t < T_LAND:
        tt = t - T_OFF
        g = V0 / (T_APEX - T_OFF)
        y = V0 * tt - 0.5 * g * tt * tt                     # ballistic
    else:
        y = herm(t, LAND_Y)
    air = R.bump(t, T_OFF, T_APEX, T_LAND)                  # 0..1..0 over the flight
    tuck = 20.0 * R.bump(t, T_OFF + 0.06, T_APEX, T_LAND - 0.02)
    crouch = max(0.0, -y) / 26.0                            # 0..1 while knees are bent
    drift = 6.0 * air                                        # slight sideways travel -> asymmetric swing
    # left arm a touch later and smaller than the right: no mirror-perfect puppet motion
    au_r, al_r = herm(t, ARM_UP), herm(t, ARM_LOW)
    au_l, al_l = 0.97 * herm(t - 0.05, ARM_UP), 0.94 * herm(t - 0.05, ARM_LOW)
    m = {"hip": (drift, y, 2.0 * air - 1.5 * crouch, 1, 1),
         "spine": (0, 0, -1.5 * air + 1.0 * crouch, 1, 1),
         "chest": (0, 0, 2.0 * air, 1, 1),
         "breath": (0, 0, 0, 1 - 0.035 * crouch + 0.03 * air, 1 + 0.02 * crouch),
         "neck": (0, 0, -2.0 * air + 1.5 * crouch, 1, 1),
         "head": (0, 0, 6.0 * air - 3.0 * crouch, 1, 1)}
    for s, sg, au, al in (("r", 1, au_r, al_r), ("l", -1, au_l, al_l)):
        raise_ = max(0.0, -au) / 76.0                       # 0 at rest .. 1 arms up
        # elbow arcs: bends most mid-swing, nearly straight at the top
        arc = -32.0 * math.sin(math.pi * min(1.0, raise_)) if au < 0 else 0.0
        m["shoulder_" + s] = (0, 0, sg * -3.0 * raise_, 1, 1)          # slight shoulder lift only
        m[f"arm_{s}_up"] = (0, 0, sg * au, 1, 1)
        m[f"arm_{s}_low"] = (0, 0, sg * (al + arc), 1, 1)
        m["hand_" + s] = (0, 0, sg * 8.0 * raise_, 1, 1)               # open, relaxed wrist at the top
    # legs: compress to keep the feet planted while the hip is below rest; in the air the
    # legs ride with the hip and the shins foreshorten a little (tuck)
    f = 1.0 - max(0.0, -y) / LEG["len"]
    widen = 1.0 + 1.5 * (1.0 - f)                           # foreshortened thigh reads wider
    tuck_f = 1.0 - 0.12 * tuck / 20.0
    for s in ("r", "l"):
        m["thigh_" + s] = (0, 0, 0, f, widen)
        m["shin_" + s] = (0, 0, 0, f * tuck_f, 1.0 + 0.5 * (1.0 - f))
    bl = R.blink_value(t, (1.50,))
    for s in ("r", "l"):
        m["eye_" + s] = (0, 0, 0, 1, bl)
    return m


def weight_map(layers, bones, atts, lm, path):
    names = [b.name for b in bones]
    a = atts["topwear"]
    v = a["vertices"]
    cols = {"breast_r": (230, 60, 60), "breast_r_tip": (255, 210, 40),
            "breast_l": (60, 90, 230), "breast_l_tip": (40, 220, 230),
            "cardigan_r": (160, 90, 220), "cardigan_l": (160, 90, 220)}
    lay = R.despeckle(layers["topwear"])
    x0, y0, x1, y1 = R.bbox(lay[:, :, 3], 4)
    x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
    img = np.full(lay.shape[:2] + (3,), 245, np.uint8)
    m = lay[:, :, 3] > 10
    img[m] = (lay[m, :3] * 0.5 + 120).astype(np.uint8)
    i = 0
    for vi in range(len(a["uvs"]) // 2):
        n = v[i]; i += 1
        c = np.zeros(3)
        for _ in range(n):
            bi, _, _, w = v[i:i + 4]; i += 4
            c += np.array(cols.get(names[bi], (200, 200, 200))) * w
        px, py = int(x0 + a["uvs"][2 * vi] * a["width"]), int(y0 + a["uvs"][2 * vi + 1] * a["height"])
        cv2.circle(img, (px, py), 2, tuple(int(x) for x in c), -1)
    T = lm["topwear"]
    crop = img[T[1] - 10:T[3] + 10, T[0] - 10:T[2] + 10]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def main():
    # re-point the shared machinery at this character
    R.WEIGHTS, R.RIGID, R.GRID = WEIGHTS, RIGID, GRID
    R.SECONDARY, R.TARGET_IDLE, R.CAP, R.LEN_TARGET = SECONDARY, TARGET_IDLE, CAP, LEN_TARGET
    R.FIXED_GAIN = FIXED_GAIN
    R.PROGRAMS = {"idle": (idle_primary, 8.0), "bust_bounce": (bust_primary, 6.4),
                  "jump_cheer": (jump_primary, 2.8)}
    R.BUST_TARGET = {"idle": 0.05, "bust_bounce": 0.22, "jump_cheer": 0.30}
    R.bust_R = lambda lm: lm["bust_R"]
    R.bust_centre = lambda lm, s: lm["bust_" + s]

    _, layers = R.load_layers()
    lm = landmarks(layers)
    Rb = lm["bust_R"]
    for side in ("r", "l"):
        WEIGHTS["topwear"]["deltoid_" + side] = (0.9, 0.24 * lm["arm_width_" + side])
        WEIGHTS["topwear"]["breast_" + side] = (1.5, 0.34 * Rb)
        WEIGHTS["topwear"]["breast_" + side + "_tip"] = (2.8, 0.50 * Rb)
    bones = build_skeleton(lm)
    bj = R.bone_json(bones)
    atts, slots, att_stats = R.make_attachments(layout, layers, bones)
    anims, anim_report = R.build_animations(bones, bj, lm)
    phys = R.physics_constraints(bones)
    for name, a in anims.items():
        val = 1 if name.endswith("_physics") else 0
        anims[name] = {"bones": a, "physics": {p["name"]: {"mix": [{"time": 0, "value": val}]} for p in phys}}
    config = {"skeleton": {"name": R.NAME, "width": W, "height": H}, "bones": bj, "slots": slots,
              "attachments": atts, "animations": [], "custom_animations": anims}
    spine = R.bsj.build_spine_json(config)
    spine["skeleton"].update({"fps": 30, "referenceScale": 100, "images": "./images/"})
    spine["physics"] = phys
    shutil.rmtree(R.SPINE_DIR, ignore_errors=True)
    os.makedirs(R.SPINE_DIR)
    write_json(spine, os.path.join(R.SPINE_DIR, R.NAME + ".json"))
    rc = os.system(f'{sys.executable} "{os.path.join(HERE, "vendor", "make_atlas.py")}" '
                   f'--parts "{R.IMG_DIR}" --output "{R.SPINE_DIR}" --name {R.NAME} --padding 2 > /dev/null')
    assert rc == 0, "make_atlas failed"
    shutil.copytree(R.IMG_DIR, os.path.join(R.SPINE_DIR, "images"))
    weight_map(layers, bones, atts, lm, os.path.join(OUT, "verification", "bust_weight_map.png"))
    T = lm["white_top"]
    pad = int(1.1 * lm["bust_R"])
    rig_report = {"landmarks_px": lm, "bust_closeup_box": [T[0] - pad, T[1] - pad, T[2] + pad, T[3] + pad // 2],
                  "bones": [{"name": b.name, "parent": b.parent, "head_spine": [round(v, 1) for v in b.head],
                             "angle": round(b.angle, 1), "length": round(b.length, 1)} for b in bones],
                  "attachments": att_stats, "physics_constraints": [p["name"] for p in phys],
                  "animations": anim_report}
    write_json(json.loads(json.dumps(rig_report, default=float)), os.path.join(OUT, "work", "rig_report.json"))
    print(f"bones={len(bones)} slots={len(slots)} physics={len(phys)} anims={list(anims)}")
    print(json.dumps(json.loads(json.dumps(anim_report, default=float)))[:1500])


if __name__ == "__main__":
    main()

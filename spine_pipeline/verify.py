#!/usr/bin/env python3
"""
Stage 4 — previews + verification.

  * preview.html          official Spine Web Player (CDN) via the skill's generate_spine_player.py
  * preview_offline.html  same page with the spine-player 4.2 runtime inlined (works without network)
  * verification/         metrics against the reference CG, GIFs, contact sheets

Expects frames rendered by verify/render_frames.mjs (official spine-webgl runtime)
in output/spine_character/verification/frames.

Usage: python3 spine_pipeline/verify.py [--player-dir node_modules/@esotericsoftware/spine-player/dist]
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT, REFERENCE, write_json  # noqa: E402

SPINE = os.path.join(OUT, "spine")
VER = os.path.join(OUT, "verification")
FR = os.path.join(VER, "frames")


def white(path):
    im = np.array(Image.open(path).convert("RGBA")).astype(np.float32)
    a = im[:, :, 3:] / 255.0
    return im[:, :, :3] * a + 255 * (1 - a)


def ref_white():
    im = np.array(Image.open(REFERENCE).convert("RGBA")).astype(np.float32)
    a = im[:, :, 3:] / 255.0
    a = np.where(a < 0.5, 0, a)          # reference background haze -> transparent
    return im[:, :, :3] * a + 255 * (1 - a), a[:, :, 0] > 0


def cmp(a, b, mask=None):
    d = np.abs(a - b)
    mse = float((d ** 2).mean())
    out = {"mae": round(float(d.mean()), 3), "psnr_db": round(10 * math.log10(255 ** 2 / max(mse, 1e-9)), 2)}
    if mask is not None:
        out["mae_in_silhouette"] = round(float(d[mask].mean()), 3)
    return out


def ssim_gray(a, b):
    import cv2
    a, b = a.mean(2), b.mean(2)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    s_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    s_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2))
    return round(float(m.mean()), 4)


def previews(player_dir):
    gen = os.path.join(HERE, "vendor", "generate_spine_player.py")
    html = os.path.join(OUT, "preview.html")
    subprocess.check_call([sys.executable, gen, "--skeleton", os.path.join(SPINE, "character.json"),
                           "--atlas", os.path.join(SPINE, "character.atlas"),
                           "--atlas-image", os.path.join(SPINE, "character.png"),
                           "--output", html, "--animation", "bust_bounce", "--background", "#f4f4f8ff",
                           "--title", "SpineAI character — idle"], stdout=subprocess.DEVNULL)
    if player_dir and os.path.isdir(player_dir):
        page = open(html, encoding="utf-8").read()
        js = open(os.path.join(player_dir, "iife", "spine-player.min.js"), encoding="utf-8").read()
        css = open(os.path.join(player_dir, "spine-player.css"), encoding="utf-8").read()
        page = page.replace('<script src="https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/iife/spine-player.js"></script>',
                            "<script>" + js.replace("</script>", "<\\/script>") + "</script>")
        page = page.replace('<link rel="stylesheet" href="https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/spine-player.css">',
                            "<style>" + css + "</style>")
        assert 'src="https://unpkg.com' not in page and 'href="https://unpkg.com' not in page, "offline preview still loads from the CDN"
        open(os.path.join(OUT, "preview_offline.html"), "w", encoding="utf-8").write(page)
        return True
    return False


def gif(frames, path, size=300, fps=10, src_fps=30):
    frames = frames[::max(1, round(src_fps / fps))]
    ims = [Image.open(f).convert("RGB").resize((size, size), Image.LANCZOS) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)


def sheet(frames, path, n=8, size=256, label=""):
    pick = [frames[int(i * len(frames) / n)] for i in range(n)]
    out = Image.new("RGB", (n * size, size + 18), (30, 30, 36))
    d = ImageDraw.Draw(out)
    for k, f in enumerate(pick):
        out.paste(Image.open(f).convert("RGB").resize((size, size), Image.LANCZOS), (k * size, 18))
        d.text((k * size + 4, 3), f"{label} #{k}", fill="white")
    out.save(path)


def motion_overlay(a, b, path):
    """Red = frame A, cyan = frame B; grey where identical (shows what moves)."""
    ga, gb = white(a).mean(2), white(b).mean(2)
    Image.fromarray(np.stack([ga, gb, gb], 2).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-dir", default=None)
    args = ap.parse_args()
    offline = previews(args.player_dir)

    refw, refmask = ref_white()
    setup = white(os.path.join(FR, "setup_pose.png"))
    stage1 = white(os.path.join(OUT, "stage1_assembly", "final_composite.png"))
    info = json.load(open(os.path.join(FR, "runtime_info.json")))
    rep = {"runtime": info,
           "setup_pose_vs_reference": {**cmp(setup, refw, refmask), "ssim": ssim_gray(setup, refw)},
           "setup_pose_vs_stage1_composite": cmp(setup, stage1),
           "animations": {}}
    # what the old layout.json produced (official position_parts output shipped with the assets)
    old = os.path.join(VER, "baseline_layout_json_composite.png")
    if os.path.exists(old):
        ow = white(old)
        rep["baseline_layout_json_vs_reference"] = {**cmp(ow, refw, refmask), "ssim": ssim_gray(ow, refw)}
    for anim in [a for a, _ in info["animations"]]:
        frames = sorted(glob.glob(os.path.join(FR, anim, "*.png")))
        diffs = [float(np.abs(white(f) - setup).mean()) for f in frames]
        to_ref = [cmp(white(f), refw)["psnr_db"] for f in frames]
        seam = float(np.abs(white(frames[0]) - white(frames[-1])).mean())
        rep["animations"][anim] = {"frames": len(frames), "mean_abs_change_vs_setup": [round(min(diffs), 2), round(max(diffs), 2)],
                                   "psnr_vs_reference_range": [min(to_ref), max(to_ref)],
                                   "loop_seam_mae_last_vs_first": round(seam, 3)}
        gif(frames, os.path.join(VER, f"{anim}.gif"))
        sheet(frames, os.path.join(VER, f"{anim}_sheet.png"), label=anim)
        peak = int(np.argmax(diffs))
        motion_overlay(frames[0], frames[peak], os.path.join(VER, f"{anim}_motion_overlay.png"))
    # bust showcase videos at full frame rate (frame dump is 30 fps), looped 3x
    bb = os.path.join(FR, "bust_bounce")
    if os.path.isdir(bb) and shutil.which("ffmpeg"):
        fps = info.get("frame_fps", 30)
        box = (245, 225, 515, 475)                      # character 1 default
        rr = os.path.join(OUT, "work", "rig_report.json")
        if os.path.exists(rr):
            box = tuple(json.load(open(rr)).get("bust_closeup_box", box))
        bw, bh = box[2] - box[0], box[3] - box[1]
        sc = 600 / bh
        for name, vf in (("bust_bounce.mp4", "scale=768:768"),
                         ("bust_bounce_closeup.mp4",
                          f"crop={bw}:{bh}:{box[0]}:{box[1]},scale={2 * round(bw * sc / 2)}:600:flags=lanczos")):
            subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-stream_loop", "2", "-framerate", str(fps),
                                   "-i", os.path.join(bb, "f%03d.png"), "-vf", vf + ",format=yuv420p",
                                   "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart",
                                   os.path.join(VER, name)])
        frames = sorted(glob.glob(os.path.join(bb, "*.png")))[::2]
        ims = [Image.open(f).convert("RGB").crop(box) for f in frames]
        ims[0].save(os.path.join(VER, "bust_bounce_closeup.gif"), save_all=True, append_images=ims[1:],
                    duration=int(round(2000 / fps)), loop=0, optimize=True)
    hh = refw.shape[0]
    side = np.concatenate([refw, np.full((hh, 8, 3), 40), setup, np.full((hh, 8, 3), 40),
                           np.clip(np.abs(setup - refw) * 3, 0, 255)], 1)
    Image.fromarray(side.astype(np.uint8)).save(os.path.join(VER, "setup_vs_reference.png"))
    rep["preview_offline_written"] = offline
    write_json(rep, os.path.join(VER, "verification_report.json"))
    print(json.dumps({k: v for k, v in rep.items() if k != "runtime"}, indent=1))


if __name__ == "__main__":
    main()

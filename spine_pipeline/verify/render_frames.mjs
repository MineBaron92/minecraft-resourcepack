// Render the exported Spine skeleton with the official spine-webgl 4.2 runtime
// in headless Chromium and dump PNG frames for verify.py.
//
//   cd <dir with node_modules containing @esotericsoftware/spine-webgl@4.2 and playwright-core>
//   node spine_pipeline/verify/render_frames.mjs <spine_dir> <out_dir> [chromium_path]
import fs from "fs";
import os from "os";
import path from "path";
import { createRequire } from "module";
import { fileURLToPath } from "url";

const require = createRequire(path.join(process.cwd(), "noop.js"));
const { chromium } = require("playwright-core");
const here = path.dirname(fileURLToPath(import.meta.url));
const [spineDir, outDir, chromePath] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "spine-verify-"));
fs.copyFileSync(path.join(here, "render_page.html"), path.join(tmp, "index.html"));
fs.copyFileSync(require.resolve("@esotericsoftware/spine-webgl/dist/iife/spine-webgl.js"), path.join(tmp, "spine-webgl.js"));

const json = fs.readFileSync(path.join(spineDir, "character.json"), "utf8");
const atlas = fs.readFileSync(path.join(spineDir, "character.atlas"), "utf8");
const png = "data:image/png;base64," + fs.readFileSync(path.join(spineDir, "character.png")).toString("base64");

const browser = await chromium.launch({ executablePath: chromePath || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 768, height: 768 } });
const errors = [];
page.on("pageerror", e => errors.push(String(e)));
page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
await page.goto("file://" + path.join(tmp, "index.html"));
const info = await page.evaluate(([j, a, p]) => window.setup(j, a, p), [json, atlas, png]);
const save = (dataUrl, name) => fs.writeFileSync(path.join(outDir, name), Buffer.from(dataUrl.split(",")[1], "base64"));

save(await page.evaluate(() => window.renderSetup()), "setup_pose.png");
const fps = 30;
for (const [anim, dur] of info.animations) {
  const times = [];
  for (let i = 0; i < Math.round(dur * fps); i++) times.push(i / fps);
  const frames = await page.evaluate(([n, t]) => window.renderAnim(n, t, 2), [anim, times]);
  fs.mkdirSync(path.join(outDir, anim), { recursive: true });
  frames.forEach((f, i) => save(f, path.join(anim, `f${String(i).padStart(3, "0")}.png`)));
}
fs.writeFileSync(path.join(outDir, "runtime_info.json"), JSON.stringify({ ...info, errors, frame_fps: fps }, null, 2));
await browser.close();
console.log(JSON.stringify({ ...info, errors }));

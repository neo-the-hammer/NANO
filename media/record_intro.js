// Record media/intro.html to media/intro.webm (and intro.mp4 with ffmpeg) using Playwright.
//   node media/record_intro.js
// Chromium plays the animation once in real time at 1280x720; the page's
// controls are hidden by the #record hash.
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path"), os = require("os"), { execSync } = require("child_process");

(async () => {
  const here = __dirname;
  const body = fs.readFileSync(path.join(here, "intro.html"), "utf8");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "intro-"));
  const page = path.join(tmp, "intro.html");
  fs.writeFileSync(page, "<!doctype html><html><head><meta charset='utf-8'>" +
    "<meta name='viewport' content='width=device-width,initial-scale=1'></head><body>" + body + "</body></html>");
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 },
    recordVideo: { dir: tmp, size: { width: 1280, height: 720 } } });
  const p = await ctx.newPage();
  await p.goto("file://" + page + "#record");
  const dur = await p.evaluate(() => window.__introDuration || 26);
  for (const t of [3, 8, 14, 20, 25]) {          // stills for a quick look
    await p.waitForTimeout((t - (t === 3 ? 0 : [3, 8, 14, 20, 25][[3, 8, 14, 20, 25].indexOf(t) - 1])) * 1000);
    await p.screenshot({ path: path.join(tmp, "still_" + t + ".png") });
  }
  await p.waitForTimeout((dur - 25) * 1000 + 300);
  const video = await p.video().path();
  await ctx.close();
  await browser.close();
  const webm = path.join(here, "intro.webm");
  fs.copyFileSync(video, webm);
  try {
    execSync(`ffmpeg -y -loglevel error -i "${webm}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "${path.join(here, "intro.mp4")}"`);
  } catch (e) { console.log("ffmpeg mp4 conversion failed; intro.webm is still there"); }
  console.log("stills in " + tmp);
  console.log("wrote " + webm);
})();

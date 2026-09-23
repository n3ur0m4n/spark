import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
const { chromium } = await import(
  process.env.SPARK_PLAYWRIGHT_MODULE || "playwright"
);
const { PNG } = await import(process.env.SPARK_PNG_MODULE || "pngjs");
const base =
  process.argv[2] || "http://127.0.0.1:5310/examples/webgpu-validation/";
const output = process.argv[3] || "results/native-composite";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  channel: process.env.SPARK_BROWSER_CHANNEL || "chrome",
  headless: true,
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(String(error)));
const results = { browser: browser.version(), checks: [], passed: false };
const check = (name, evidence) => results.checks.push({ name, ...evidence });
async function load(query) {
  await page.goto(`${base}?${query}`);
  await page.waitForFunction(() => window.__sparkValidation, null, {
    timeout: 120000,
  });
  const ready = await page.evaluate(() => window.__sparkValidation);
  assert.equal(ready.ready, true, JSON.stringify(ready));
  return ready;
}
async function screenshot(name) {
  const png = PNG.sync.read(
    await page
      .locator("canvas")
      .screenshot({ path: `${output}/${name}.png`, omitBackground: true }),
  );
  const offset =
    (Math.floor(png.height / 2) * png.width + Math.floor(png.width / 2)) * 4;
  return {
    pixel: Array.from(png.data.slice(offset, offset + 4)),
    corner: Array.from(png.data.slice(0, 4)),
    width: png.width,
    height: png.height,
  };
}
try {
  await load("midgray");
  results.adapter = await page.evaluate(() => {
    const info = window.validation.renderer.backend.device.adapterInfo;
    return {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    };
  });
  assert(
    !/swiftshader|llvmpipe|software/i.test(JSON.stringify(results.adapter)),
    "A real GPU is required",
  );
  let linear = await page.evaluate(() => window.validation.linearPixel());
  assert(linear[0] > 0.95 && linear[1] < 0.01 && linear[2] < 0.01);
  check("opaque-depth-occlusion", {
    linear,
    screenshot: await screenshot("occlusion"),
  });
  await page.evaluate(async () => {
    const v = window.validation;
    v.box.visible = false;
    await v.draw();
  });
  linear = await page.evaluate(() => window.validation.linearPixel());
  assert(linear.slice(0, 3).every((v) => v > 0.2 && v < 0.23));
  const opaqueGray = await screenshot("gray");
  assert(opaqueGray.pixel.slice(0, 3).every((v) => Math.abs(v - 128) <= 3));
  check("linear-gray-and-final-color", { linear, screenshot: opaqueGray });
  await page.evaluate(async () => {
    const v = window.validation;
    v.renderer.setSize(640, 480);
    v.camera.updateProjectionMatrix();
    await v.draw();
  });
  linear = await page.evaluate(() => window.validation.linearPixel());
  assert(linear.slice(0, 3).every((v) => v > 0.2 && v < 0.23));
  check("resize", { linear });
  const callback = await page.evaluate(async () => {
    const v = window.validation;
    const freeBefore = v.spark.accumulators.length;
    let uploads = 0;
    const owner = v.spark.native?.backend ?? v.spark;
    const method = v.spark.native ? "generateNative" : "uploadSplatsWebGPU";
    const original = owner[method];
    owner[method] = function (...args) {
      uploads++;
      return original.apply(this, args);
    };
    v.splats.onFrame = ({ mesh }) => mesh.scale.set(2, 1, 1);
    let rejected;
    try {
      await v.spark.renderAsync(v.scene, v.camera);
    } catch (error) {
      rejected = String(error);
    }
    const freeAfter = v.spark.accumulators.length;
    v.splats.onFrame = undefined;
    v.splats.scale.set(1, 1, 1);
    owner[method] = original;
    await v.draw();
    return { rejected, uploads, freeBefore, freeAfter, errors: v.errors };
  });
  assert.match(callback.rejected, /nonuniform/);
  assert.equal(callback.uploads, 0);
  assert.equal(callback.freeBefore, callback.freeAfter);
  assert.deepEqual(callback.errors, []);
  check("callback-unsupported-before-upload-and-recovery", callback);
  const transparentMesh = await page.evaluate(async () => {
    const v = window.validation;
    v.box.visible = true;
    v.box.material.transparent = true;
    try {
      await v.spark.renderAsync(v.scene, v.camera);
      return null;
    } catch (error) {
      return String(error);
    } finally {
      v.box.material.transparent = false;
      v.box.visible = false;
    }
  });
  assert.match(transparentMesh, /transparent or transmissive/);
  check("unsupported-transparent-mesh-rejected", { message: transparentMesh });
  const disposed = await page.evaluate(async () => {
    const v = window.validation;
    v.spark.dispose();
    v.spark.dispose();
    try {
      await v.spark.renderAsync(v.scene, v.camera);
      return null;
    } catch (error) {
      return String(error);
    }
  });
  assert.match(disposed, /disposed/);
  check("idempotent-dispose-and-render-rejection", { message: disposed });
  await load("midgray&transparent");
  const transparent = await screenshot("transparent-gray");
  linear = await page.evaluate(() => window.validation.linearPixel());
  assert(
    transparent.pixel
      .slice(0, 3)
      .every((v, i) => Math.abs(v - opaqueGray.pixel[i]) <= 2),
  );
  assert(Math.abs(transparent.pixel[3] - 128) <= 2);
  assert.deepEqual(transparent.corner, [0, 0, 0, 0]);
  assert(linear[3] > 0.49 && linear[3] < 0.51);
  check("straight-color-through-transparent-canvas", {
    linear,
    screenshot: transparent,
  });
  // Changing the host settings at runtime must update the final conversion and
  // restore those settings after submission; compare straight colors at alpha 1/2.
  const changed = await page.evaluate(async () => {
    const v = window.validation;
    const { ACESFilmicToneMapping } = v.THREE;
    v.renderer.toneMapping = ACESFilmicToneMapping;
    await v.draw();
    return {
      toneMapping: v.renderer.toneMapping,
      colorSpace: v.renderer.outputColorSpace,
      errors: v.errors,
    };
  });
  assert.equal(changed.toneMapping, 4);
  assert.equal(changed.colorSpace, "srgb");
  assert.deepEqual(changed.errors, []);
  const transparentTone = await screenshot("transparent-tone-mapped");
  await load("midgray&toneMapping");
  await page.evaluate(async () => {
    const v = window.validation;
    v.box.visible = false;
    await v.draw();
  });
  const opaqueTone = await screenshot("opaque-tone-mapped");
  assert(
    transparentTone.pixel
      .slice(0, 3)
      .every((v, i) => Math.abs(v - opaqueTone.pixel[i]) <= 2),
  );
  assert(opaqueTone.pixel[0] > opaqueGray.pixel[0] + 5);
  check("tone-map-before-premultiplication-and-settings-restoration", {
    changed,
    transparent: transparentTone,
    opaque: opaqueTone,
  });
  const interrupted = await page.evaluate(async () => {
    const v = window.validation;
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    v.spark.onBeforeRender = () => gate;
    const render = v.spark.renderAsync(v.scene, v.camera);
    v.spark.dispose();
    release();
    let rejection;
    try {
      await render;
    } catch (error) {
      rejection = String(error);
    }
    const hostScene = new v.THREE.Scene();
    hostScene.background = new v.THREE.Color(0xff0000);
    v.renderer.render(hostScene, v.camera);
    await v.renderer.backend.device.queue.onSubmittedWorkDone();
    return { rejection, errors: v.errors };
  });
  assert.match(interrupted.rejection, /disposed/);
  assert.deepEqual(interrupted.errors, []);
  check("dispose-during-render-rejects-and-preserves-host-device", interrupted);
  const forced = await load("forceWebGL");
  assert.equal(forced.capabilities.backend, "webgpu-force-webgl");
  assert.match(forced.rejected, /forceWebGL/);
  assert.deepEqual(forced.errors, []);
  check("forceWebGL-explicit-rejection", { result: forced });
  assert.deepEqual(errors, []);
  results.passed = true;
} finally {
  results.errors = errors;
  await writeFile(`${output}/result.json`, JSON.stringify(results, null, 2));
  await browser.close();
}
console.log(JSON.stringify(results));

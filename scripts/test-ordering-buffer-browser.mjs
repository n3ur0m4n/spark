/** Optional real-browser ownership test. See test/browser/README.md. */
import assert from "node:assert/strict";
const { chromium } = await import(
  process.env.SPARK_PLAYWRIGHT_MODULE || "playwright"
);
const url =
  process.argv[2] ||
  "http://127.0.0.1:5311/test/browser/ordering-buffer-pool.html";
const browser = await chromium.launch({ channel: "chrome", headless: true });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(url);
  await page.waitForFunction(() => window.__sparkValidation, null, {
    timeout: 120000,
  });
  const initial = await page.evaluate(() => window.__sparkValidation);
  assert.equal(initial.ready, true, JSON.stringify(initial));
  const adapter = await page.evaluate(() => {
    const gl = window.validation.renderer.getContext();
    const extension = gl.getExtension("WEBGL_debug_renderer_info");
    return extension
      ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)
      : null;
  });
  assert(
    adapter && !/swiftshader|llvmpipe|softpipe|software/i.test(adapter),
    "real GPU identity required",
  );
  const before = await page.evaluate(() => window.validation.readOrdering());
  assert.deepEqual(
    before.gpu,
    before.cpu,
    "committed CPU order must match actual GPU texture",
  );
  await page.evaluate(() => window.validation.restore());
  const after = await page.evaluate(() => window.validation.readOrdering());
  assert.deepEqual(
    after.gpu,
    before.gpu,
    "latest ordering must survive context restoration",
  );
  const stats = await page.evaluate(() => window.validation.stats());
  assert.equal(stats.allocations, 2);
  assert(stats.reuses >= 10);
  assert.deepEqual(errors, []);
  assert.deepEqual(await page.evaluate(() => window.validation.errors), []);
  console.log(
    JSON.stringify({
      passed: true,
      browser: browser.version(),
      adapter,
      ...stats,
      coverage:
        "Ordering buffer ownership and GPU texture restoration only. Full accumulator restoration is a separate existing upstream limitation.",
    }),
  );
} finally {
  await browser.close();
}

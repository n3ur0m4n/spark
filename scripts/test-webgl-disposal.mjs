import assert from "node:assert/strict";
import { chromium } from "playwright";
import { createServer } from "vite";
const server = await createServer({
  optimizeDeps: { entries: ["test/browser/webgl-disposal.html"] },
  server: { host: "127.0.0.1", port: 0 },
  logLevel: "error",
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/test/browser/webgl-disposal.html`,
  );
  await page.waitForFunction(() => window.disposalResult);
  const result = await page.evaluate(() => window.disposalResult);
  console.log(JSON.stringify(result, null, 2));
  assert(result.passed, JSON.stringify(result));
  assert.deepEqual(errors, []);
} finally {
  await browser?.close();
  await server.close();
}

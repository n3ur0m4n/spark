import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium } from "playwright";
import { createServer } from "vite";
import { checkNativeRetention } from "./check-native-retention.mjs";

const output = resolve(
  process.env.SPARK_BROWSER_RESULTS ||
    `test-results/native/${new Date().toISOString().replace(/[:.]/g, "-")}`,
);
await mkdir(output, { recursive: true });
const server = await createServer({
  server: { host: "127.0.0.1", port: 0 },
  logLevel: "error",
});
let browser;
const results = [];
async function run(script, args) {
  await new Promise((resolve, reject) => {
    const process = spawn(globalThis.process.execPath, [script, ...args], {
      stdio: "inherit",
    });
    process.on("error", reject);
    process.on("exit", (code) =>
      code === 0 ? resolve() : reject(Error(`${script} exited ${code}`)),
    );
  });
}
try {
  await server.listen();
  const address = server.httpServer.address();
  assert(address && typeof address !== "string");
  const base = `http://127.0.0.1:${address.port}/test/browser/`;
  browser = await chromium.launch({
    channel: process.env.SPARK_BROWSER_CHANNEL || "chrome",
    headless: true,
  });
  for (const name of [
    "native-lifecycle",
    "webgl-disposal",
    "webgl-radial-invalidation",
    "native-counts",
    "native-device",
    "native-features",
    "native-host-modes",
    "native-visibility",
    "native-retention",
    "native-resize-inflight",
    "native-multiple-hosts",
    "native-encoding",
    "native-sh-generation",
    "native-radix-fallback",
  ]) {
    const page = await browser.newPage();
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(String(error)));
    await page.goto(`${base}${name}.html`);
    if (name === "native-retention") {
      const result = await checkNativeRetention(page);
      await page.evaluate((result) => {
        window.lifecycleResult = result;
      }, result);
    }
    await page.waitForFunction(() => window.lifecycleResult, null, {
      timeout: 120000,
    });
    const result = await page.evaluate(() => window.lifecycleResult);
    results.push({ name, result, pageErrors });
    await writeFile(
      `${output}/${name}.json`,
      JSON.stringify(
        { browser: browser.version(), result, pageErrors },
        null,
        2,
      ),
    );
    assert.equal(result.passed, true, JSON.stringify(result));
    assert.deepEqual(result.errors, []);
    assert.deepEqual(pageErrors, []);
    console.log(`${name}: ${result.checks.length} checks passed`);
    await page.close();
  }
  await browser.close();
  browser = null;
  await run("scripts/validate-native-browser.mjs", [
    `${base}native-composite.html`,
    `${output}/composite`,
  ]);
  await run("scripts/validate-radix-browser.mjs", [
    `${base}radix-sort.html`,
    `${output}/radix.json`,
  ]);
  console.log(`Native browser results: ${output}`);
} finally {
  await writeFile(
    `${output}/lifecycle-summary.json`,
    JSON.stringify(results, null, 2),
  );
  await browser?.close();
  await server.close();
}

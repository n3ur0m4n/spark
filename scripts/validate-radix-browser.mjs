import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
const { chromium } = await import(
  process.env.SPARK_PLAYWRIGHT_MODULE || "playwright"
);
const url =
  process.argv[2] || "http://127.0.0.1:5310/examples/radix-sort-selftest/";
const output = process.argv[3] || "results/native-radix-new.json";
const browser = await chromium.launch({
  channel: process.env.SPARK_BROWSER_CHANNEL || "chrome",
  headless: true,
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(url);
  await page.waitForFunction(() => window.__sparkRadixSelftest, null, {
    timeout: 120000,
  });
  const result = await page.evaluate(() => window.__sparkRadixSelftest);
  await mkdir(dirname(output), { recursive: true });
  await writeFile(
    output,
    JSON.stringify({ browser: browser.version(), result, errors }, null, 2),
  );
  assert.equal(result.pass, true, JSON.stringify(result));
  assert.equal(result.cases.length, 20);
  assert(result.cases.every((test) => test.pass));
  assert.deepEqual(result.errors, []);
  assert.deepEqual(errors, []);
  assert(
    result.adapter?.vendor &&
      !/swiftshader|llvmpipe|software/i.test(JSON.stringify(result.adapter)),
    "Real GPU required",
  );
  console.log(
    JSON.stringify({
      pass: true,
      cases: result.cases.length,
      directionsPerCase: 2,
      adapter: result.adapter,
    }),
  );
} finally {
  await browser.close();
}

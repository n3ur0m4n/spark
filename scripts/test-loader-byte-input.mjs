import assert from "node:assert/strict";
import { mkdir, readdir, writeFile } from "node:fs/promises";

const { chromium } = await import(
  process.env.SPARK_PLAYWRIGHT_MODULE || "playwright"
);
const url =
  process.argv[2] ||
  "http://127.0.0.1:5323/test/browser/loader-byte-input.html";
const output = process.argv[3];
assert(output, "Specify a new evidence directory as the second argument");
await mkdir(output, { recursive: true });
assert.equal((await readdir(output)).length, 0, "Retain previous evidence");
const browser = await chromium.launch({ channel: "chrome", headless: true });
let result = {};
const errors = [];
try {
  const page = await browser.newPage();
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(url);
  await page.waitForFunction(() => window.__loaderValidation, null, {
    timeout: 120000,
  });
  result = await page.evaluate(() => window.__loaderValidation);
  assert.equal(result.ready, true, JSON.stringify(result));
  assert.equal(result.checks.length, 8);
  assert.equal(result.pass, true, JSON.stringify(result));
  assert.deepEqual(errors, []);
} finally {
  await writeFile(
    `${output}/result.json`,
    JSON.stringify(
      { ...result, url, browser: browser.version(), errors },
      null,
      2,
    ),
  );
  await browser.close();
}
console.log(
  JSON.stringify({
    passed: result.pass,
    checks: result.checks?.length,
    output,
  }),
);

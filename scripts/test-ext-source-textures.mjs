import assert from "node:assert/strict";
import { mkdir, readdir, writeFile } from "node:fs/promises";

const { chromium } = await import(
  process.env.SPARK_PLAYWRIGHT_MODULE || "playwright"
);
const url =
  process.argv[2] ||
  "http://127.0.0.1:5322/test/browser/ext-source-textures.html";
const output = process.argv[3];
assert(output, "Specify a new evidence directory as the second argument");
await mkdir(output, { recursive: true });
assert.equal((await readdir(output)).length, 0, "Retain previous evidence");
const browser = await chromium.launch({ channel: "chrome", headless: true });
const errors = [];
let result = {};
try {
  const page = await browser.newPage();
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(url);
  await page.waitForFunction(() => window.__extTextureValidation, null, {
    timeout: 120000,
  });
  result = await page.evaluate(() => window.__extTextureValidation);
  assert.equal(result.ready, true, JSON.stringify(result));
  assert(
    result.adapter && !/swiftshader|software|llvmpipe/i.test(result.adapter),
    "Real GPU required",
  );
  assert.deepEqual(errors, []);
  assert.equal(result.pass, true, JSON.stringify(result));
} finally {
  await writeFile(
    `${output}/result.json`,
    JSON.stringify(
      { ...result, url, browser: browser.version(), pageErrors: errors },
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

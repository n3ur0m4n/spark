import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { checkNativeRetention } from "./check-native-retention.mjs";

// Serve the npm archive with import maps and a plain HTTP server. No Vite
// aliases or source imports can accidentally substitute checkout code.
const root = fileURLToPath(new URL("..", import.meta.url));
const npmCli = process.env.npm_execpath;
assert(npmCli, "Run via npm run test:package:browser");
const output = resolve(
  process.env.SPARK_PACKAGE_BROWSER_RESULTS ||
    `test-results/package-browser/${new Date().toISOString().replace(/[:.]/g, "-")}`,
);
await mkdir(output, { recursive: true });
const directory = await mkdtemp(join(tmpdir(), "spark-browser-consumer-"));
const results = { passed: false, node: process.version, cases: [] };
let browser;
let server;
async function run(args, cwd = directory) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { cwd });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (data) => {
      stdout += data;
    });
    child.stderr.on("data", (data) => {
      stderr += data;
    });
    child.on("error", reject);
    child.on("exit", (code) =>
      code === 0
        ? resolve(stdout)
        : reject(Error(`${args[0]} exited ${code}\n${stdout}\n${stderr}`)),
    );
  });
}
try {
  const [archive] = JSON.parse(
    await run(
      [npmCli, "pack", "--json", "--pack-destination", directory],
      root,
    ),
  );
  results.archiveSha256 = createHash("sha256")
    .update(await readFile(join(directory, archive.filename)))
    .digest("hex");
  assert(
    !archive.files.some((file) =>
      /\.(?:spz|splat|ply|rad|png|jpe?g|webm|mp4)$/i.test(file.path),
    ),
  );
  await writeFile(
    join(directory, "package.json"),
    JSON.stringify({
      name: "spark-browser-consumer",
      private: true,
      type: "module",
    }),
  );
  server = createServer(async (request, response) => {
    try {
      const path = resolve(
        directory,
        `.${decodeURIComponent(new URL(request.url, "http://localhost").pathname)}`,
      );
      assert(path.startsWith(`${directory}${sep}`));
      const body = await readFile(path);
      response.setHeader(
        "Content-Type",
        {
          ".html": "text/html",
          ".js": "text/javascript",
          ".wasm": "application/wasm",
          ".json": "application/json",
        }[extname(path)] || "application/octet-stream",
      );
      response.end(body);
    } catch {
      response.writeHead(404).end();
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  for (const three of ["0.180.0", "0.186.0"]) {
    await run([
      npmCli,
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--save-exact",
      join(directory, archive.filename),
      `three@${three}`,
    ]);
    browser = await chromium.launch({
      channel: process.env.SPARK_BROWSER_CHANNEL || "chrome",
      headless: true,
    });
    results.browser = browser.version();
    for (const bundle of ["spark.module.js", "spark.module.min.js"]) {
      const imports = {
        three: "/node_modules/three/build/three.module.js",
        "three/webgpu": "/node_modules/three/build/three.webgpu.js",
        "three/tsl": "/node_modules/three/build/three.tsl.js",
        "three/addons/": "/node_modules/three/examples/jsm/",
        "/src/index.ts": `/node_modules/@sparkjsdev/spark/dist/${bundle}`,
      };
      const importMap = `<script type="importmap">${JSON.stringify({ imports })}</script>`;
      for (const name of [
        "native-generation",
        "native-lifecycle",
        "webgl-disposal",
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
        "native-composite",
      ]) {
        const fixture = await readFile(
          join(root, `test/browser/${name}.html`),
          "utf8",
        );
        await writeFile(
          join(directory, `${name}.html`),
          fixture.replace("<head>", `<head>${importMap}`),
        );
      }
      for (const name of [
        "native-lifecycle",
        "webgl-disposal",
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
      ]) {
        const page = await browser.newPage();
        const errors = [];
        const failedResponses = [];
        page.on("pageerror", (error) => errors.push(String(error)));
        page.on("response", (response) => {
          if (response.status() >= 400) failedResponses.push(response.url());
        });
        await page.goto(`${base}/${name}.html`);
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
        const adapter = await page.evaluate(async () => {
          const info = (await navigator.gpu.requestAdapter())?.info;
          return info
            ? {
                vendor: info.vendor,
                architecture: info.architecture,
                device: info.device,
                description: info.description,
              }
            : null;
        });
        const record = {
          three,
          bundle,
          name,
          adapter,
          result,
          errors,
          failedResponses,
        };
        results.cases.push(record);
        await writeFile(
          `${output}/${three}-${bundle}-${name}.json`,
          JSON.stringify(record, null, 2),
        );
        assert(
          adapter &&
            !/swiftshader|llvmpipe|software/i.test(JSON.stringify(adapter)),
          "Hardware WebGPU is required",
        );
        assert.equal(result.passed, true, JSON.stringify(record));
        assert.deepEqual(result.errors, []);
        assert.deepEqual(errors, []);
        assert.deepEqual(failedResponses, []);
        console.log(
          `Packed browser ${three} ${bundle} ${name}: ${result.checks.length} checks`,
        );
        await page.close();
      }
      await run(
        [
          join(root, "scripts/validate-native-browser.mjs"),
          `${base}/native-composite.html`,
          `${output}/${three}-${bundle}-composite`,
        ],
        root,
      );
    }
    await browser.close();
    browser = null;
  }
  results.passed = true;
} catch (error) {
  results.failure = String(error);
  console.error(`Packed browser reproduction retained at ${directory}`);
  throw error;
} finally {
  await browser?.close();
  await new Promise((resolve) => (server ? server.close(resolve) : resolve()));
  await writeFile(`${output}/manifest.json`, JSON.stringify(results, null, 2));
  if (results.passed) await rm(directory, { recursive: true, force: true });
  console.log(`Packed browser results: ${output}`);
}

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const directory = mkdtempSync(join(tmpdir(), "spark-package-consumer-"));
const npmCli = process.env.npm_execpath;
if (!npmCli) throw new Error("Run this check via npm run test:package");
function run(executable, args, cwd = directory) {
  return execFileSync(executable, args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}
try {
  const [archive] = JSON.parse(
    run(
      process.execPath,
      [npmCli, "pack", "--json", "--pack-destination", directory],
      root,
    ),
  );
  assert(
    archive.files.some((file) => file.path === "NOTICE"),
    "Bundled third-party notices must be included in the package",
  );
  assert(
    !archive.files.some((file) =>
      /\.(?:spz|splat|ply|spx|rad|png|jpe?g|webm|mp4)$/i.test(file.path),
    ),
    "Scene assets or captures must not enter the library package",
  );
  writeFileSync(
    join(directory, "package.json"),
    JSON.stringify({
      name: "spark-package-consumer",
      private: true,
      type: "module",
    }),
  );
  writeFileSync(
    join(directory, "consumer.cjs"),
    `const assert = require('node:assert/strict');
const Module = require('node:module');
const originalLoad = Module._load;
Module._load = function(specifier, ...args) {
  assert(!specifier.startsWith('three/'), 'Classic entry eagerly requires an ESM-only Three.js submodule: ' + specifier);
  return originalLoad.call(this, specifier, ...args);
};
const { SparkRenderer, utils } = require('@sparkjsdev/spark');
assert.equal(typeof SparkRenderer, 'function');
assert.equal(utils.toHalf(1), 15360);
`,
  );
  writeFileSync(
    join(directory, "consumer.mjs"),
    `import assert from 'node:assert/strict';
import { SparkRenderer, utils } from '@sparkjsdev/spark';
assert.equal(typeof SparkRenderer, 'function');
assert.equal(utils.toHalf(1), 15360);
`,
  );
  copyFileSync(
    new URL("../test/package/consumer.ts", import.meta.url),
    join(directory, "consumer.ts"),
  );
  assert(
    !process.env.SPARK_PACKAGE_THREE_VERSION ||
      ["0.180.0", "0.186.0"].includes(process.env.SPARK_PACKAGE_THREE_VERSION),
  );
  for (const [three, types] of [
    ["0.180.0", "0.180.0"],
    ["0.186.0", "0.186.0"],
  ].filter(
    ([version]) =>
      !process.env.SPARK_PACKAGE_THREE_VERSION ||
      version === process.env.SPARK_PACKAGE_THREE_VERSION,
  )) {
    run(process.execPath, [
      npmCli,
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--save-exact",
      resolve(directory, archive.filename),
      `three@${three}`,
      `@types/three@${types}`,
      "typescript@5.8.3",
    ]);
    run(process.execPath, ["consumer.cjs"]);
    run(process.execPath, ["consumer.mjs"]);
    run(process.execPath, [
      "node_modules/typescript/bin/tsc",
      "--noEmit",
      "--strict",
      "--target",
      "ES2020",
      "--module",
      "ESNext",
      "--moduleResolution",
      "Bundler",
      "--skipLibCheck",
      "consumer.ts",
    ]);
    console.log(
      `Packed CommonJS, ESM and TypeScript consumer passed: Three.js ${three}; Node ${process.version}`,
    );
  }
  assert(archive.filename);
  rmSync(directory, { recursive: true, force: true });
} catch (error) {
  console.error(`Consumer reproduction retained at ${directory}`);
  if (error.stdout) console.error(error.stdout);
  if (error.stderr) console.error(error.stderr);
  throw error;
}

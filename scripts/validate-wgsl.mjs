import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
const cargo = join(homedir(), ".cargo", "bin", "naga");
const naga = process.env.NAGA_BIN || (existsSync(cargo) ? cargo : "naga");
// This gate is mandatory when invoked; missing Naga must fail, never skip.
execFileSync(naga, ["--version"], { stdio: "inherit" });
const directory = mkdtempSync(join(tmpdir(), "spark-wgsl-"));
const shader = (name) =>
  name === "splatPickVertex"
    ? // Same transform as pickVertexWgsl() in src/shaders-wgsl.ts.
      shader("splatVertex").replace(/^\/\/PICK /gm, "")
    : readFileSync(
        new URL(`../src/shaders/${name}.wgsl`, import.meta.url),
        "utf8",
      );
try {
  for (const name of [
    "generateSourceBuffers",
    "generateSourcePaged",
    "splatVertex",
    "splatFragment",
    "splatPickVertex",
    "splatPickFragment",
    "nativeSort",
    "packOrdering",
    "radixSort4bit",
    "radixSortReorder",
    "prefixSum",
  ]) {
    // Generation is validated once per source accessor variant.
    const text = name.startsWith("generateSource")
      ? `${shader("splatDefines")}\n${shader(name)}\n${shader("generateSplats")}`
      : ([
          "splatVertex",
          "splatFragment",
          "splatPickVertex",
          "splatPickFragment",
        ].includes(name)
          ? `${shader("splatDefines")}\n`
          : "") + shader(name);
    const file = join(directory, `${name}.wgsl`);
    writeFileSync(file, text);
    execFileSync(naga, [file], { stdio: "inherit" });
    console.log(`Validated ${name}`);
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}

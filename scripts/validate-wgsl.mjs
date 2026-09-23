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
  readFileSync(new URL(`../src/shaders/${name}.wgsl`, import.meta.url), "utf8");
try {
  for (const name of ["generateSplats", "splatVertex", "splatFragment"]) {
    const text =
      (["generateSplats", "splatVertex", "splatFragment"].includes(name)
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

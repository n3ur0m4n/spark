import * as THREE from "three";
import { Dyno, unindentLines } from "./base";
import { CovSplat, Gsplat, defineCovSplat, defineGsplat } from "./splats";
import {
  DynoOutput,
  type DynoVal,
  type DynoValue,
  type HasDynoOut,
} from "./value";

export const outputPackedSplat = (
  gsplat: DynoVal<typeof Gsplat>,
  rgbMinMaxLnScaleMinMax: DynoVal<"vec4">,
) => new OutputPackedSplat({ gsplat, rgbMinMaxLnScaleMinMax });

export const outputCovSplat = (
  covsplat: DynoVal<typeof CovSplat>,
  rgbMinMaxLnScaleMinMax: DynoVal<"vec4">,
) => new OutputCovSplat({ covsplat, rgbMinMaxLnScaleMinMax });

export const outputExtendedSplat = (gsplat: DynoVal<typeof Gsplat>) =>
  new OutputExtendedSplat({ gsplat });

export const outputExtCovSplat = (covsplat: DynoVal<typeof CovSplat>) =>
  new OutputExtCovSplat({ covsplat });

export const outputSplatDepth = (
  gsplat: DynoVal<typeof Gsplat>,
  viewCenter: DynoVal<"vec3">,
  viewDir: DynoVal<"vec3">,
  sortRadial: DynoVal<"bool">,
  sortCullCos: DynoVal<"float">,
) =>
  new OutputSplatDepth({
    gsplat,
    viewCenter,
    viewDir,
    sortRadial,
    sortCullCos,
  });

export const outputCovSplatDepth = (
  covsplat: DynoVal<typeof CovSplat>,
  viewCenter: DynoVal<"vec3">,
  viewDir: DynoVal<"vec3">,
  sortRadial: DynoVal<"bool">,
  sortCullCos: DynoVal<"float">,
) =>
  new OutputCovSplatDepth({
    covsplat,
    viewCenter,
    viewDir,
    sortRadial,
    sortCullCos,
  });

export const outputRgba8 = (rgba8: DynoVal<"vec4">) =>
  new OutputRgba8({ rgba8 });

export class OutputPackedSplat extends Dyno<
  { gsplat: typeof Gsplat; rgbMinMaxLnScaleMinMax: "vec4" },
  Record<string, never>
> {
  constructor({
    gsplat,
    rgbMinMaxLnScaleMinMax,
  }: {
    gsplat?: DynoVal<typeof Gsplat>;
    rgbMinMaxLnScaleMinMax?: DynoVal<"vec4">;
  }) {
    super({
      inTypes: { gsplat: Gsplat, rgbMinMaxLnScaleMinMax: "vec4" },
      inputs: { gsplat, rgbMinMaxLnScaleMinMax },
      globals: () => [defineGsplat],
      statements: ({ inputs, outputs }) => {
        const { gsplat, rgbMinMaxLnScaleMinMax } = inputs;
        if (gsplat && rgbMinMaxLnScaleMinMax) {
          return unindentLines(/* glsl */ `
            if (isGsplatActive(${gsplat}.flags)) {
              target = packSplatEncoding(${gsplat}.center, ${gsplat}.scales, ${gsplat}.quaternion, ${gsplat}.rgba, ${rgbMinMaxLnScaleMinMax});
            } else {
              target = uvec4(0u, 0u, 0u, 0u);
            }
          `);
        }
        return ["target = uvec4(0u, 0u, 0u, 0u);"];
      },
    });
  }
}

export class OutputCovSplat extends Dyno<
  { covsplat: typeof CovSplat; rgbMinMaxLnScaleMinMax: "vec4" },
  Record<string, never>
> {
  constructor({
    covsplat,
    rgbMinMaxLnScaleMinMax,
  }: {
    covsplat?: DynoVal<typeof CovSplat>;
    rgbMinMaxLnScaleMinMax?: DynoVal<"vec4">;
  }) {
    super({
      inTypes: { covsplat: CovSplat, rgbMinMaxLnScaleMinMax: "vec4" },
      inputs: { covsplat, rgbMinMaxLnScaleMinMax },
      globals: () => [defineCovSplat],
      statements: ({ inputs }) => {
        const { covsplat, rgbMinMaxLnScaleMinMax } = inputs;
        if (covsplat && rgbMinMaxLnScaleMinMax) {
          return unindentLines(/* glsl */ `
            if (isCovSplatActive(${covsplat}.flags)) {
              target = packSplatCovEncoding(${covsplat}.center, ${covsplat}.rgba, ${covsplat}.xxyyzz, ${covsplat}.xyxzyz, ${rgbMinMaxLnScaleMinMax});
            } else {
              target = uvec4(0u);
            }
          `);
        }
        return ["target = uvec4(0u);"];
      },
    });
  }
}

export class OutputExtendedSplat extends Dyno<
  { gsplat: typeof Gsplat },
  Record<string, never>
> {
  constructor({
    gsplat,
  }: {
    gsplat?: DynoVal<typeof Gsplat>;
  }) {
    super({
      inTypes: { gsplat: Gsplat },
      inputs: { gsplat },
      globals: () => [defineGsplat],
      statements: ({ inputs }) => {
        const { gsplat } = inputs;
        if (gsplat) {
          return unindentLines(/* glsl */ `
            if (isGsplatActive(${gsplat}.flags)) {
              packSplatExt(target, target2, ${gsplat}.center, ${gsplat}.scales, ${gsplat}.quaternion, ${gsplat}.rgba);
            } else {
              target = uvec4(0u);
              target2 = uvec4(0u);
            }
          `);
        }
        return ["target = uvec4(0u);", "target2 = uvec4(0u);"];
      },
    });
  }
}

export class OutputExtCovSplat extends Dyno<
  { covsplat: typeof CovSplat },
  Record<string, never>
> {
  constructor({
    covsplat,
  }: {
    covsplat?: DynoVal<typeof CovSplat>;
  }) {
    super({
      inTypes: { covsplat: CovSplat },
      inputs: { covsplat },
      globals: () => [defineCovSplat],
      statements: ({ inputs }) => {
        const { covsplat } = inputs;
        if (covsplat) {
          return unindentLines(/* glsl */ `
            if (isCovSplatActive(${covsplat}.flags)) {
              packSplatExtCov(target, target2, ${covsplat}.center, ${covsplat}.rgba, ${covsplat}.xxyyzz, ${covsplat}.xyxzyz);
            } else {
              target = uvec4(0u);
              target2 = uvec4(0u);
            }
          `);
        }
        return ["target = uvec4(0u);", "target2 = uvec4(0u);"];
      },
    });
  }
}

class OutputSplatDepth extends Dyno<
  {
    gsplat: typeof Gsplat;
    viewCenter: "vec3";
    viewDir: "vec3";
    sortRadial: "bool";
    sortCullCos: "float";
  },
  Record<string, never>
> {
  constructor({
    gsplat,
    viewCenter,
    viewDir,
    sortRadial,
    sortCullCos,
  }: {
    gsplat: DynoVal<typeof Gsplat>;
    viewCenter: DynoVal<"vec3">;
    viewDir: DynoVal<"vec3">;
    sortRadial: DynoVal<"bool">;
    sortCullCos: DynoVal<"float">;
  }) {
    super({
      inTypes: {
        gsplat: Gsplat,
        viewCenter: "vec3",
        viewDir: "vec3",
        sortRadial: "bool",
        sortCullCos: "float",
      },
      inputs: { gsplat, viewCenter, viewDir, sortRadial, sortCullCos },
      globals: () => [defineGsplat],
      statements: ({ inputs }) => {
        const { gsplat, viewCenter, viewDir, sortRadial, sortCullCos } = inputs;
        if (gsplat && viewCenter && viewDir && sortRadial && sortCullCos) {
          return unindentLines(/* glsl */ `
            float metric = 1.0 / 0.0;
            if (isGsplatActive(${gsplat}.flags)) {
              vec3 center = ${gsplat}.center - ${viewCenter};
              float dist = length(center);
              // Outside the sort cone (sortCullFov): stays +inf, not drawn.
              // Disabled when sortCullCos <= -1; dist 0 is never culled.
              if (dist > 0.0 && dot(center, ${viewDir}) < ${sortCullCos} * dist) {
              } else if (${sortRadial}) {
                metric = dist;
              } else {
                float bias = 100.0; // reduce popping
                metric = dot(center, ${viewDir}) + bias;
              }
            }
            target3 = floatToVec4(metric);
          `);
        }
        return [];
      },
    });
  }
}

class OutputCovSplatDepth extends Dyno<
  {
    covsplat: typeof CovSplat;
    viewCenter: "vec3";
    viewDir: "vec3";
    sortRadial: "bool";
    sortCullCos: "float";
  },
  Record<string, never>
> {
  constructor({
    covsplat,
    viewCenter,
    viewDir,
    sortRadial,
    sortCullCos,
  }: {
    covsplat: DynoVal<typeof CovSplat>;
    viewCenter: DynoVal<"vec3">;
    viewDir: DynoVal<"vec3">;
    sortRadial: DynoVal<"bool">;
    sortCullCos: DynoVal<"float">;
  }) {
    super({
      inTypes: {
        covsplat: CovSplat,
        viewCenter: "vec3",
        viewDir: "vec3",
        sortRadial: "bool",
        sortCullCos: "float",
      },
      inputs: { covsplat, viewCenter, viewDir, sortRadial, sortCullCos },
      globals: () => [defineCovSplat],
      statements: ({ inputs }) => {
        const { covsplat, viewCenter, viewDir, sortRadial, sortCullCos } =
          inputs;
        if (covsplat && viewCenter && viewDir && sortRadial && sortCullCos) {
          return unindentLines(/* glsl */ `
            float metric = 1.0 / 0.0;
            if (isCovSplatActive(${covsplat}.flags)) {
              vec3 center = ${covsplat}.center - ${viewCenter};
              float dist = length(center);
              // Outside the sort cone (sortCullFov): stays +inf, not drawn.
              // Disabled when sortCullCos <= -1; dist 0 is never culled.
              if (dist > 0.0 && dot(center, ${viewDir}) < ${sortCullCos} * dist) {
              } else if (${sortRadial}) {
                metric = dist;
              } else {
                float bias = 100.0; // reduce popping
                metric = dot(center, ${viewDir}) + bias;
              }
            }
            target3 = floatToVec4(metric);
          `);
        }
        return [];
      },
    });
  }
}

export class OutputRgba8
  extends Dyno<{ rgba8: "vec4" }, { rgba8: "vec4" }>
  implements HasDynoOut<"vec4">
{
  constructor({ rgba8 }: { rgba8?: DynoVal<"vec4"> }) {
    super({
      inTypes: { rgba8: "vec4" },
      inputs: { rgba8 },
      statements: ({ inputs, outputs }) => [
        `target = ${inputs.rgba8 ?? "vec4(0.0, 0.0, 0.0, 0.0)"};`,
      ],
    });
  }

  dynoOut(): DynoValue<"vec4"> {
    return new DynoOutput(this, "rgba8");
  }
}

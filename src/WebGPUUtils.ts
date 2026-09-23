export function calcDispatchSize(
  workgroups: number,
  maxPerDimension = 65535,
): [number, number] {
  const clamped = Math.max(1, Math.ceil(workgroups));
  if (clamped <= maxPerDimension) {
    return [clamped, 1];
  }
  const x = Math.min(maxPerDimension, Math.ceil(Math.sqrt(clamped)));
  const y = Math.ceil(clamped / x);
  if (y > maxPerDimension) {
    throw new Error(
      `Spark: compute dispatch of ${clamped} workgroups exceeds device limits ` +
        `(maxComputeWorkgroupsPerDimension=${maxPerDimension}); reduce the splat count.`,
    );
  }
  return [x, y];
}

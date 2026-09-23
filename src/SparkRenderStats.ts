/** Counts for one committed accumulator, not fragment coverage or GPU timing. */
export type SparkRenderStats = Readonly<{
  generation: number;
  selectedSplats: number;
  /** Submitted instances after invalid depths are excluded; null until measured. */
  drawnSplats: number | null;
  ordering: "readback" | "radix";
}>;

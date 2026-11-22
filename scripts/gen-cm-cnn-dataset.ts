/**
 * Regenerate the deterministic 32x32 CM-CNN-Accel benchmark.
 *
 *   node scripts/gen-cm-cnn-dataset.ts
 *
 * Reads data/digits-8x8.csv.gz (1,797 samples), builds the 1,437/360
 * train/test split, an independent 360-image dev set for threshold tuning,
 * and the 360 x 6-frame saccade stream.  All seeds are fixed, so the output
 * bytes are reproducible.
 */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  buildBenchmark,
  buildDevSet,
  encodeImageSet,
  encodeStreamSet,
  parseDigitsGz,
  type BuiltBenchmark,
} from "../src/lib/vision/cm-cnn-dataset.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

export function writeDataset(root = ROOT): BuiltBenchmark {
  const gz = readFileSync(join(root, "data/digits-8x8.csv.gz"));
  const digits = parseDigitsGz(gz);
  const bench = buildBenchmark(digits);
  const dev = buildDevSet(digits);
  mkdirSync(join(root, "experiments/cm-cnn-accel/dataset"), { recursive: true });
  const dir = join(root, "experiments/cm-cnn-accel/dataset");
  writeFileSync(join(dir, "train.bin"), encodeImageSet(bench.train));
  writeFileSync(join(dir, "test.bin"), encodeImageSet(bench.test));
  writeFileSync(join(dir, "dev.bin"), encodeImageSet(dev));
  writeFileSync(join(dir, "stream.bin"), encodeStreamSet(bench.stream));
  return bench;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const bench = writeDataset();
  console.log("train", bench.train.count, "test", bench.test.count, "stream", bench.stream.seqs, "x", bench.stream.framesPerSeq);
  console.log("stats", JSON.stringify(bench.stats));
}

/**
 * CM-CNN-Accel v3.1 regression tests.
 *
 *   node --test src/__tests__/cm-cnn-accel.test.ts
 */
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  ACCEL_DIR,
  BINARY,
  CODEC_SHA256,
  DENSE_MACS,
  DATASET,
  ROOT,
  WEIGHTS,
  evaluateGates,
  runBenchmark,
  sha256File,
  weightsLocked,
  type BenchResults,
} from "../lib/vision/cm-cnn-accel.ts";

const M1 = "M1: v1-input-sparse-only";
const M2 = "M2: v3-clustered-tap-only";
const M3 = "M3: v3-clustered-cm-stageA-only";
const M4 = "M4: v3-clustered-stageB-only";
const M5 = "M5: v3-clustered-cm-full-turbo";

function recorded(): BenchResults {
  return JSON.parse(readFileSync(join(ACCEL_DIR, "results.json"), "utf8")) as BenchResults;
}

test("src/lib/codec.ts is byte-for-byte untouched", () => {
  assert.equal(sha256File(join(ROOT, "src/lib/codec.ts")), CODEC_SHA256);
});

test("pretrained weights are frozen, read-only and checksum-pinned", () => {
  const lock = weightsLocked(WEIGHTS);
  assert.equal(lock.readOnly, true, "weights must not be writable");
  const pinned = readFileSync(join(ACCEL_DIR, "weights.sha256"), "utf8").trim().split(/\s+/)[0];
  assert.equal(lock.sha256, pinned);
});

test("dataset files are the documented deterministic benchmark", () => {
  const expected = {
    train: [1437, 32, 32],
    test: [360, 32, 32],
    dev: [360, 32, 32],
  } as const;
  for (const [name, [count, h, w]] of Object.entries(expected)) {
    const b = readFileSync(DATASET[name as keyof typeof expected]);
    assert.equal(b.subarray(0, 8).toString("ascii"), "CMIMG001");
    assert.equal(b.readUInt32LE(8), count, `${name} count`);
    assert.equal(b.readUInt32LE(12), h);
    assert.equal(b.readUInt32LE(16), w);
  }
  const s = readFileSync(DATASET.stream);
  assert.equal(s.subarray(0, 8).toString("ascii"), "CMVID001");
  assert.equal(s.readUInt32LE(8), 360);
  assert.equal(s.readUInt32LE(12), 6);
});

test("dense MAC total matches the documented 75,792,384", () => {
  assert.equal(DENSE_MACS, 294912 + 18874368 + 18874368 + 37748736);
});

test("all 13 acceptance gates pass on the recorded run", () => {
  const report = evaluateGates(recorded());
  for (const check of report.checks) {
    assert.equal(check.ok, true, `${check.id}: ${check.detail}`);
  }
  assert.equal(report.pass, true);
  assert.equal(report.checks.length, 13);
});

test("waterfall isolates loop layout from Stage A and Stage B", () => {
  const r = recorded();
  const m1 = r.modes[M1], m2 = r.modes[M2], m3 = r.modes[M3], m4 = r.modes[M4], m5 = r.modes[M5];
  assert.equal(m2.macs_per_image, m1.macs_per_image, "M1 and M2 execute identical tap sets");
  assert.ok((m2.logit_rmse ?? 1) < 1e-4, "isomorphic layout changes logits only by FP rounding");
  assert.ok((m2.dense_agreement ?? 0) === 1);
  assert.ok(m3.macs_per_image < m2.macs_per_image * 0.8, "Stage A removes ~5.5 pp of MACs");
  assert.ok((m4.macs_per_image ?? Infinity) < m2.macs_per_image, "Stage B alone must reduce some MACs");
  assert.ok(m5.median_ms < m1.median_ms && m5.median_ms < m2.median_ms);
  assert.ok((m5.logit_rmse ?? 1) < 1.0, "M5 logit error stays well below random's");
});

test("unclustered random collapse and block-random fidelity separation", () => {
  const r = recorded();
  const m5 = r.modes[M5];
  const random = r.modes["block-matched-random-skip"];
  const unc = r.modes["unclustered-random-skip"];
  assert.ok(r.dense.accuracy - unc.accuracy >= 0.5, "unclustered random drops >= 50 pp");
  assert.ok(unc.mac_elimination > 0.83 && unc.mac_elimination < 0.86, "same budget regime");
  assert.ok(random.accuracy_final < m5.accuracy_final, "block random must be less accurate");
  assert.ok((random.logit_rmse ?? 0) >= 2.5 * (m5.logit_rmse ?? 1));
  assert.ok((random.conv4_rel_l2 ?? 0) >= 2.5 * (m5.conv4_rel_l2 ?? 1));
  assert.ok(random.stageb_aborts === 0);
});

test("gate verifier rejects a tampered result", () => {
  const results = recorded();
  results.modes[M5].accuracy_final = results.dense.accuracy - 0.05;
  const report = evaluateGates(results);
  assert.equal(report.pass, false);
  assert.equal(report.checks.find((c) => c.id === "v3-accuracy")?.ok, false);
});

test("gate verifier rejects a broken waterfall", () => {
  const results = recorded();
  results.modes[M2].macs_per_image += 1_000_000;
  const report = evaluateGates(results);
  assert.equal(report.checks.find((c) => c.id === "waterfall-consistency")?.ok, false);
});

test("live native benchmark reproduces the waterfall and gates", { timeout: 180_000 }, async (t) => {
  if (!existsSync(BINARY)) {
    t.skip(`native binary missing: ${BINARY}`);
    return;
  }
  const out = join(tmpdir(), "cm-cnn-accel-live.json");
  const { results } = await runBenchmark({ reps: 1, json: out, stream: null });
  const m1 = results.modes[M1], m2 = results.modes[M2], m5 = results.modes[M5];
  assert.equal(m2.macs_per_image, m1.macs_per_image);
  assert.ok((m2.logit_rmse ?? 1) < 1e-4);
  assert.ok(Math.abs(m5.accuracy_final - results.dense.accuracy) <= 0.003);
  assert.ok(m5.mac_elimination >= 0.835);
  assert.ok(m5.speedup >= 2.5);
  assert.ok(m5.median_ms < m1.median_ms && m5.median_ms < m2.median_ms);
  const unc = results.modes["unclustered-random-skip"];
  assert.ok(results.dense.accuracy - unc.accuracy >= 0.5);
});

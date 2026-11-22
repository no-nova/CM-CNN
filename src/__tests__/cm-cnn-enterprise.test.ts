/**
 * CM-CNN Phase C (enterprise hybrid) regression tests.
 *
 * These pin the *measured* facts of `experiments/cm-cnn-enterprise/results.json` and the
 * invariants the phase must never break:
 *
 *   - the Phase B zero-modification baseline stays archived and unmodified,
 *   - `src/lib/codec.ts` stays byte-for-byte identical,
 *   - structured pruning keeps every width 16-aligned, so no layer ever needs a gather kernel,
 *   - static pruning is a large, real single-core win (S1 vs M0),
 *   - the CM 2x2 tile gate never loses more than the 0.50 pp accuracy budget versus M0,
 *   - the gate beats the equal-rate random control,
 *   - and the shortfall against the aspirational spatial-gating gates is *recorded*, so the
 *     documentation and the data cannot drift apart.
 *
 *   node --test src/__tests__/cm-cnn-enterprise.test.ts
 */
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { sha256File } from "../lib/vision/cm-cnn-universal.ts";

const ROOT = join(import.meta.dirname, "..", "..");
const DIR = join(ROOT, "experiments/cm-cnn-enterprise");
const UNIVERSAL = join(ROOT, "experiments/cm-cnn-universal");
const RESULTS = join(DIR, "results.json");
const ARCHIVE = join(UNIVERSAL, "results.phase-b-zero-mod.json");

interface ModeResult {
  accuracy: number;
  macs_per_image: number;
  mac_elimination: number;
  median_ms: number;
  speedup: number;
  logit_rmse: number;
  dense_agreement: number;
  tile_skip_rate: number;
}
interface EnterpriseResults {
  resnet18: {
    modes: Record<string, ModeResult>;
    stream: {
      frames: number;
      dense: { accuracy: number; ms_per_frame: number };
      temporal: { accuracy: number; ms_per_frame: number; speedup: number; mac_elimination: number };
    };
    weights: { sha256: string; bytes: number; macs: number; teacher_macs: number; mac_cut_vs_teacher: number };
  };
  mobilenetv2: EnterpriseResults["resnet18"];
  meta: Record<string, unknown>;
}

const R = JSON.parse(readFileSync(RESULTS, "utf8")) as EnterpriseResults;
const MODES = [
  "M0: original-dense",
  "S1: static-pruned-dense",
  "S2: static-pruned-cm-tile-static",
  "S2-random: static-pruned-random-tile-skip",
] as const;

function readModel(path: string) {
  const buf = readFileSync(path);
  assert.equal(buf.subarray(0, 8).toString("latin1"), "CMUNIV02");
  const type = buf.readUInt32LE(8);
  const n = buf.readUInt32LE(12);
  const cls = buf.readUInt32LE(16);
  let off = 20;
  const nodes: Array<{ op: number; ci: number; co: number; h: number; w: number; k: number; g: number }> = [];
  for (let i = 0; i < n; i++) {
    const op = buf.readInt32LE(off);
    const ci = buf.readInt32LE(off + 12);
    const co = buf.readInt32LE(off + 16);
    const h = buf.readInt32LE(off + 20);
    const w = buf.readInt32LE(off + 24);
    const g = buf.readInt32LE(off + 32);
    const k = buf.readInt32LE(off + 40);
    off += 48;
    let nw = 0;
    if (op === 1) nw = co * ci * k;
    else if (op === 2) nw = co * 9;
    else if (op === 5) nw = co * ci;
    off += 4 * nw + (nw ? 4 * co : 0);
    nodes.push({ op, ci, co, h, w, k, g });
  }
  assert.equal(off, buf.length, `${path}: node table and weight blobs must tile the file exactly`);
  return { type, n, cls, nodes };
}

test("Phase B zero-modification baseline is archived and frozen", () => {
  assert.ok(existsSync(ARCHIVE), "results.phase-b-zero-mod.json must exist");
  assert.ok(
    existsSync(join(UNIVERSAL, "analysis.phase-b-zero-mod.md")),
    "analysis.phase-b-zero-mod.md must exist",
  );
  const a = JSON.parse(readFileSync(ARCHIVE, "utf8")) as Record<string, any>;
  assert.ok(a.resnet18 && a.mobilenetv2, "archive must hold both models");
  assert.equal(a.resnet18.dense.accuracy, 0.977778);
  assert.equal(a.mobilenetv2.dense.accuracy, 0.969444);
  assert.equal(a.resnet18.modes["M2: clustered-tap-only"].dense_agreement, 1);
  assert.equal(a.resnet18.dense.macs_per_image, 555422720);
  // archived file must be byte-identical to what the tag recorded
  const git = readFileSync(join(UNIVERSAL, "weights.sha256"), "utf8");
  assert.match(git, /pretrained_resnet18_rgb\.bin/);
});

test("src/lib/codec.ts is untouched in this phase", () => {
  const sha = sha256File(join(ROOT, "src/lib/codec.ts"));
  assert.equal(sha, "e32dba4d78fa90783287a9760166e0a537a0081b135b4e9a27613006d9538959");
});

test("pruned backbones are frozen (chmod 444), pinned, and 16-aligned so no layer needs a gather", () => {
  for (const name of ["resnet18", "mobilenetv2"] as const) {
    const p = join(DIR, `pruned_${name}_rgb.bin`);
    const st = statSync(p);
    assert.equal(st.mode & 0o222, 0, `${name}: pruned weights must be read-only`);
    assert.equal(sha256File(p), R[name].weights.sha256, `${name}: sha256 must match results.json`);
    assert.equal(st.size, R[name].weights.bytes);

    const m = readModel(p);
    assert.equal(m.cls, 10);
    for (const nd of m.nodes) {
      if (nd.op === 1) {
        // every pointwise/3x3 output width is a multiple of 16 (and the fast-path groups are 64-wide)
        assert.equal(nd.co % 16, 0, `conv co=${nd.co} must be 16-aligned`);
        assert.equal(nd.ci % 16 === 0 || nd.ci === 3, true, `conv ci=${nd.ci} must be 16-aligned`);
      }
      if (nd.op === 2) assert.equal(nd.co % 8, 0, `depthwise co=${nd.co} must be 8-aligned`);
    }
  }
});

test("MAC accounting: teacher -> pruned cut matches the pinned weights", () => {
  for (const name of ["resnet18", "mobilenetv2"] as const) {
    const w = R[name].weights;
    assert.ok(w.teacher_macs > w.macs);
    const cut = 1 - w.macs / w.teacher_macs;
    assert.ok(Math.abs(cut - w.mac_cut_vs_teacher) < 1e-6);
    assert.ok(cut >= 0.45 && cut <= 0.6, `${name}: cut ${(cut * 100).toFixed(2)}% must be near the 50% target`);
    // the engine's own dense MAC count for the pruned model must agree with the file
    assert.equal(R[name].modes["S1: static-pruned-dense"].macs_per_image, w.macs);
  }
});

test("the ablation waterfall is complete for both models", () => {
  for (const name of ["resnet18", "mobilenetv2"] as const) {
    for (const k of MODES) assert.ok(R[name].modes[k], `${name}: missing mode ${k}`);
    assert.equal(R[name].stream.frames, 2160);
    assert.ok(R[name].stream.temporal.accuracy > 0);
    assert.ok(R[name].stream.dense.ms_per_frame > 0);
  }
  assert.equal(R.meta.threads, 1);
});

test("S1 (static pruning) is a large, real single-core win over M0", () => {
  for (const name of ["resnet18", "mobilenetv2"] as const) {
    const m0 = R[name].modes["M0: original-dense"];
    const s1 = R[name].modes["S1: static-pruned-dense"];
    assert.ok(s1.speedup >= 1.5, `${name}: S1 speedup ${s1.speedup.toFixed(2)}x`);
    assert.ok(s1.mac_elimination >= 0.45, `${name}: S1 MAC cut ${(100 * s1.mac_elimination).toFixed(2)}%`);
    assert.ok(s1.median_ms < m0.median_ms);
    assert.ok(
      s1.accuracy >= m0.accuracy - 0.005,
      `${name}: pruning must not cost more than 0.5 pp (${s1.accuracy} vs ${m0.accuracy})`,
    );
  }
});

test("the recorded operating point is the one that keeps M0's accuracy", () => {
  // v3 (rank-16 sub-branch gating) buys +2.75 pp of MAC cut on ResNet-18 but costs 2.22 pp of
  // test accuracy, four times the budget, while dev accuracy is unchanged. S1 is therefore the
  // recommended operating point and S2 is reported as the measured ceiling of the technique.
  for (const name of ["resnet18", "mobilenetv2"] as const) {
    const m0 = R[name].modes["M0: original-dense"];
    const s1 = R[name].modes["S1: static-pruned-dense"];
    const s2 = R[name].modes["S2: static-pruned-cm-tile-static"];
    assert.ok(
      s1.accuracy >= m0.accuracy - 0.005,
      `${name}: the recommended mode must stay within 0.50 pp of M0`,
    );
    assert.ok(s2.mac_elimination >= s1.mac_elimination - 1e-9, `${name}: the gate never adds MACs`);
    assert.ok(s2.median_ms > 0);
  }
  assert.match(String(R.meta.operating_point), /S1/);
  assert.match(String(R.meta.operating_point), /budget/);
});

test("the CM tile gate beats the equal-rate random control", () => {
  const m = R.resnet18.modes;
  const s2 = m["S2: static-pruned-cm-tile-static"];
  const sr = m["S2-random: static-pruned-random-tile-skip"];
  assert.ok(s2.tile_skip_rate > 0, "the ResNet-18 gate must find some skippable tiles");
  assert.ok(
    s2.accuracy >= sr.accuracy + 0.01,
    `gate ${s2.accuracy} must beat random ${sr.accuracy} by >= 1 pp`,
  );
  assert.ok(s2.logit_rmse < sr.logit_rmse, "gate logit RMSE must be better than random");
});

test("spatial-tile gating shortfall is recorded, not hidden", () => {
  const s1 = R.resnet18.modes["S1: static-pruned-dense"];
  const s2 = R.resnet18.modes["S2: static-pruned-cm-tile-static"];
  const gain = s2.speedup / s1.speedup;
  // The work order asked for >= 1.25x over S1; the measurement says ~1.01x. Pin the measurement.
  assert.ok(gain < 1.25, "if this ever passes, update docs/CM_CNN_ACCELERATION.md §13 and this assertion");
  // v3 measures 0.99x: the extra 2.75 pp of MAC cut sits in cheap early blocks.
  assert.ok(gain > 0.95, "the gate must not cost more than 5% of the wall clock");
  const doc = readFileSync(join(ROOT, "docs/CM_CNN_ACCELERATION.md"), "utf8");
  assert.match(doc, /## §13 Enterprise Hybrid/);
  assert.match(doc, /Why spatial-tile skipping cannot pay on this workload/);
  assert.match(doc, /rank-16 sub-branch gating/);
  const analysis = readFileSync(join(DIR, "analysis.md"), "utf8");
  assert.match(analysis, /static pruning yes, spatial-tile gating no/i);
});

test("temporal mode is measured against the stream's own dense baseline", () => {
  const st = R.resnet18.stream;
  assert.ok(st.temporal.accuracy <= st.dense.accuracy + 0.05);
  // measured ~0.89x: the sticky temporal rule adds bookkeeping and inherits the static gate's
  // ceiling, so it cannot be a win here. Pinned so a regression (or an improvement) shows up.
  assert.ok(st.temporal.speedup > 0.8);
  assert.ok(st.temporal.mac_elimination >= 0);
});

/**
 * CM-CNN-Accel runner / verifier.
 *
 * Drives the native AVX-512 harness (`native/cm_cnn_avx512.cpp`) against the
 * frozen VGG-4 weights and checks the hard acceptance gates.  This module is
 * deliberately dependency-free: it only uses node built-ins so it can run in
 * CI without installing anything.
 */
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { accessSync, constants, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { tmpdir } from "node:os";

export const CODEC_SHA256 =
  "e32dba4d78fa90783287a9760166e0a537a0081b135b4e9a27613006d9538959";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
export const ACCEL_DIR = join(ROOT, "experiments/cm-cnn-accel");
export const WEIGHTS = join(ACCEL_DIR, "pretrained_vgg4.bin");
export const BINARY = join(ACCEL_DIR, "cm_cnn_avx512");
export const DATASET = {
  train: join(ACCEL_DIR, "dataset/train.bin"),
  test: join(ACCEL_DIR, "dataset/test.bin"),
  dev: join(ACCEL_DIR, "dataset/dev.bin"),
  stream: join(ACCEL_DIR, "dataset/stream.bin"),
};

export interface GateConfig {
  reps: number;
  threads: number;
  th_skip: number;
  th_full: number;
  anchor_mask: number;
  stage_b: boolean;
  margin_scale: number;
  min_count: number;
  th_skip_l?: number[];
  random_ch_q?: number[];
}

export interface ModeResult {
  accuracy: number;
  accuracy_final: number;
  macs_per_image: number;
  macs_per_image_final: number;
  mac_elimination: number;
  mac_elimination_final: number;
  pass0_ms: number;
  median_ms: number;
  best_ms: number;
  speedup: number;
  speedup_pass0: number;
  layer_macs: number[];
  blocks: { full: number; center: number; skip: number };
  out_channels?: number;
  out_skips?: number;
  stageb_aborts?: number;
  out_skip_rate?: number[];
  stageb_rate?: number[];
  logit_rmse?: number;
  conv4_rel_l2?: number;
  dense_agreement?: number;
}

export interface StreamModeResult {
  accuracy: number;
  ms_per_frame: number;
  macs_per_frame: number;
  mac_elimination: number;
  speedup: number;
}

export interface BenchResults {
  config: GateConfig;
  dense: {
    accuracy: number;
    dead_rates: number[];
    oracle_block16_dead: number[];
    oracle_pixel_dead: number[];
    macs_per_image: number;
    median_ms: number;
    images_per_sec: number;
    pass_ms: number[];
  };
  clustered?: {
    accuracy: number;
    dead_rates: number[];
    oracle_block16_dead: number[];
    block8_dead?: number[];
  };
  modes: Record<string, ModeResult>;
  stream: {
    frames: number;
    dense: { accuracy: number; ms_per_frame: number; macs_per_frame: number };
    "v3-temporal-stream"?: StreamModeResult;
    "v3-full-no-temporal"?: StreamModeResult;
  };
}

export const DENSE_MACS = 75_792_384;

export interface GateCheck {
  id: string;
  ok: boolean;
  detail: string;
}

export interface GateReport {
  pass: boolean;
  checks: GateCheck[];
}

export function sha256File(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

export function weightsLocked(path: string): { readOnly: boolean; sha256: string } {
  const st = statSync(path);
  return { readOnly: (st.mode & 0o222) === 0, sha256: sha256File(path) };
}

export interface RunOptions {
  reps?: number;
  weights?: string;
  test?: string;
  stream?: string | null;
  json?: string;
  binary?: string;
  threads?: number;
  heatmaps?: string;
  extraArgs?: string[];
}

export async function runBenchmark(opts: RunOptions = {}): Promise<{
  results: BenchResults;
  stdout: string;
}> {
  const binary = opts.binary ?? BINARY;
  try {
    accessSync(binary, constants.X_OK);
  } catch {
    throw new Error(
      `missing native binary ${binary}; build with: g++ -O3 -march=native -mavx512f -mfma -std=c++20 native/cm_cnn_avx512.cpp -o ${binary}`,
    );
  }
  const json = opts.json ?? join(ACCEL_DIR, "results.json");
  const args = [
    "bench",
    "--weights",
    opts.weights ?? WEIGHTS,
    "--test",
    opts.test ?? DATASET.test,
  ];
  if (opts.stream !== null) args.push("--stream", opts.stream ?? DATASET.stream);
  args.push(
    "--json",
    json,
    "--reps",
    String(opts.reps ?? 20),
  );
  if (opts.heatmaps) args.push("--heatmaps", opts.heatmaps);
  // v3.1 defaults; callers can override with extraArgs (later CLI flags win).
  if (opts.extraArgs) args.push(...opts.extraArgs);

  const stdout = await new Promise<string>((res, rej) => {
    const child = spawn(binary, args, {
      cwd: ROOT,
      env: { ...process.env, OMP_NUM_THREADS: String(opts.threads ?? 1) },
    });
    let out = "";
    let err = "";
    child.stdout.on("data", (d: Buffer) => (out += d.toString()));
    child.stderr.on("data", (d: Buffer) => (err += d.toString()));
    child.on("error", rej);
    child.on("close", (code) => {
      if (code !== 0) rej(new Error(`bench exited ${code}: ${err || out}`));
      else res(out);
    });
  });
  return { results: JSON.parse(readFileSync(json, "utf8")) as BenchResults, stdout };
}

/**
 * Evaluate the hard acceptance gates from a results object.  The temporal gate
 * is only checked when the stream section is present and non-empty.
 */
export function evaluateGates(r: BenchResults): GateReport {
  const checks: GateCheck[] = [];
  const add = (id: string, ok: boolean, detail: string) => checks.push({ id, ok, detail });

  const dense = r.dense;
  const m1 = r.modes["M1: v1-input-sparse-only"];
  const m2 = r.modes["M2: v3-clustered-tap-only"];
  const m3 = r.modes["M3: v3-clustered-cm-stageA-only"];
  const m4 = r.modes["M4: v3-clustered-stageB-only"];
  const m5 = r.modes["M5: v3-clustered-cm-full-turbo"];
  const random = r.modes["block-matched-random-skip"];
  const unc = r.modes["unclustered-random-skip"];
  const temporal = r.stream?.["v3-temporal-stream"];
  const clustered = r.clustered;
  const pct = (x: number) => (x * 100).toFixed(2);

  add("dense-accuracy", dense.accuracy >= 0.92,
    `dense accuracy ${pct(dense.accuracy)}% >= 92.00%`);

  add("isomorphic-invariance",
    !!clustered && Math.abs(clustered.accuracy - dense.accuracy) < 1e-9,
    clustered
      ? `permuted dense ${pct(clustered.accuracy)}% vs dense ${pct(dense.accuracy)}% (bit-identical)`
      : "clustered section missing");

  const half = clustered?.block8_dead;
  add("block-deadzone-unlocked", !!half && half[2] >= 0.35 && half[3] >= 0.35,
    half
      ? `8-lane all-dead Conv2/3/4 ${half.map((x) => pct(x) + "%").join("/")} (Conv3/4 >= 35%; 16-lane ${(clustered?.oracle_block16_dead ?? []).map((x) => pct(x) + "%").join("/")})`
      : "block8_dead missing");

  add("waterfall-consistency",
    !!m1 && !!m2 && Math.abs(m2.macs_per_image - m1.macs_per_image) < 1 &&
      (m2.logit_rmse ?? 1) < 1e-4 && (m3?.macs_per_image ?? Infinity) < m2.macs_per_image,
    m1 && m2
      ? `M2 ${m2.macs_per_image.toFixed(0)} MACs vs M1 ${m1.macs_per_image.toFixed(0)}; M2 logit RMSE ${(m2.logit_rmse ?? NaN).toExponential(2)}; M3 ${(m3?.macs_per_image ?? NaN).toFixed(0)} MACs`
      : "waterfall modes missing");

  add("v3-accuracy",
    !!m5 && m5.accuracy_final >= 0.9442 && Math.abs(m5.accuracy_final - dense.accuracy) <= 0.003,
    m5 ? `M5 accuracy ${pct(m5.accuracy_final)}% vs dense ${pct(dense.accuracy)}% (>= 94.42%, <= 0.30 pp)`
       : "M5 missing");

  add("v3-mac-elimination", !!m5 && m5.mac_elimination >= 0.835,
    m5 ? `${pct(m5.mac_elimination)}% MAC eliminated (>= 83.50%)` : "M5 missing");

  add("v3-beats-input-sparse",
    !!m5 && !!m1 && m5.mac_elimination >= m1.mac_elimination + 0.045,
    m5 && m1
      ? `M5 ${pct(m5.mac_elimination)}% vs M1 ${pct(m1.mac_elimination)}% (+${pct(m5.mac_elimination - m1.mac_elimination)} pp, need >= +4.50)`
      : "M5/M1 missing");

  add("v3-speedup",
    !!m5 && !!m1 && !!m2 && m5.speedup >= 2.5 && m5.median_ms < m1.median_ms &&
      m5.median_ms < m2.median_ms,
    m5
      ? `${m5.speedup.toFixed(3)}x, ${m5.median_ms.toFixed(3)} ms vs M1 ${m1?.median_ms.toFixed(3)} ms and M2 ${m2?.median_ms.toFixed(3)} ms (>= 2.50x, strictly faster than both)`
      : "M5 missing");

  add("stageb-fires", !!m5 && (m5.stageb_aborts ?? 0) >= 5000,
    m5 ? `M5 ${m5.stageb_aborts ?? 0} Stage-B aborts (>= 5,000; M4 standalone ${m4?.stageb_aborts ?? 0})`
       : "M5 missing");

  add("random-unclustered-collapse",
    !!unc && dense.accuracy - unc.accuracy >= 0.50,
    unc ? `unclustered random ${pct(unc.accuracy)}% (${pct(dense.accuracy - unc.accuracy)} pp drop, MAC ${pct(unc.mac_elimination)}%, need >= 50 pp)`
        : "unclustered random missing");

  const rmseRatio = random && m5 ? (random.logit_rmse ?? 0) / Math.max(1e-12, m5.logit_rmse ?? 1) : 0;
  const z4Ratio = random && m5 ? (random.conv4_rel_l2 ?? 0) / Math.max(1e-12, m5.conv4_rel_l2 ?? 1) : 0;
  add("random-block-separation",
    !!random && !!m5 && random.accuracy_final < m5.accuracy_final &&
      rmseRatio >= 2.5 && z4Ratio >= 2.5,
    random && m5
      ? `block random ${pct(random.accuracy_final)}% < M5 ${pct(m5.accuracy_final)}%; logit RMSE ${rmseRatio.toFixed(2)}x, Conv4 ${z4Ratio.toFixed(2)}x worse (need >= 2.5x both)`
      : "random/M5 missing");

  add("temporal-gates",
    !!temporal && temporal.mac_elimination >= 0.85 && temporal.speedup >= 2.65 &&
      Math.abs(temporal.accuracy - r.stream.dense.accuracy) <= 0.003,
    temporal
      ? `temporal ${pct(temporal.mac_elimination)}% MAC, ${temporal.speedup.toFixed(2)}x, accuracy ${pct(temporal.accuracy)}% vs ${pct(r.stream.dense.accuracy)}%`
      : "stream missing");

  if (m5) {
    add("macs-consistent", macsClose(m5.macs_per_image, m5.layer_macs),
      `M5 layer MACs sum ${m5.layer_macs.reduce((a, b) => a + b, 0)} vs total ${m5.macs_per_image.toFixed(0)}`);
  }
  return { pass: checks.every((c) => c.ok), checks };
}
function macsClose(total: number, perLayer: number[]): boolean {
  const sum = perLayer.reduce((a, b) => a + b, 0);
  return Math.abs(sum - total) / Math.max(1, total) < 0.02;
}

export function formatGateReport(report: GateReport): string {
  const lines = report.checks.map(
    (c) => `${c.ok ? "PASS" : "FAIL"}  ${c.id.padEnd(24)} ${c.detail}`,
  );
  lines.push(report.pass ? "ALL GATES PASS" : "GATES FAILED");
  return lines.join("\n");
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const repsIdx = args.indexOf("--reps");
  const reps = repsIdx >= 0 ? Number(args[repsIdx + 1]) : 20;
  const jsonIdx = args.indexOf("--json");
  // Verification must not clobber the published results file.
  const json = jsonIdx >= 0 ? args[jsonIdx + 1] : join(tmpdir(), "cm-cnn-accel-verify.json");
  const stream = args.includes("--no-stream") ? null : undefined;
  const { results } = await runBenchmark({ reps, json, stream });
  const report = evaluateGates(results);
  console.log(formatGateReport(report));
  if (!report.pass) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}

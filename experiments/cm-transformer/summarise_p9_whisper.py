"""Summarise the 100-utterance LibriSpeech run with sentence-level bootstrap CIs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
RES = EXP / "results"
RAW = RES / "p9_whisper100_raw.json"


def boot(v, n=1000, seed=1234):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    return dict(mean=round(float(v.mean()), 4), lo=round(float(np.percentile(m, 2.5)), 4),
                hi=round(float(np.percentile(m, 97.5)), 4), n=int(v.size))


def paired(a, b, n=1000, seed=1234):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, (n, d.size))].mean(1)
    return dict(mean=round(float(d.mean()), 4), lo=round(float(np.percentile(m, 2.5)), 4),
                hi=round(float(np.percentile(m, 97.5)), 4))


def main():
    raw = json.loads(RAW.read_text())
    rows = []
    per_utt = raw["per_utterance"]
    labels = [k for k in per_utt[0] if k not in ("id", "ref_words", "seconds")]
    dense = "dense" if "dense" in labels else labels[0]
    for lab in labels:
        wer = [u[lab]["wer"] for u in per_utt]
        cer = [u[lab]["cer"] for u in per_utt]
        tf = [u[lab]["teacher_forced_top1"] for u in per_utt]
        rl2 = [u[lab]["enc_rel_l2"] for u in per_utt]
        rows.append({
            "label": lab, "n_utterances": len(wer),
            "wer": boot(wer)["mean"], "wer_ci": [boot(wer)["lo"], boot(wer)["hi"]],
            "cer": boot(cer)["mean"], "cer_ci": [boot(cer)["lo"], boot(cer)["hi"]],
            "teacher_forced_top1": round(float(np.mean(tf)), 4),
            "enc_rel_l2": round(float(np.mean(rl2)), 4),
            "paired_wer_vs_dense": paired(wer, [u[dense]["wer"] for u in per_utt])
            if lab != dense else None,
            "actual_skip": None})
    # copy the aggregate skip / timing fields from the raw file's summary table
    for r, src in zip(rows, raw.get("rows", [])):
        r["actual_skip"] = src.get("actual_skip")
    tim = {t["label"]: t for t in raw.get("timing", [])}
    for r in rows:
        t = tim.get(r["label"]) or tim.get(r["label"].replace("0.2", "0.2"))
        if t:
            r["enc_ms_per_utt"] = t.get("enc_ms_per_utt")
    out = {"dataset": raw.get("dataset"), "rows": rows,
           "dead_blocks": raw.get("dead_blocks"), "kernel_avx512": raw.get("kernel_avx512"),
           "iso_check": raw.get("iso_check"), "theta": raw.get("theta"),
           "note": "LibriSpeech dev-clean, whisper-base, CPU 16 threads, 100 utterances; "
                   "intervals are sentence-level bootstrap with 1,000 resamples."}
    (RES / "p9_whisper100.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['label']:>5}  WER {r['wer']:.4f} [{r['wer_ci'][0]:.4f},{r['wer_ci'][1]:.4f}]"
              f"  CER {r['cer']:.4f}  skip {r['actual_skip']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download small, pinned evaluation subsets from the HuggingFace datasets-server.

Outputs newline-delimited JSON into ``data/benchmarks/`` plus a ``manifest.json``
with SHA-256 and provenance. Re-runnable and deterministic (fixed row ranges,
fixed ordering, no shuffling); files whose SHA-256 still matches the manifest
are skipped.

Proxies are taken from the environment (requests default handling); nothing is
unset. No parquet / pyarrow is needed: everything goes through the
datasets-server ``/rows`` JSON endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "benchmarks"
MANIFEST = OUT_DIR / "manifest.json"

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # datasets-server caps /rows length at 100
RETRIES = 10
TIMEOUT = 90
MIN_INTERVAL = 2.0  # seconds between requests: the rows API rate-limits hard (HTTP 429)
_last_request = 0.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(msg: str) -> None:
    print(msg, flush=True)


def rows_url(dataset: str, config: str, split: str, offset: int = 0, length: int = PAGE) -> str:
    return (
        f"{ROWS_API}?dataset={dataset}&config={config}&split={split}"
        f"&offset={offset}&length={length}"
    )


def fetch_page(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    """One /rows page, paced and retried (429/5xx are common)."""
    global _last_request
    url = rows_url(dataset, config, split, offset, length)
    last = ""
    for attempt in range(RETRIES):
        wait = MIN_INTERVAL - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, timeout=TIMEOUT)
            _last_request = time.time()
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:120]}"
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep = float(retry_after) if retry_after and retry_after.isdigit() else min(
                    MIN_INTERVAL * (attempt + 1) * 3, 60.0
                )
            elif 400 <= r.status_code < 500:
                # other 4xx will not get better by retrying
                raise RuntimeError(last)
            else:
                sleep = min(2 ** attempt, 30)
        except RuntimeError:
            raise
        except Exception as exc:  # network hiccup
            last = f"{type(exc).__name__}: {exc}"
            sleep = min(2 ** attempt, 30)
        log(f"    retry {attempt + 1}/{RETRIES} after {sleep:.1f}s ({last})")
        time.sleep(sleep)
    raise RuntimeError(f"giving up on {url}: {last}")


def fetch_rows(dataset: str, config: str, split: str, limit: int) -> list[dict]:
    """Fetch the first ``limit`` rows in dataset order (deterministic)."""
    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        want = min(PAGE, limit - len(rows))
        page = fetch_page(dataset, config, split, offset, want)
        batch = page.get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
        if offset >= page.get("num_rows_total", offset):
            break
    return rows[:limit]


def all_rows(dataset: str, config: str, split: str, keep) -> list[dict]:
    """Fetch every row (paged, deterministic order), keeping ``keep(row)``."""
    kept: list[dict] = []
    offset = 0
    total = None
    while total is None or offset < total:
        page = fetch_page(dataset, config, split, offset, PAGE)
        total = page.get("num_rows_total", 0)
        batch = page.get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item["row"]
            if keep(row):
                kept.append(row)
        offset += len(batch)
    return kept


# --------------------------------------------------------------------------- #
# dataset -> rows
# --------------------------------------------------------------------------- #
def build_wikitext() -> tuple[list[dict], dict]:
    dataset, config, split = "Salesforce/wikitext", "wikitext-2-raw-v1", "test"

    def non_empty(row) -> bool:
        return bool(str(row.get("text", "")).strip())

    raw = all_rows(dataset, config, split, non_empty)
    out = [{"id": i, "text": row["text"]} for i, row in enumerate(raw)]
    meta = {"dataset": dataset, "config": config, "split": split, "rows": len(out)}
    return out, meta


def build_arc_easy() -> tuple[list[dict], dict]:
    dataset, config, split = "allenai/ai2_arc", "ARC-Easy", "test"
    raw = fetch_rows(dataset, config, split, 500)
    out = []
    for row in raw:
        choices = row["choices"]
        texts = list(choices["text"])
        labels = list(choices["label"])
        key = str(row["answerKey"]).strip()
        if key not in labels:
            raise ValueError(f"ARC answerKey {key!r} not in labels {labels!r} for {row.get('id')}")
        out.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "choices": texts,
                "answer": labels.index(key),
            }
        )
    meta = {"dataset": dataset, "config": config, "split": split, "rows": len(out)}
    return out, meta


def build_boolq() -> tuple[list[dict], dict]:
    dataset, config, split = "google/boolq", "default", "validation"
    raw = fetch_rows(dataset, config, split, 500)
    out = [
        {
            "id": str(i),
            "question": row["question"],
            "passage": row["passage"],
            "answer": bool(row["answer"]),
        }
        for i, row in enumerate(raw)
    ]
    meta = {"dataset": dataset, "config": config, "split": split, "rows": len(out)}
    return out, meta


def build_piqa_or_hellaswag() -> tuple[list[dict], dict, str]:
    """PIQA first (official repo has no viewer, so known mirrors are tried),
    then HellaSwag as the documented fallback."""
    candidates = [
        ("ybisk/piqa", "default", "validation"),
        ("nthngdy/piqa", "default", "validation"),
        ("gimmaru/piqa", "default", "validation"),
    ]
    errors = []
    for dataset, config, split in candidates:
        try:
            raw = fetch_rows(dataset, config, split, 500)
            if len(raw) < 500:
                raise RuntimeError(f"only {len(raw)} rows available")
            out = []
            for i, row in enumerate(raw):
                label = int(row["label"])
                if label not in (0, 1):
                    raise ValueError(f"bad PIQA label {label!r}")
                out.append(
                    {
                        "id": str(i),
                        "goal": row["goal"],
                        "choices": [row["sol1"], row["sol2"]],
                        "answer": label,
                    }
                )
            meta = {"dataset": dataset, "config": config, "split": split, "rows": len(out)}
            return out, meta, "piqa_val.jsonl"
        except Exception as exc:
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
            log(f"  PIQA source {dataset} unusable -> {exc}")

    dataset, config, split = "Rowan/hellaswag", "default", "validation"
    raw = fetch_rows(dataset, config, split, 500)
    out = [
        {
            "id": str(row.get("ind", i)),
            "ctx": row["ctx"],
            "choices": list(row["endings"]),
            "answer": int(row["label"]),
        }
        for i, row in enumerate(raw)
    ]
    meta = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "rows": len(out),
        "piqa_errors": errors,
    }
    return out, meta, "hellaswag_val.jsonl"


def build_gsm8k() -> tuple[list[dict], dict]:
    dataset, config, split = "openai/gsm8k", "main", "test"
    raw = fetch_rows(dataset, config, split, 100)
    out = [
        {"id": str(i), "question": row["question"], "answer": row["answer"]}
        for i, row in enumerate(raw)
    ]
    meta = {"dataset": dataset, "config": config, "split": split, "rows": len(out)}
    return out, meta


def build_pope() -> tuple[list[dict], dict]:
    """POPE was renamed lmms-lab/POPE -> lmms-lab-encoder/POPE."""
    attempts = [
        ("lmms-lab-encoder/POPE", "default", "test"),
        ("lmms-lab-encoder/POPE", "default", "val"),
        ("lmms-lab/POPE", "default", "test"),
    ]
    errors = []
    raw = None
    dataset = config = split = ""
    for ds, cfg, sp in attempts:
        try:
            raw = fetch_rows(ds, cfg, sp, 300)
            if len(raw) < 300:
                raise RuntimeError(f"only {len(raw)} rows")
            dataset, config, split = ds, cfg, sp
            break
        except Exception as exc:
            errors.append(f"{ds}/{cfg}/{sp}: {type(exc).__name__}: {exc}")
            log(f"  POPE source {ds}/{cfg}/{sp} unusable -> {exc}")
    if raw is None:
        raise RuntimeError("POPE unavailable: " + "; ".join(errors))

    out = []
    for i, row in enumerate(raw):
        text = row["question"]
        answer = row["answer"]
        # POPE is a binary yes/no QA set: synthesize the choice list, or map
        # the label onto an explicit choices field when one is present.
        row_choices = row.get("choices")
        if isinstance(row_choices, list) and row_choices:
            choices = [str(c) for c in row_choices]
            if isinstance(answer, str):
                lowered = [c.strip().lower() for c in choices]
                answer = lowered.index(answer.strip().lower())
            elif not isinstance(answer, int):
                answer = int(answer)
        else:
            choices = ["yes", "no"]
            label = str(answer).strip().lower()
            answer = 0 if label == "yes" else 1
        out.append(
            {
                "id": str(row.get("id", i)),
                "question": text,
                "choices": choices,
                "answer": int(answer),
            }
        )
    meta = {"dataset": dataset, "config": config, "split": split, "rows": len(out)}
    return out, meta


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
REQUIRED_KEYS = {
    "wikitext2_test.jsonl": ("id", "text"),
    "arc_easy_test.jsonl": ("id", "question", "choices", "answer"),
    "boolq_val.jsonl": ("id", "question", "passage", "answer"),
    "piqa_val.jsonl": ("id", "goal", "choices", "answer"),
    "hellaswag_val.jsonl": ("id", "ctx", "choices", "answer"),
    "gsm8k_test.jsonl": ("id", "question", "answer"),
    "pope_test.jsonl": ("id", "question", "choices", "answer"),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_writable(path: Path) -> None:
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def write_jsonl(path: Path, records: list[dict]) -> None:
    make_writable(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(path)
    path.chmod(0o444)


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    try:
        with MANIFEST.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def verify(path: Path, keys: tuple[str, ...]) -> tuple[int, int]:
    """Check every line parses and carries the required keys. Returns (rows, chars)."""
    rows = 0
    chars = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"VERIFY FAIL {path.name}:{lineno}: {exc}")
            if not isinstance(rec, dict):
                raise SystemExit(f"VERIFY FAIL {path.name}:{lineno}: not an object")
            missing = [k for k in keys if k not in rec]
            if missing:
                raise SystemExit(f"VERIFY FAIL {path.name}:{lineno}: missing {missing}")
            rows += 1
            chars += len(line)
    return rows, chars


def first_record(path: Path, limit: int = 200) -> str:
    with path.open("r", encoding="utf-8") as fh:
        line = fh.readline().strip()
    return line[:limit] + ("..." if len(line) > limit else "")


def manifest_entry_valid(name: str, old_files: dict) -> dict | None:
    """Return the manifest entry if the on-disk file still matches its sha256."""
    prev = old_files.get(name)
    path = OUT_DIR / name
    if not prev or not path.exists():
        return None
    if prev.get("sha256") != sha256_of(path):
        log(f"  {name}: sha256 changed on disk, rebuilding")
        return None
    if prev.get("bytes") != path.stat().st_size:
        log(f"  {name}: size changed on disk, rebuilding")
        return None
    log(f"  skip {name} (sha256 matches manifest)")
    return prev


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    old_files = manifest.get("files", {})

    builders = [
        ("wikitext2_test.jsonl", build_wikitext),
        ("arc_easy_test.jsonl", build_arc_easy),
        ("boolq_val.jsonl", build_boolq),
        ("gsm8k_test.jsonl", build_gsm8k),
        ("pope_test.jsonl", build_pope),
    ]

    files_manifest: dict[str, dict] = {}
    produced: dict[str, int] = {}

    def record(name: str, records: list[dict], meta: dict) -> None:
        path = OUT_DIR / name
        if path.exists():
            make_writable(path)
        write_jsonl(path, records)
        files_manifest[name] = {
            "dataset": meta["dataset"],
            "config": meta["config"],
            "split": meta["split"],
            "url": rows_url(meta["dataset"], meta["config"], meta["split"]),
            "rows": len(records),
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            "download_utc": utc_now(),
        }
        produced[name] = len(records)

    # simple datasets: skip the network entirely when the manifest still matches
    for name, fn in builders:
        prev = manifest_entry_valid(name, old_files)
        if prev is not None:
            files_manifest[name] = prev
            produced[name] = prev["rows"]
            continue
        log(f"[build] {name}")
        records, meta = fn()
        record(name, records, meta)

    # PIQA / HellaSwag: exactly one of the two names is produced
    hs_name = None
    for candidate in ("piqa_val.jsonl", "hellaswag_val.jsonl"):
        prev = manifest_entry_valid(candidate, old_files)
        if prev is not None:
            hs_name = candidate
            files_manifest[candidate] = prev
            produced[candidate] = prev["rows"]
            break
    if hs_name is None:
        log("[build] piqa / hellaswag")
        hs_records, hs_meta, hs_name = build_piqa_or_hellaswag()
        record(hs_name, hs_records, hs_meta)

    # drop the unused PIQA/HellaSwag variant so the directory stays unambiguous
    for stale in ("piqa_val.jsonl", "hellaswag_val.jsonl"):
        if stale not in produced:
            stale_path = OUT_DIR / stale
            if stale_path.exists():
                make_writable(stale_path)
                stale_path.unlink()
                log(f"  removed stale {stale}")

    out_manifest = {
        "generated_utc": utc_now(),
        "source": "huggingface datasets-server / hub resolve",
        "files": files_manifest,
    }
    if MANIFEST.exists():
        make_writable(MANIFEST)
    tmp = MANIFEST.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(out_manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(MANIFEST)
    MANIFEST.chmod(0o644)

    # ----------------------------------------------------------------- verify
    log("")
    log("VERIFICATION")
    header = f"{'file':<24}{'rows':>7}{'bytes':>12}  sha256"
    log(header)
    log("-" * len(header))
    total_chars_wikitext = 0
    for name in sorted(produced):
        path = OUT_DIR / name
        rows, chars = verify(path, REQUIRED_KEYS[name])
        if rows != produced[name]:
            raise SystemExit(f"VERIFY FAIL {name}: expected {produced[name]} rows, got {rows}")
        if name == "wikitext2_test.jsonl":
            total_chars_wikitext = chars
        digest = sha256_of(path)
        entry = files_manifest[name]
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise SystemExit(f"VERIFY FAIL {name}: manifest mismatch")
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise SystemExit(f"VERIFY FAIL {name}: mode is not 444")
        log(f"{name:<24}{rows:>7}{path.stat().st_size:>12}  {digest}")

    log("")
    log("FIRST RECORD (truncated to 200 chars)")
    for name in (
        "arc_easy_test.jsonl",
        "boolq_val.jsonl",
        hs_name,
        "gsm8k_test.jsonl",
        "pope_test.jsonl",
    ):
        log(f"  {name}: {first_record(OUT_DIR / name)}")

    log("")
    log(f"wikitext2_test.jsonl total characters: {total_chars_wikitext:,} "
        f"(~{total_chars_wikitext / 6:,.0f} words est.)")
    log(f"variant produced: {hs_name}  (source {files_manifest[hs_name]['dataset']}/"
        f"{files_manifest[hs_name]['config']}/{files_manifest[hs_name]['split']})")
    log(f"manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

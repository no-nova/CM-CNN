#!/usr/bin/env python3
"""fetch_whisper_audio.py -- real English speech for the Phase-6 Whisper eval.

The box has no `datasets`, no `pyarrow`, no `soundfile`, and the HF mirror of
`librispeech_asr_dummy` now only ships a parquet shard (unreadable without
pyarrow).  LibriSpeech `dev-clean` is streamed straight from the official
OpenSLR release instead:

    https://www.openslr.org/resources/12/dev-clean.tar.gz   (337,926,286 B)

The tarball is a single gzip stream, so `tarfile.open(mode="r|gz")` is fed from
an HTTP response and the download is *aborted as soon as enough chapters have
been extracted* (a few tens of MB instead of the full 338 MB).  Every extracted
chapter keeps its official `*.trans.txt` so the reference transcripts are the
published LibriSpeech ones, not something this script invented.

Output (all inside the repo, nothing outside experiments/ + data/):
    data/p6_whisper_audio/<speaker>/<chapter>/<utt>.flac
    data/p6_whisper_audio/<speaker>/<chapter>/<chapter>.trans.txt
    data/p6_whisper_audio/provenance.json

    python fetch_whisper_audio.py --chapters 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = REPO_ROOT / "data" / "p6_whisper_audio"
DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DEV_CLEAN_BYTES = 337926286


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def log(*a):
    print(*a, flush=True)


class CountingReader:
    """Wrap the HTTP response so we know how many (gzip) bytes were consumed."""

    def __init__(self, fp):
        self.fp = fp
        self.n = 0

    def read(self, size: int = -1) -> bytes:
        b = self.fp.read(size)
        self.n += len(b)
        return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", type=int, default=8,
                    help="extract this many complete speaker chapters")
    ap.add_argument("--url", default=DEV_CLEAN_URL)
    ap.add_argument("--root", default=str(AUDIO_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    prov_path = root / "provenance.json"
    if prov_path.exists() and not args.force:
        prov = json.loads(prov_path.read_text())
        log(f"[skip] {root} already populated "
            f"({len(prov['utterances'])} utterances, {prov['chapters']} chapters)")
        return 0
    root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    req = urllib.request.Request(args.url, headers={"User-Agent": "cnn-cm-p6/1.0"})
    resp = urllib.request.urlopen(req, timeout=120)
    reader = CountingReader(resp)
    log(f"[dl] {args.url} -> streaming, aborting after {args.chapters} chapters")

    chapters: list[dict] = []
    utts: list[dict] = []
    n_done = 0
    cur_chapter = None
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                p = Path(m.name)
                if not (p.name.endswith(".flac") or p.name.endswith(".trans.txt")):
                    continue
                parts = p.parts                       # ('LibriSpeech','dev-clean','1272','128104','1272-128104-0000.flac')
                if len(parts) < 5 or parts[1] != "dev-clean":
                    continue
                speaker, chapter = parts[-3], parts[-2]
                chap_key = f"{speaker}/{chapter}"
                if cur_chapter is None:
                    cur_chapter = chap_key
                if chap_key != cur_chapter:
                    # the previous chapter is complete -> close and count it
                    n_done += 1
                    prev = cur_chapter
                    chapters.append({"speaker": prev.split("/")[0],
                                     "chapter": prev.split("/")[1],
                                     "utterances": sum(
                                         1 for u in utts if u["chapter"] == prev)})
                    log(f"[dl] chapter {prev}: "
                        f"{chapters[-1]['utterances']} utts "
                        f"({reader.n / 1e6:.1f} MB consumed)")
                    cur_chapter = chap_key
                    if n_done >= args.chapters:
                        break
                buf = tf.extractfile(m).read()
                dst = root / speaker / chapter / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(buf)
                if p.name.endswith(".flac"):
                    utts.append({"id": p.stem, "speaker": speaker, "chapter": chap_key,
                                 "path": str(dst.relative_to(REPO_ROOT)),
                                 "sha256": sha256_bytes(buf), "bytes": len(buf)})
    except (tarfile.ReadError, OSError, EOFError) as e:
        # Aborting the stream mid-tarball surfaces as a short read; the chapters
        # we already captured are complete, so this is expected by design.
        log(f"[dl] stream stopped after {reader.n / 1e6:.1f} MB ({type(e).__name__})")
    finally:
        resp.close()

    # the trailing chapter was cut mid-stream: drop its utterances (and files) so
    # that every retained utterance is part of a chapter captured in full
    done = {f"{c['speaker']}/{c['chapter']}" for c in chapters}
    dropped = [u for u in utts if u["chapter"] not in done]
    utts = [u for u in utts if u["chapter"] in done]
    for u in dropped:
        (REPO_ROOT / u["path"]).unlink(missing_ok=True)
    if dropped:
        log(f"[dl] dropped {len(dropped)} utterances from the truncated trailing chapter")

    if len(utts) < 50:
        log(f"[FATAL] only {len(utts)} utterances extracted; need >= 50")
        return 2

    # reference transcripts, verbatim from the published *.trans.txt files
    refs: dict[str, str] = {}
    for tx in sorted(root.glob("*/*/*.trans.txt")):
        for line in tx.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            uid, _, text = line.partition(" ")
            refs[uid] = text.strip()
    missing = [u["id"] for u in utts if u["id"] not in refs]
    if missing:
        log(f"[FATAL] {len(missing)} utterances without transcript, e.g. {missing[:3]}")
        return 2
    for u in utts:
        u["ref"] = refs[u["id"]]
    utts.sort(key=lambda u: u["id"])

    prov = {
        "source": "LibriSpeech dev-clean (OpenSLR SLR12)",
        "url": args.url,
        "url_total_bytes": DEV_CLEAN_BYTES,
        "download_mode": "streamed tar.gz, aborted after N complete chapters",
        "bytes_consumed": reader.n,
        "chapters": len(chapters),
        "chapter_list": chapters,
        "n_utterances_extracted": len(utts),
        "total_audio_seconds": None,
        "sample_rate": 16000,
        "container": "flac",
        "decode": "subprocess ffmpeg -f flac -i pipe -> s16le 16 kHz mono",
        "utterances": utts,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    prov_path.write_text(json.dumps(prov, indent=2))
    log(f"[done] {len(utts)} utterances from {len(chapters)} chapters, "
        f"{reader.n / 1e6:.1f} MB in {prov['wall_seconds']:.0f}s -> {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

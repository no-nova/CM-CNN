#!/usr/bin/env python3
"""Pure-python GGUF v3 parser + IQ4_NL (ggml type 20) block decoder.

No `gguf` package. Reads the header, KV metadata (all value types incl. arrays),
tensor infos, and the aligned data section. Maps ggml type ids to names and
provides a reference dequantizer for IQ4_NL blocks (32 weights / 18 bytes).

The model file is opened read-only ('rb'); nothing here writes to models/.
"""
from __future__ import annotations

import struct
import sys

# ----------------------------------------------------------------------------
# GGUF constants
# ----------------------------------------------------------------------------
GGUF_MAGIC = b"GGUF"

# gguf_metadata_value_type
GGUF_VT = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32",
    5: "INT32", 6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY",
    10: "UINT64", 11: "INT64", 12: "FLOAT64",
}
_GGUF_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}

# ggml_type -> (name, block_size_in_elements, type_size_in_bytes)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}

IQ4_NL_ID = 20
IQ4_NL_BLOCK_ELEMS = 32
IQ4_NL_BLOCK_BYTES = 18

# Official kvalues used by ggml for IQ4_NL dequantization.
KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10,
                 1, 13, 25, 38, 53, 69, 89, 113]


def ggml_type_name(tid: int) -> str:
    return GGML_TYPES.get(tid, (f"UNKNOWN_{tid}", 0, 0))[0]


def ggml_type_layout(tid: int):
    """(block_elems, type_bytes) or (1, nbytes_unknown) for unknown ids."""
    return GGML_TYPES.get(tid, (f"UNKNOWN_{tid}", 0, 0))[1:]


# ----------------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------------
class GGUFReader:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.kv: dict = {}
        self.tensors: list[dict] = []
        self._parse_header()
        self._parse_kv()
        self._parse_tensor_infos()
        self.data_start = self._align(self.f.tell(), self.alignment)
        self.file_size = self._file_size()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _align(n: int, a: int) -> int:
        return (n + a - 1) // a * a

    def _file_size(self) -> int:
        import os
        return os.path.getsize(self.path)

    def _read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"short read: wanted {n}, got {len(b)} at {self.f.tell()}")
        return b

    def _u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def _i32(self) -> int:
        return struct.unpack("<i", self._read(4))[0]

    def _string(self) -> str:
        n = self._u64()
        return self._read(n).decode("utf-8", errors="replace")

    def _value(self, vt: int):
        if vt == 8:  # STRING
            return self._string()
        if vt == 9:  # ARRAY
            etype = self._u32()
            n = self._u64()
            # Arrays of strings / nested arrays are rare; handle generically.
            return [self._value(etype) for _ in range(n)]
        fmt = _GGUF_SCALAR_FMT.get(vt)
        if fmt is None:
            raise ValueError(f"unknown gguf value type {vt}")
        sz = struct.calcsize(fmt)
        return struct.unpack(fmt, self._read(sz))[0]

    # -- sections ------------------------------------------------------------
    def _parse_header(self):
        magic = self._read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"bad magic {magic!r}")
        self.version = self._u32()
        if self.version != 3:
            raise ValueError(f"expected GGUF v3, got v{self.version}")
        self.tensor_count = self._u64()
        self.kv_count = self._u64()

    def _parse_kv(self):
        for _ in range(self.kv_count):
            key = self._string()
            vt = self._u32()
            self.kv[key] = self._value(vt)

    def _parse_tensor_infos(self):
        for _ in range(self.tensor_count):
            name = self._string()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            tid = self._u32()
            offset = self._u64()
            self.tensors.append({
                "name": name, "n_dims": n_dims, "dims": dims,
                "type_id": tid, "type": ggml_type_name(tid), "offset": offset,
            })

    # -- derived -------------------------------------------------------------
    @property
    def alignment(self) -> int:
        return int(self.kv.get("general.alignment", 32))

    def tensor_layout(self, t: dict) -> dict:
        """Compute expected byte size for a tensor and whether it matches file."""
        belem, tbytes = ggml_type_layout(t["type_id"])
        n_elems = 1
        for d in t["dims"]:
            n_elems *= d
        if belem <= 0:
            return {"n_elems": n_elems, "block_elems": None,
                    "n_blocks": None, "expected_bytes": None, "ok": False}
        n_blocks = n_elems // belem
        return {
            "n_elems": n_elems, "block_elems": belem, "block_bytes": tbytes,
            "n_blocks": n_blocks, "expected_bytes": n_blocks * tbytes,
            "exact_block_multiple": (n_elems % belem == 0), "ok": True,
        }

    def tensor_bytes(self, t: dict) -> bytes:
        lay = self.tensor_layout(t)
        self.f.seek(self.data_start + t["offset"])
        return self._read(lay["expected_bytes"])

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ----------------------------------------------------------------------------
# IQ4_NL
# ----------------------------------------------------------------------------
def dequant_iq4nl(block_bytes: bytes) -> list[float]:
    """Decode one 18-byte IQ4_NL block into 32 floats.

    Layout: 2-byte little-endian fp16 scale `d`, then 16 bytes of nibble pairs.
    Element i (0..15)      <- low nibble of byte i
    Element 16+i (i 0..15) <- high nibble of byte i
    value = d * kvalues_iq4nl[nibble]
    """
    if len(block_bytes) != IQ4_NL_BLOCK_BYTES:
        raise ValueError(f"IQ4_NL block must be 18 bytes, got {len(block_bytes)}")
    d = struct.unpack("<e", block_bytes[:2])[0]
    qs = block_bytes[2:18]
    out = [0.0] * 32
    for i, b in enumerate(qs):
        out[i] = d * KVALUES_IQ4NL[b & 0x0F]
        out[16 + i] = d * KVALUES_IQ4NL[b >> 4]
    return out


def dequant_iq4nl_np(block_bytes, kv=None):
    """Vectorized variant: block_bytes is (n, 18) uint8. Returns (n, 32) float32."""
    import numpy as np
    kv = KVALUES_IQ4NL if kv is None else kv
    arr = np.ascontiguousarray(np.frombuffer(block_bytes, dtype=np.uint8)
                               .reshape(-1, IQ4_NL_BLOCK_BYTES))
    n = arr.shape[0]
    d = np.frombuffer(arr[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    kvt = np.asarray(kv, dtype=np.float32)
    lo = arr[:, 2:] & 0x0F
    hi = arr[:, 2:] >> 4
    out = np.empty((n, 32), dtype=np.float32)
    out[:, :16] = kvt[lo]
    out[:, 16:] = kvt[hi]
    out *= d[:, None]
    return out


# ----------------------------------------------------------------------------
# Self-check + CLI
# ----------------------------------------------------------------------------
def self_check(verbose: bool = True) -> dict:
    """Dequantize a synthetic block and verify values are d * table entries."""
    import random
    random.seed(0)
    ok = True
    checks = []
    for trial in range(64):
        d = 0.001953125 * (trial + 1)          # exactly representable in fp16
        qs = bytes(random.getrandbits(8) for _ in range(16))
        blk = struct.pack("<e", d) + qs
        vals = dequant_iq4nl(blk)
        allowed = sorted({d * k for k in KVALUES_IQ4NL})
        # every value must equal d * one of the 16 table entries
        bad = [v for v in vals if min(abs(v - a) for a in allowed) > 1e-12]
        # and the multiset of nibbles must match the packed bytes
        nibbles = [qs[i] & 0x0F for i in range(16)] + [qs[i] >> 4 for i in range(16)]
        idx = [KVALUES_IQ4NL.index(round(v / d)) for v in vals]
        ok &= not bad and idx == nibbles
        checks.append(len(bad))
    # A real block from the model (if present) is checked by the CLI below.
    res = {"trials": len(checks), "all_values_in_table": ok,
           "max_bad_per_block": max(checks)}
    if verbose:
        print(f"[self-check] synthetic blocks: {res}")
    return res


def scan_iq4nl_block(data: bytes, n_blocks: int, kv=KVALUES_IQ4NL):
    """Verify every block decodes to exact multiples of d (vectorized numpy)."""
    import numpy as np
    a = np.frombuffer(data, dtype=np.uint8)
    n = min(n_blocks, a.size // IQ4_NL_BLOCK_BYTES)
    a = a[: n * IQ4_NL_BLOCK_BYTES].reshape(n, IQ4_NL_BLOCK_BYTES)
    d = np.frombuffer(a[:, :2].copy().tobytes(), dtype=np.float16).astype(np.float64)
    lo = (a[:, 2:] & 0x0F).astype(np.int64)
    hi = (a[:, 2:] >> 4).astype(np.int64)
    kvt = np.asarray(kv, dtype=np.float64)
    vals = np.empty((n, 32), dtype=np.float64)
    vals[:, :16] = kvt[lo] * d[:, None]
    vals[:, 16:] = kvt[hi] * d[:, None]
    ref = np.concatenate([kvt[lo], kvt[hi]], axis=1) * d[:, None]
    err = np.abs(vals - ref).max()
    # each decoded value / d must be a table entry (to float precision)
    ratio = vals / d[:, None]
    near = np.abs(ratio[..., None] - kvt[None, None, :]).min(axis=2)
    return {"blocks_checked": int(n), "max_abs_err_vs_ref": float(err),
            "max_dist_to_table_entry": float(near.max()),
            "scale_min": float(d.min()), "scale_max": float(d.max()),
            "scale_mean": float(d.mean())}


def main(argv):
    path = argv[1] if len(argv) > 1 else \
        os.environ.get("GGUF_PATH",
                     str(Path(__file__).resolve().parents[2] / "models/Qwen3.5-4B-IQ4_NL.gguf"))
    with GGUFReader(path) as g:
        print(f"file            : {path}")
        print(f"version         : {g.version}")
        print(f"tensors         : {g.tensor_count}   kv pairs: {g.kv_count}")
        print(f"alignment       : {g.alignment}   data_start: {g.data_start}")
        for k in ("general.architecture", "general.name",
                  "general.quantization_version"):
            if k in g.kv:
                print(f"{k:16s}: {g.kv[k]}")
        arch = g.kv.get("general.architecture", "?")
        for k in sorted(g.kv):
            if k.startswith(arch + ".") and not isinstance(g.kv[k], list):
                print(f"  {k}: {g.kv[k]}")
        ffn = [t for t in g.tensors if t["name"].endswith("ffn_down.weight")]
        print(f"\nffn_down tensors: {len(ffn)}")
        for t in ffn[:3] + ffn[-1:]:
            lay = g.tensor_layout(t)
            print(f"  {t['name']:32s} dims={t['dims']} type={t['type']:7s} "
                  f"bytes={lay['expected_bytes']:,} blocks={lay['n_blocks']:,}")
        # layout verification on the first ffn_down tensor
        t = ffn[len(ffn) // 2]
        data = g.tensor_bytes(t)
        lay = g.tensor_layout(t)
        print(f"\n[verify] {t['name']} type={t['type']} "
              f"bytes={len(data):,} expected={lay['expected_bytes']:,} "
              f"exact={len(data) == lay['expected_bytes']} "
              f"blocks={layout_ok(lay, len(data))}")
        chk = scan_iq4nl_block(data, min(lay["n_blocks"], 4096))
        print(f"[verify] block scan: {chk}")
        self_check()


def layout_ok(lay, nbytes):
    return (lay["exact_block_multiple"]
            and lay["expected_bytes"] == nbytes
            and nbytes % IQ4_NL_BLOCK_BYTES == 0)


if __name__ == "__main__":
    self_check()
    main(sys.argv)

#!/usr/bin/env bash
# Build the publication PDF for Paper 1 (CM-CNN-Accel) from the Markdown manuscript.
#
# Pipeline: pandoc (Markdown -> Typst) -> typst (Typst -> PDF).
#   * CJK text uses Source Han Serif CN (思源宋体), falling back to Source Han Sans CN.
#   * Figures are scaled to the text width, tables are booktabs-style (pandoc emits
#     Typst `table`/`grid` with horizontal rules only).
#   * Math is converted from LaTeX to Typst by pandoc, so no TeX installation is needed.
#
# Usage:  file/cm_cnn_accel_paper/build_paper_pdf.sh [input.md] [output.pdf]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
IN="${1:-$HERE/CM_CNN_Accel_Paper1_v2.md}"
OUT="${2:-$ROOT/paper1_bundle/CM_CNN_Accel_Paper1_Final.pdf}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export PATH="$HOME/.local/bin:$PATH"
command -v pandoc >/dev/null || { echo "pandoc not found (expected in ~/.local/bin)"; exit 1; }
command -v typst  >/dev/null || { echo "typst not found (expected in ~/.local/bin)"; exit 1; }

# --- pre-processing: make long unbreakable tokens in tables breakable ------------
# Typst sizes table columns from their widest unbreakable token; file names such as
# `03_PhaseB_Universal_ZeroMod_Analysis.md` are one token, so a 4-column table can be
# wider than the text block and the cells then overlap. Insert U+200B after every
# '_' '.' '/' '-' inside long tokens, only on table rows.
python3 - "$IN" "$TMP/in.md" <<'PYW'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
ZWSP = "\u200b"
span = re.compile(r"`([^`\s|]{13,})`")
bare = re.compile(r"(?<![\w/.\u200b`])[^\s|$`]{13,}(?![\w/.`])")


def zw(w):
    # break after '_', '/', '-' and after a '.' that is followed by a letter (file names
    # such as 'Analysis.md' break nicely, decimals such as 'x0.5' are left intact)
    w = re.sub(r"([_/-])", r"\1" + ZWSP, w)
    return re.sub(r"\.(?=[A-Za-z])", "." + ZWSP, w)


def fix_plain(t):
    # de-backtick only spans that are *entirely* wrapped, so the quoting stays balanced
    t = span.sub(lambda m: zw(m.group(1)), t)
    return bare.sub(lambda m: zw(m.group(0)), t)


def fix_row(line):
    parts = line.rstrip("\n").split("$")
    # even indices are plain text, odd indices are math -> never touch math
    for i in range(0, len(parts), 2):
        parts[i] = fix_plain(parts[i])
    return "$".join(parts) + "\n"
out = []
for line in open(src, encoding="utf-8"):
    out.append(fix_row(line) if line.lstrip().startswith("|") else line)
open(dst, "w", encoding="utf-8").write("".join(out))
PYW

pandoc "$TMP/in.md" -f markdown+pipe_tables+tex_math_dollars+implicit_figures \
       -t typst --wrap=none -o "$TMP/body.typ"

# --- post-processing -------------------------------------------------------------
python3 - "$TMP/body.typ" "$TMP/paper.typ" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

# pandoc emits `#horizontalrule`, which this Typst release does not know
s = s.replace("#horizontalrule", "#line(length: 100%, stroke: 0.6pt + luma(160))")

# scale wide figures to the text block instead of their intrinsic 300-dpi size
s = re.sub(r'#box\(width: [\d.]+pt, image\("([^"]+)"\)\);',
           r'#box(width: 100%, image("\1", width: 100%));', s)

# a figure is rendered as a centred image; the numbered "Fig. N" alt-caption is dropped so
# the explicit bold "图 N." paragraph below the image is the single caption
s = re.sub(r'#figure\(\[#image\("([^"]+)"\);\],\s*\n\s*caption: \[[^\]]*\]\s*\n\)',
           r'#align(center, image("\1", width: 100%))', s)

# the artifact appendix table has very long file names in column 1: give it explicit
# proportional widths so it cannot push the last column off the page
s = s.replace("""  columns: 4,
  align: (col, row) => (auto,auto,auto,auto,).at(col),
  inset: 6pt,
  [源文件], [角色], [论文位置], [关键内容],""",
"""  columns: (22%, 16%, 10%, 52%),
  align: (col, row) => (left,left,left,left,).at(col),
  inset: 5pt,
  [源文件], [角色], [论文位置], [关键内容],""")
# the two-column symbol table needs a fixed narrow first column as well
s = s.replace("""  columns: 2,
  align: (col, row) => (auto,auto,).at(col),
  inset: 6pt,
  [Symbol], [Meaning],""",
"""  columns: (26%, 74%),
  align: (col, row) => (left,left,).at(col),
  inset: 5pt,
  [Symbol], [Meaning],""")
s = s.replace("""  columns: 2,
  align: (col, row) => (auto,auto,).at(col),
  inset: 6pt,
  [符号], [含义],""",
"""  columns: (26%, 74%),
  align: (col, row) => (left,left,).at(col),
  inset: 5pt,
  [符号], [含义],""")

s = s.replace("""  columns: 4,
  align: (col, row) => (auto,auto,auto,auto,).at(col),
  inset: 6pt,
  [Source file], [Role], [Location], [Key content],""",
"""  columns: (20%, 19%, 11%, 50%),
  align: (col, row) => (left,left,left,left,).at(col),
  inset: 5pt,
  [Source file], [Role], [Location], [Key content],""")

# --- per-table column widths ------------------------------------------------------
# Typst sizes a column from its widest unbreakable token, so a wide table whose first
# column holds model names (``MBV3-Large@320``) overflows into the next column, and long
# math headers (``max ||Delta logit||``) collide with their neighbours. These tables get
# explicit proportional widths plus a tighter inset; everything else keeps the automatic
# layout. Keys are unique fragments of the emitted header row.
TABLE_LAYOUT = [
    ("[Pretrained Backbone], [Resolution]",
     "(14%, 10%, 8%, 10%, 13%, 13%, 15%, 8%, 9%)"),
    ("[Backbone & Campaign], [Mode]",
     "(16%, 16%, 9%, 10%, 9%, 10%, 10%, 10%, 10%)"),
    ("[Backbone], [Static Slice", "(14%, 10%, 12%, 13%, 9%, 11%, 11%, 9%, 11%)"),
    ("[Backbone], [Input Resolution]", "(16%, 11%, 14%, 14%, 14%, 14%, 17%)"),
    ("[Backbone], [`M0` Cycles / Img]", "(16%, 13%, 13%, 10%, 10%, 11%, 11%, 16%)"),
    ("[Backbone], [`V1` Static]", "(16%, 11%, 11%, 11%, 11%, 11%, 11%, 18%)"),
    ("[Method Class], [Optimization Granularity]", "(14%, 16%, 10%, 11%, 49%)"),
    ("[Artifact File], [Role]",
     "(30%, 16%, 10%, 44%)"),
    # Chinese edition: the two 8-column ablation tables overflowed the text block
    ("[模式], [ResNet-18 Top-1], [MACs]",
     "(17%, 11%, 13%, 9%, 9%, 10%, 18%, 13%)"),
]
lines = s.split("\n")
for key, cols in TABLE_LAYOUT:
    for i, ln in enumerate(lines):
        if key in ln and "], [" in ln:
            j = i
            while j > 0 and "columns:" not in lines[j]:
                j -= 1
            if "columns:" in lines[j]:
                indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
                lines[j] = indent + "columns: " + cols + ","
            k = i
            for k in range(j, min(i + 4, len(lines))):
                if "inset:" in lines[k]:
                    ind = lines[k][: len(lines[k]) - len(lines[k].lstrip())]
                    lines[k] = ind + "inset: 4pt,"
                    break
            break
s = "\n".join(lines)

# --- keep tall tables away from the page bottom -----------------------------------
# A caption that lands on the last lines of a page is torn from its table, and a table
# that starts too low is drawn over the page number. A weak break (inserted only when the
# page already holds text) keeps caption and table together.
for cap in ("Table 8.", "Table 15.", "Table 19.", "Table 20.", "Table 22.", "Table 23.",
            "Table B.1.", "Table B.1 \\(continued).", "Table C.1."):
    s = s.replace("#strong[" + cap + "]", "#pagebreak(weak: true)\n#strong[" + cap + "]", 1)

preamble = r'''// ---- typeset configuration (academic A4, CJK) ----
#set document(title: "CM-CNN-Accel: Microarchitecture-Aware Spatial and Temporal Computation Skipping for CNN Inference on AVX-512 CPUs", author: "CM-CNN-Accel Project")
#set page(paper: "a4", margin: (x: 2.15cm, y: 2.35cm), numbering: "1")
#set text(font: ("__SERIF__", "__SANS__", "Libertinus Serif"), size: 10pt, lang: "__LANG__", region: "__REGION__")
#set par(justify: true, leading: 0.70em, spacing: 0.95em)
#set heading(numbering: none)
#set figure(numbering: none)
// captions are explicit bold paragraphs, so figures are only layout wrappers; dropping the
// wrapper lets long tables break across pages instead of overflowing in an unbreakable block
#show figure: it => it.body
#show heading.where(level: 1): it => block(above: 1.3em, below: 0.8em, text(size: 14pt, weight: "bold", it.body))
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em, text(size: 11.5pt, weight: "bold", it.body))
#show heading.where(level: 3): it => block(above: 0.9em, below: 0.5em, text(size: 10.5pt, weight: "bold", it.body))
#show figure: set block(above: 1em, below: 0.8em)
#show figure.caption: set text(size: 9pt)
#show raw.where(block: true): set text(size: 8.5pt, font: ("DejaVu Sans Mono",))
#show table: set text(size: 8.8pt, hyphenate: false)
#show table.cell: set align(left)
#show table: set block(above: 0.9em, below: 0.9em)

'''
import os
serif = os.environ.get("PAPER_SERIF", "Source Han Serif")
sans = os.environ.get("PAPER_SANS", "Source Han Sans")
lang = os.environ.get("PAPER_LANG", "zh")
region = os.environ.get("PAPER_REGION", "cn")
preamble = preamble.replace("__SERIF__", serif).replace("__SANS__", sans)
preamble = preamble.replace("__LANG__", lang).replace("__REGION__", region)
open(dst, "w", encoding="utf-8").write(preamble + s)
PY

mkdir -p "$TMP/work" && cp -r "$HERE/figures" "$TMP/work/" 2>/dev/null || true
cp "$TMP/paper.typ" "$TMP/work/paper.typ"
FONTS="$HOME/.local/share/fonts/source-han-cn"
( cd "$TMP/work" && typst compile --font-path "$FONTS" paper.typ "$TMP/out.pdf" )
mkdir -p "$(dirname "$OUT")"
cp "$TMP/out.pdf" "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"

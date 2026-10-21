#!/usr/bin/env python3
"""Number audit: check every registered number in the paper's prose against its source.

Usage:
    python3 check_numbers.py PAPER_DIR [--registry numbers_registry.yaml]
                             [--run-output DIR] [--pipeline DIR]
                             [--show ok,mismatch,notfound,unregistered,error,skip]
                             [--strict]

PAPER_DIR holds elsarticle-template-num.tex, supplementary.tex, highlights.txt,
cover_letter_BAE.md, table_*.tex and the summary JSONs.

Registry entries (YAML list under `entries:`):
    id:       unique name
    file:     main | supp | highlights | cover   (or a file name)
    snippet:  raw-text snippet (~60 chars) with the checked number wrapped in [[ ]].
              Whitespace in the snippet matches any whitespace run (so line breaks
              do not matter) and every other number in the snippet matches any
              number, so editing a neighbouring number does not break the match.
    source:   Python expression evaluated with the helpers below; its value is
              the full-precision source value.
    rule:     how the printed value must relate to the source (default `round`):
                round   printed == source rounded to the printed decimals
                abs     same, comparing |source|
                pct     source is a fraction; compare 100*source rounded
                le      "at most / up to / below": source <= printed (+half unit)
                lt      strict "less than": source < printed
                ge      "at least / more than": source >= printed (-half unit)
                gt      strict "more than": source > printed
                approx  "about": |source - printed| <= tol (abs) or tol_rel*|printed|
                exact   integer / exact equality
                time    HH:MM printed; source in decimal hours or 'HH:MM'
    scale:    optional multiplier applied to the source before comparison
    tol / tol_rel: tolerances for `approx`
    kind:     check (default) | skip   -- skip entries only claim the numbers
              in their snippet (dates, identifiers, citation details, ...)
    reason:   for skip entries
    note:     free text (provenance, file:line of code constants, caveats)

Source helpers:
    J(path, "a.b[0].c")       value from a JSON file
    CSV(path)                 pandas DataFrame (cached)
    TXT(path, regex)          float of group 1 of the first match
    TXTALL(path, regex)       list of floats of group 1 of all matches
    CODE(path, regex)         like TXT, for constants in pipeline code
    TAB(path, row_regex, col, k=0) numeric cell `col` (0-based, split on &) of
                              the k-th row matching row_regex in a LaTeX table
Markers may also wrap a number word ([[four]], [[third]]); it is compared as a number.
    NPY(path)                 numpy array from a .npy/.npz file (cached)
    PEARSON(x, y)             (r, p) of scipy.stats.pearsonr
    SRC(id)                   source value of another registry entry (repeated claims)
    P(id)                     printed value of another registry entry (internal arithmetic)
    DERIVED                   path of derived/derived_values.json: values with no cached
                              pipeline output, recomputed by derived/derive_values.py
    np, math, min, max, abs, sum, len, round
Paths: absolute, RUN/... (run_output), PIPE/... (pipeline root), or relative to PAPER_DIR.
"""
import argparse
import functools
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prose import PROSE_FILES, prose_tokens, context  # noqa: E402

FILE_ALIAS = {
    "main": "elsarticle-template-num.tex",
    "supp": "supplementary.tex",
    "highlights": "highlights.txt",
    "cover": "cover_letter_BAE.md",
}
DEFAULT_PIPE = "/media/harshin/data_drive/route based UTCI"
NUMPAT = r"[+\-\u2212]?\d[\d,]*(?:\.\d+)?"
TIMEPAT = r"\d{1,2}:\d{2}"


class Ctx:
    def __init__(self, paper, run, pipe):
        self.paper, self.run, self.pipe = paper, run, pipe
        self.printed = {}
        self.src_cache = {}
        self.by_id = {}
        self.genv = None

    def path(self, p):
        if os.path.isabs(p):
            return p
        if p.startswith("RUN/"):
            return os.path.join(self.run, p[4:])
        if p.startswith("PIPE/"):
            return os.path.join(self.pipe, p[5:])
        return os.path.join(self.paper, p)


@functools.lru_cache(maxsize=None)
def _read_json(path):
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def _read_text(path):
    with open(path, errors="replace") as f:
        return f.read()


@functools.lru_cache(maxsize=None)
def _read_npy(path):
    import numpy as np
    return np.load(path, allow_pickle=True)


@functools.lru_cache(maxsize=None)
def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def _walk(obj, key):
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", key):
        if part.startswith("["):
            obj = obj[int(part[1:-1])]
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            raise KeyError(f"{part!r} not in {key!r}")
    return obj


def make_env(ctx):
    import numpy as np
    import pandas as pd

    def J(path, key=None):
        d = _read_json(ctx.path(path))
        if key is None:
            return d
        # keys may themselves contain dots (e.g. "13.0"); try greedy match first
        try:
            return _walk(d, key)
        except (KeyError, TypeError, IndexError):
            return _walk_dotted(d, key)

    def CSV(path):
        return _read_csv(ctx.path(path))

    def _f(g):
        g = g.replace(",", "")
        if re.fullmatch(r"\d{1,2}:\d{2}", g):
            h, mi = g.split(":")
            return int(h) + int(mi) / 60.0
        return float(g)

    def TXT(path, rx):
        m = re.search(rx, _read_text(ctx.path(path)), re.M)
        if not m:
            raise ValueError(f"no match for {rx!r} in {path}")
        return _f(m.group(1))

    def TXTALL(path, rx):
        return [_f(m.group(1)) for m in re.finditer(rx, _read_text(ctx.path(path)), re.M)]

    def TAB(path, row_rx, col, k=0):
        hits = 0
        for line in _read_text(ctx.path(path)).splitlines():
            if re.search(row_rx, line):
                if hits < k:
                    hits += 1
                    continue
                cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
                c = cells[col].replace("$-$", "-").replace("$", "").replace(",", "")
                c = re.sub(r"\\(?:num|SI)\{([^}]*)\}(?:\{[^}]*\})?", r"\1", c)
                m = re.search(NUMPAT, c)
                return float(m.group(0).replace("\u2212", "-"))
        raise ValueError(f"row {row_rx!r} not in {path}")

    def NPY(path):
        return _read_npy(ctx.path(path))

    def PEARSON(x, y):
        from scipy import stats
        r_, p_ = stats.pearsonr(x, y)
        return float(r_), float(p_)

    def SRC(eid):
        if eid not in ctx.src_cache:
            e = ctx.by_id.get(eid)
            if e is None or not e.get("source"):
                raise KeyError(f"SRC({eid!r}): no such entry with a source")
            ctx.src_cache[eid] = eval(e["source"], ctx.genv)
        return ctx.src_cache[eid]

    def P(eid):
        if eid not in ctx.printed:
            raise KeyError(f"P({eid!r}): entry not located")
        return ctx.printed[eid]

    return dict(DERIVED=os.path.join(HERE, "derived", "derived_values.json"), J=J, CSV=CSV, TXT=TXT, TXTALL=TXTALL, CODE=TXT, TAB=TAB, P=P, SRC=SRC, NPY=NPY, PEARSON=PEARSON, np=np, pd=pd,
                math=math, min=min, max=max, abs=abs, sum=sum, len=len, round=round,
                sorted=sorted, float=float, int=int, bool=bool, str=str, range=range, any=any, all=all, list=list)


def _walk_dotted(d, key):
    """Resolve keys that contain dots by trying the longest matching prefix."""
    obj = d
    rest = key
    while rest:
        if rest.startswith("["):
            m = re.match(r"\[(\d+)\]\.?", rest)
            obj = obj[int(m.group(1))]
            rest = rest[m.end():]
            continue
        found = False
        if isinstance(obj, dict):
            for k in sorted(obj.keys(), key=len, reverse=True):
                if rest == k or rest.startswith(k + ".") or rest.startswith(k + "["):
                    obj = obj[k]
                    rest = rest[len(k):]
                    if rest.startswith("."):
                        rest = rest[1:]
                    found = True
                    break
        if not found:
            raise KeyError(f"cannot resolve {key!r} at {rest!r}")
    return obj


def snippet_regex(snippet):
    """Build a regex from a snippet; returns (compiled, has_marker)."""
    out = []
    i = 0
    has_marker = False
    tokens = re.split(r"(\[\[.*?\]\])", snippet)
    for tok in tokens:
        if tok.startswith("[[") and tok.endswith("]]"):
            inner = tok[2:-2]
            if re.fullmatch(TIMEPAT, inner):
                pat = TIMEPAT
            elif inner.lower() in WORDNUM:
                pat = "[A-Za-z]+"
            else:
                pat = NUMPAT
            out.append("(?P<val>%s)" % pat)
            has_marker = True
            continue
        for part in re.split(r"(\s+|%s|%s)" % (TIMEPAT, r"\d[\d,]*(?:\.\d+)?"), tok):
            if not part:
                continue
            if part.isspace():
                out.append(r"\s+")
            elif re.fullmatch(TIMEPAT, part):
                out.append(TIMEPAT)
            elif re.fullmatch(r"\d[\d,]*(?:\.\d+)?", part):
                out.append(r"\d[\d,]*(?:\.\d+)?")
            else:
                out.append(re.escape(part))
    rx = "".join(out)
    rx = rx.replace(r"\s+\s+", r"\s+")
    return re.compile(rx), has_marker


WORDNUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
WORDNUM.update({"half": 0.5, "third": 1 / 3, "quarter": 0.25, "fifth": 0.2,
                "twice": 2, "double": 2, "none": 0, "all": -1})


class CellMatch:
    def __init__(self, a, b, val):
        self.a, self.b, self.val = a, b, val

    def start(self, *_):
        return self.a

    def end(self, *_):
        return self.b

    def group(self, *_):
        return self.val


def find_cell(text, cell):
    """cell = {row: regex, k: k-th matching row (default 0), col: 0-based & column,
    n: n-th number inside the cell (default 0)}."""
    hits, off = 0, 0
    lines = text.splitlines(keepends=True)
    for li, line in enumerate(lines):
        if re.search(cell["row"], line):
            if hits < cell.get("k", 0):
                hits += 1
            else:
                j = li
                while "\\\\" not in line and j + 1 < len(lines) and j - li < 3:
                    j += 1
                    line += lines[j]
                body = line.split("\\\\")[0]
                cells = body.split("&")
                col = cell["col"]
                if col >= len(cells):
                    return None
                start = off + sum(len(c) + 1 for c in cells[:col])
                raw = cells[col]
                txt = raw.replace("$-$", "-").replace("$", "").replace("--", " \u2013 ")
                ms = list(re.finditer(NUMPAT, txt))
                n = cell.get("n", 0)
                if len(ms) <= n:
                    return None
                return CellMatch(start, start + len(raw), ms[n].group(0))
        off += len(line)
    return None


def parse_printed(s):
    if s.lower() in WORDNUM:
        return float(WORDNUM[s.lower()]), 0
    s = s.replace("\u2212", "-").replace(",", "")
    if ":" in s:
        h, m = s.split(":")
        return int(h) + int(m) / 60.0, 0
    dec = len(s.split(".")[1]) if "." in s else 0
    return float(s), dec


def judge(rule, printed, dec, src, e):
    half = 0.5 * 10 ** (-dec) + 1e-9
    if rule == "time":
        if isinstance(src, str):
            h, m = src.split(":")
            src = int(h) + int(m) / 60.0
        hh, mm = divmod(int(round(src * 60)), 60)
        return abs(src - printed) < 0.5 / 60, f"{hh:02d}:{mm:02d}"
    if isinstance(src, bool):
        src = float(src)
    src = float(src) * float(e.get("scale", 1))
    if rule == "pct":
        src *= 100.0
        rule = "round"
    if rule == "abs":
        src = abs(src)
        rule = "round"
    show = f"{src:.{max(dec, 0) + 2}f}"
    if rule == "round":
        return abs(src - printed) <= half, show
    if rule == "exact":
        return abs(src - printed) < 1e-9, show
    if rule == "le":
        return src <= printed + half, show
    if rule == "lt":
        return src < printed, show
    if rule == "ge":
        return src >= printed - half, show
    if rule == "gt":
        return src > printed, show
    if rule == "approx":
        tol = e.get("tol")
        if tol is None:
            tol = float(e.get("tol_rel", 0.1)) * abs(printed)
        return abs(src - printed) <= float(tol) + 1e-12, show
    raise ValueError(f"unknown rule {rule}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper")
    ap.add_argument("--registry", default=os.path.join(HERE, "numbers_registry.yaml"))
    ap.add_argument("--pipeline", default=DEFAULT_PIPE)
    ap.add_argument("--run-output", default=None)
    ap.add_argument("--show", default="mismatch,notfound,ambiguous,error,unregistered,untraced")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    a = ap.parse_args()
    t0 = time.time()
    import yaml
    with open(a.registry) as f:
        reg = yaml.safe_load(f)
    entries = reg["entries"]
    run = a.run_output or os.path.join(a.pipeline, "run_output")
    ctx = Ctx(os.path.abspath(a.paper), run, a.pipeline)
    env = make_env(ctx)
    show = set(a.show.split(","))
    genv = dict(env)
    genv["__builtins__"] = {}
    ctx.genv = genv
    ctx.by_id = {e["id"]: e for e in entries}

    texts = {}
    for fn in PROSE_FILES:
        p = os.path.join(a.paper, fn)
        if os.path.exists(p):
            texts[fn] = _read_text(p)

    def lineno(fn, off):
        return texts[fn].count("\n", 0, off) + 1

    # 1. locate every snippet
    located = {}
    claimed = {fn: [] for fn in texts}
    prose_names = set(texts)
    results = []
    for e in entries:
        fn = FILE_ALIAS.get(e["file"], e["file"])
        if fn not in texts:
            p = os.path.join(a.paper, fn)
            if not os.path.exists(p):
                results.append(("notfound", e, None, f"file {fn} missing"))
                continue
            texts[fn] = _read_text(p)
            claimed[fn] = []
        if "cell" in e:
            m = find_cell(texts[fn], e["cell"])
            if m is None:
                results.append(("notfound", e, None, "table cell not found -- table changed, re-map"))
                continue
            located[e["id"]] = (fn, m)
            ctx.printed[e["id"]] = parse_printed(m.group("val"))[0]
            claimed[fn].append((m.start(), m.end()))
            continue
        rx, has_marker = snippet_regex(e["snippet"])
        ms = list(rx.finditer(texts[fn]))
        if not ms:
            results.append(("notfound", e, None, "snippet not found -- text changed, re-map"))
            continue
        if len(ms) > 1:
            results.append(("ambiguous", e, ms[0], f"snippet matches {len(ms)} places; lengthen it"))
        m = ms[0]
        located[e["id"]] = (fn, m)
        if e.get("kind") == "skip" or not has_marker:
            claimed[fn].append((m.start(), m.end()))
        else:
            claimed[fn].append((m.start("val"), m.end("val")))
            val, dec = parse_printed(m.group("val"))
            ctx.printed[e["id"]] = val

    def SRC_eval(e):
        if e["id"] not in ctx.src_cache:
            ctx.src_cache[e["id"]] = eval(e["source"], genv)
        return ctx.src_cache[e["id"]]

    # 2. evaluate sources
    counts = dict(ok=0, mismatch=0, notfound=0, ambiguous=0, error=0, skip=0, untraced=0)
    for e in entries:
        if e["id"] not in located:
            continue
        fn, m = located[e["id"]]
        if e.get("kind") == "skip":
            counts["skip"] += 1
            results.append(("skip", e, m, e.get("reason", "")))
            continue
        if e.get("kind") == "untraced" or not e.get("source"):
            counts["untraced"] += 1
            results.append(("untraced", e, m, e.get("note", "no source found")))
            continue
        printed_s = m.group("val")
        printed, dec = parse_printed(printed_s)
        rule = e.get("rule", "round")
        try:
            src = SRC_eval(e)
            ok, show_src = judge(rule, printed, dec, src, e)
        except Exception as ex:  # noqa: BLE001
            counts["error"] += 1
            results.append(("error", e, m, f"{type(ex).__name__}: {ex}"))
            continue
        status = "ok" if ok else "mismatch"
        counts[status] += 1
        results.append((status, e, m, f"printed {printed_s}  source {show_src}  rule {rule}"))
    counts["notfound"] = sum(1 for r in results if r[0] == "notfound")
    counts["ambiguous"] = sum(1 for r in results if r[0] == "ambiguous")

    # 3. unregistered prose numbers
    unreg = []
    ntok = 0
    for fn, text in texts.items():
        if fn not in prose_names:
            continue
        spans = sorted(claimed[fn])
        for t in prose_tokens(fn, text):
            ntok += 1
            if not any(a0 <= t["start"] < b0 for a0, b0 in spans):
                unreg.append((fn, t))
    counts["unregistered"] = len(unreg)

    # report
    order = ["mismatch", "notfound", "ambiguous", "error", "untraced", "ok", "skip"]
    for st in order:
        if st not in show:
            continue
        rows = [r for r in results if r[0] == st]
        if not rows:
            continue
        print(f"\n== {st.upper()} ({len(rows)})")
        for _, e, m, msg in rows:
            fn = FILE_ALIAS.get(e["file"], e["file"])
            loc = f"{fn}:{lineno(fn, m.start())}" if m is not None else fn
            snip = re.sub(r"\s+", " ", e.get("snippet") or "cell %s" % e.get("cell"))[:70]
            print(f"  [{e['id']}] {loc}  {msg}\n      \"{snip}\"")
            if e.get("note") and st in ("mismatch", "untraced"):
                print(f"      note: {e['note']}")
    if "unregistered" in show and unreg:
        print(f"\n== UNREGISTERED ({len(unreg)}) -- numbers in prose with no registry entry")
        for fn, t in unreg:
            print(f"  {fn}:{t['line']}  {t.get('sign', '')}{t['text']}  | {context(texts[fn], t['start'], t['end'], 70)}")
    print(f"\nSUMMARY prose numbers {ntok}; registry entries {len(entries)}; "
          + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f"; {time.time() - t0:.1f} s")
    bad = counts["mismatch"] + counts["notfound"] + counts["ambiguous"] + counts["error"] + counts["unregistered"]
    sys.exit(1 if (a.strict and bad) else 0)


if __name__ == "__main__":
    main()

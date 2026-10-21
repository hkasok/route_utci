"""Prose number extraction shared by check_numbers.py and the registry builder.

Finds numeric quantities in the *prose* of the paper's text files, i.e. after
masking LaTeX regions that are not prose (preamble, comments, display
equations, \\cite/\\ref/\\label arguments, optional arguments, sub/superscripts,
graphics and layout commands) and identifiers (Lisbon~1, Route~3, S11, JOS-3,
GMX500, ...).  Each token keeps its character offset in the raw file so that
registry snippets (matched on the raw text) can claim it.
"""
import re

PROSE_FILES = [
    "elsarticle-template-num.tex",
    "supplementary.tex",
    "highlights.txt",
    "cover_letter_BAE.md",
]

NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
TIME_RE = re.compile(r"(?<!\d)([0-2]\d):([0-5]\d)(?!\d)")


def _mask(text, spans):
    m = bytearray(len(text))
    for a, b in spans:
        for i in range(max(a, 0), min(b, len(text))):
            m[i] = 1
    return m


def _brace_end(text, i):
    """i points at '{'; return index just past the matching '}'."""
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "\\":
            continue
        if c == "{" and (j == 0 or text[j - 1] != "\\"):
            depth += 1
        elif c == "}" and (j == 0 or text[j - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return j + 1
    return len(text)


def masked_spans(name, text):
    spans = []
    if name.endswith(".tex"):
        b = text.find("\\begin{document}")
        if b > 0:
            spans.append((0, b))
        # comments
        for m in re.finditer(r"(?<!\\)%.*", text):
            spans.append(m.span())
        # display math environments
        for env in ("equation", "align", "gathered", "eqnarray"):
            for m in re.finditer(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env), text, re.S):
                spans.append(m.span())
        # commands whose (first) brace arguments are not prose
        cmd_args = {
            "cite": 1, "ref": 1, "eqref": 1, "label": 1, "input": 1, "includegraphics": 1,
            "graphicspath": 1, "resizebox": 2, "setlength": 2, "vspace": 1, "hspace": 1,
            "cmidrule": 1, "multicolumn": 2, "externaldocument": 1, "usepackage": 1,
            "setcounter": 2, "renewcommand": 2, "captionsetup": 1, "affiliation": 1,
            "ead": 1, "texttt": 1, "begin": 1, "end": 1, "journal": 1, "tabcolsep": 0,
            "sisetup": 1, "url": 1,
        }
        for m in re.finditer(r"\\([A-Za-z]+)\*?", text):
            cmd = m.group(1)
            j = m.end()
            # optional args of any command
            while j < len(text) and (text[j] == "[" or (text[j] == "(" and cmd == "cmidrule")):
                close = "]" if text[j] == "[" else ")"
                k = text.find(close, j)
                if k < 0:
                    break
                if cmd not in ("bigl", "bigr", "Bigl", "Bigr", "left", "right", "!"):
                    spans.append((j, k + 1))
                j = k + 1
            n = cmd_args.get(cmd, 0)
            if cmd == "begin":
                # tabular column spec is the 2nd brace argument
                if text.startswith("{tabular}", j) or text.startswith("{longtable}", j):
                    n = 2
            for _ in range(n):
                while j < len(text) and text[j] in " \t\n":
                    j += 1
                if j < len(text) and text[j] == "{":
                    e = _brace_end(text, j)
                    spans.append((j, e))
                    j = e
        # line-break spacing \\[4pt]
        for m in re.finditer(r"\\\\\[[^\]]*\]", text):
            spans.append(m.span())
        # sub/superscripts in math
        for m in re.finditer(r"[_^]\{[^{}]*\}|[_^]-?\d", text):
            spans.append(m.span())
        # unit macros such as \per\metre carry no digits; 1:1 line
        for m in re.finditer(r"\b1:1\b", text):
            spans.append(m.span())
    return spans


IDENT_BEFORE = re.compile(
    r"(?:Lisbon|Lisbon~|Routes?|Routes?~|Route~|L|R|S|Fig\.~S|Fig\.~|Figure~|Section~|"
    r"Table~|Sections~|Material\s+S|level-of-detail~|Silva|n_|GMX|SN-|RTX\s+A|"
    r"No\.\s+|UI/BD/|Summer)\s*$")


def is_identifier(text, a, b):
    """Heuristics for digits that name things rather than quantify them."""
    before = text[max(0, a - 25):a]
    after = text[b:b + 6]
    if a > 0 and (text[a - 1].isalpha() or text[a - 1] in "_\\"):
        return True                      # S11, R3, x2, \num-like macros
    if a > 1 and text[a - 1] == "-" and text[a - 2].isalpha():
        return True                      # JOS-3, SN-500
    if re.search(r"(?:Lisbon|Routes?|Route|Sections?|Table|Figure|Fig\.|Equation|"
                 r"Material|level-of-detail)[~\s]*$", before):
        return True
    # lists of identifiers: "Routes~1--4", "Lisbon~1, 2, 3, 5 and~6", "Routes~2 and~3"
    if re.search(r"(?:Lisbon|Routes?)[~\s]*\d+(?:(?:,|--|\s+and[~\s]*|\s*&\s*)\s*~?\d+)*"
                 r"(?:,|--|\s+and[~\s]*)\s*~?$", before):
        return True
    if re.match(r"[-\u2013]?D\b", after) or re.match(r"-D\b", after):
        return True                      # 2.5-D, 3-D
    if re.match(r"(?:st|nd|rd|th)\b", after):
        return False                     # ordinals/percentiles are quantities
    return False


def prose_tokens(name, text):
    """Return list of dicts: start, end, text, value, signed text, line."""
    mask = _mask(text, masked_spans(name, text))
    toks = []
    for m in TIME_RE.finditer(text):
        if mask[m.start()]:
            continue
        toks.append(dict(start=m.start(), end=m.end(), text=m.group(0),
                         value=int(m.group(1)) + int(m.group(2)) / 60.0, kind="time"))
    time_spans = [(t["start"], t["end"]) for t in toks]
    for m in NUM_RE.finditer(text):
        a, b = m.span()
        if mask[a]:
            continue
        if any(s <= a < e for s, e in time_spans):
            continue
        if a > 0 and (text[a - 1].isdigit() or text[a - 1] == "."):
            continue
        if is_identifier(text, a, b):
            continue
        s = m.group(0)
        sign = ""
        pre = text[max(0, a - 3):a]
        if pre.endswith("$-$"):
            sign = "-"
        elif a > 0 and text[a - 1] in "+-\u2212" and (a < 2 or not text[a - 2].isalnum()):
            sign = "-" if text[a - 1] in "-\u2212" else "+"
            # "26--31" ranges: the '-' belongs to the dash
            if a >= 2 and text[a - 2] == "-":
                sign = ""
        try:
            v = float(s.replace(",", ""))
        except ValueError:
            continue
        if sign == "-":
            v = -v
        toks.append(dict(start=a, end=b, text=s, value=v, sign=sign, kind="num"))
    toks.sort(key=lambda t: t["start"])
    for t in toks:
        t["line"] = text.count("\n", 0, t["start"]) + 1
    return toks


def context(text, a, b, width=60):
    lo = max(0, a - width // 2)
    hi = min(len(text), b + width // 2)
    return re.sub(r"\s+", " ", text[lo:hi])

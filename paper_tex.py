"""Small LaTeX helpers shared by the paper table generators."""
import re

# a text-mode minus before a number ("-0.67") prints as a hyphen; typeset a
# true minus sign. Only touches a minus that starts a number in text mode
# (after a space, '&', '(', '[' or '/'), never inside $...$ math.
_TEXT_MINUS = re.compile(r"(?<=[\s&(\[/])-(?=\d|\.\d)")


def latex_minus(tex: str) -> str:
    out, math = [], False
    for part in re.split(r"(\$)", tex):
        if part == "$":
            math = not math
            out.append(part)
        else:
            out.append(part if math else _TEXT_MINUS.sub("$-$", part))
    return "".join(out)

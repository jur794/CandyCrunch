"""Corpus preprocessing for the tokenizer benchmark.

Makes no changes to any tokenizer code. Implements the steps described in the
methods section:

1. Entries with an empty IUPAC column are dropped.
2. Glycans containing ambiguous symbols ("?") are excluded.
3. Uncertain linkage notations such as ``a1-3/5`` are expanded into all
   admissible alternatives (``a1-3``, ``a1-5``), each alternative being
   treated as an independent glycan entry.
4. After expansion, identical glycan strings are collapsed (we report on the
   distinct structural entries).
"""
from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path

# Matches a parenthesised group, e.g. the linkage part of "Gal(b1-3/4)".
_PAREN_GROUP = re.compile(r"\(([^()]*)\)")


def load_iupac(path: str | Path) -> list[tuple[str, str]]:
    """Return [(GlyTouCan ID, IUPAC Condensed), ...] with empties dropped."""
    path = Path(path)
    rows = list(csv.reader(open(path)))
    pairs = []
    for row in rows[1:]:
        if len(row) > 1 and row[1].strip():
            pairs.append((row[0].strip(), row[1].strip()))
    return pairs


def _linkage_alternatives(inner: str) -> list[str]:
    """Expand a linkage like 'a2-3/6' into ['a2-3', 'a2-6'].

    'b1-2/3/4' -> ['b1-2', 'b1-3', 'b1-4']; '1-3/5' -> ['1-3', '1-5'].
    The splitting happens around the last '-' so the anomer marker stays put.
    """
    parts = inner.split("/")
    head = parts[0]
    base = head.rsplit("-", 1)[0] + "-"
    return [head] + [base + alt_suffix for alt_suffix in parts[1:]]


def expand_alternatives(iupac: str) -> list[str]:
    """Return all admissible expansions of uncertain ('/') linkages."""
    groups = list(_PAREN_GROUP.finditer(iupac))
    expansions = []  # (span, [alternatives])
    for match in groups:
        inner = match.group(1)
        if "/" in inner:
            expansions.append((match.span(), _linkage_alternatives(inner)))
    if not expansions:
        return [iupac]

    spans = [span for span, _ in expansions]
    alternatives = [alts for _, alts in expansions]
    results = []
    for combination in itertools.product(*alternatives):
        pieces, previous_end = [], 0
        for (start, end), alt in zip(spans, combination):
            pieces.append(iupac[previous_end:start])
            pieces.append("(" + alt + ")")
            previous_end = end
        pieces.append(iupac[previous_end:])
        results.append("".join(pieces))
    return results


def prepare(path: str | Path) -> tuple[list[str], dict]:
    """Run the full preprocessing. Returns (unique_iupac_strings, stats)."""
    pairs = load_iupac(path)
    stats = {"rows": len(pairs)}

    dropped_empty = stats["rows"] - len(pairs)  # loaded already drops empties
    stats["nonempty"] = len(pairs)
    stats["empty"] = dropped_empty + (stats["rows"] - len(pairs))

    glycans_with_question_mark = [iupac for _, iupac in pairs if "?" in iupac]
    stats["excluded_has_question_mark"] = len(glycans_with_question_mark)
    glycans_without_question_mark = [(glytoucan_id, iupac) for glytoucan_id, iupac in pairs if "?" not in iupac]

    expanded = []
    uncertain_linkage_count = 0
    for glytoucan_id, iupac in glycans_without_question_mark:
        expansions = expand_alternatives(iupac)
        if len(expansions) > 1:
            uncertain_linkage_count += 1
        for expansion in expansions:
            expanded.append(expansion)
    stats["entries_with_uncertain_linkage"] = uncertain_linkage_count
    stats["after_expansion_entries"] = len(expanded)

    unique = list(dict.fromkeys(expanded))  # dedupe, keeps first occurrence
    stats["unique_glycans"] = len(unique)
    stats["duplicates_removed"] = len(expanded) - len(unique)
    return unique, stats
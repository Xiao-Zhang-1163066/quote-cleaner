"""Checks run on the AI's output before anything is stored. Plain code, no AI.

The main one is the grounding check: every number the AI reports must literally appear in
the source line it came from. That stops invented numbers, and it also stops the AI quietly
"fixing" a supplier's typo (687.30 -> 68.73), which we want reported, not repaired.
"""
import re
from decimal import ROUND_HALF_UP, Decimal

from src.readers import SourceLine

# "6,350.40", "518.4", "2400": digits with optional thousands commas and decimals.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CENT = Decimal("0.01")


def _key(value: Decimal) -> Decimal:
    # Compare at cent precision so 687.30 == 687.3, and so a float artefact in a spreadsheet
    # (1120.1400000000001) still matches the 1120.14 the AI read.
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def numbers_in(text: str) -> set[Decimal]:
    return {_key(Decimal(m.replace(",", ""))) for m in _NUMBER.findall(text)}


def find_problems(lines, source_lines: list[SourceLine]) -> list[str]:
    """Return human-readable problems with the extracted lines; empty list means all clear."""
    # Look each line up by its position, so a number is only checked against ITS OWN row.
    # Checking against the whole file would accept a value copied from a different row.
    by_no = {s.line_no: s.text for s in source_lines}
    problems = []

    for line in lines:
        where = f"line {line.line_no}"
        if not 0 <= line.match_confidence <= 1:
            problems.append(f"{where}: match_confidence {line.match_confidence} is outside 0..1")

        text = by_no.get(line.line_no)
        if text is None:
            problems.append(f"{where}: no such line in the source file")
            continue

        present = numbers_in(text)
        for field in ("qty", "unit_price", "line_total"):
            value = getattr(line, field)
            # None means "not quoted" (N/Q). It is skipped, and must not be turned into 0.
            if value is not None and _key(Decimal(str(value))) not in present:
                problems.append(f"{where}: {field}={value} does not appear in the source text")
    return problems

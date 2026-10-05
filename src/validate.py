"""Checks run on the AI's output before anything is stored. Plain code, no AI.

The main one is the grounding check: every number the AI reports must literally appear in
the source line it came from. That stops invented numbers, and it also stops the AI quietly
"fixing" a supplier's typo (687.30 -> 68.73), which we want reported, not repaired.
"""
import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal

from src.readers import SourceLine
from src.schemas import OUT_OF_RFQ

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

        # A bundle is one price for several items, so it can't also be matched to ONE of them.
        if line.bundle_item_codes and line.item_code != OUT_OF_RFQ:
            problems.append(f"{where}: bundle_item_codes given but item_code is "
                            f"{line.item_code}, not {OUT_OF_RFQ}")

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

    # A code assigned to two different lines in the SAME quote is suspicious: a supplier
    # normally quotes each RFQ item at most once. OUT_OF_RFQ is excluded because several
    # unrelated extra charges (delivery, a handling fee) legitimately share that one code.
    counts = Counter(line.item_code for line in lines if line.item_code != OUT_OF_RFQ)
    for code, n in counts.items():
        if n > 1:
            where = ", ".join(f"line {l.line_no}" for l in lines if l.item_code == code)
            problems.append(f"item_code {code} is used by {n} lines ({where}); "
                            "at most one of them can be the real match")
    return problems


def check_stated_total(quote, source_lines: list[SourceLine]) -> list[str]:
    """Ground quote.stated_total the same way find_problems grounds line numbers.

    Separate from find_problems because this is a QUOTE-level field (one value per file),
    not a LINE-level one, so it doesn't fit that function's per-line loop.
    """
    if quote.stated_total is None:
        return []   # nothing to check; a file with no printed total is not an error
    if quote.stated_total_line_no is None:
        return ["stated_total was given without stated_total_line_no to ground it against"]

    by_no = {s.line_no: s.text for s in source_lines}
    text = by_no.get(quote.stated_total_line_no)
    if text is None:
        return [f"stated_total_line_no {quote.stated_total_line_no}: no such line in the source file"]

    # A PDF's text layer often splits a label from its value: "TOTAL (incl GST)" on one line,
    # "$59,589.49" on the next. The model then cites the label's line, which is a fair reading
    # of the page, not an invented number -- and rewording the prompt didn't stop it. So the
    # value may also sit on the line directly below. Still strict: it must be printed in the
    # source, and only one line further down.
    below = by_no.get(quote.stated_total_line_no + 1, "")
    if _key(Decimal(str(quote.stated_total))) not in numbers_in(text) | numbers_in(below):
        return [f"stated_total={quote.stated_total} does not appear on line "
                f"{quote.stated_total_line_no} ({text!r})"]
    return []

"""FR-9: score the pipeline's output against the hand-built answer key.

Same pattern as rules.py and export.py: pure comparison functions that take already-shaped
data (SupplierQuote/SupplierLine -- the pipeline's real output, once Phase 8 wires it up) and
a parsed answer key. No OpenAI calls needed to test the scoring logic itself.

This file covers two of the acceptance table's numbers: item-matching accuracy, and the
per-row price/qty tolerance. Issue-detection recall/precision (the "error/warning findings"
rows of the table) is a separate piece, built next -- it needs the answer key's free-text
`flags` column mapped to rules.py's rule names, which load_expected() doesn't parse yet.
"""
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from src import config
from src.rules import Issue, Severity, SupplierQuote
from src.schemas import item_codes_from_rfq

PRICE_TOLERANCE = Decimal("0.01")
QTY_TOLERANCE = Decimal("0.01")


@dataclass(frozen=True)
class ExpectedRow:
    supplier: str
    item_code: str
    # None means "this supplier never quoted this item" -- the answer key represents
    # "missing" as a real row with blank numbers, not by leaving the row out entirely, so
    # a present-but-empty ExpectedRow IS the ground truth for the missing_item case.
    qty_canonical: Decimal | None
    unit_price_ex_gst: Decimal | None
    line_total_ex_gst: Decimal | None
    # Raw free text, e.g. "gst:unstated (assumed ex); qty short: 1100 of 1200 lm". Defaulted
    # so existing positional constructions (match/value tests) don't need updating.
    flags: str = ""


def _decimal_or_none(value: str) -> Decimal | None:
    return Decimal(value) if value else None


def load_expected(csv_path: Path = config.ANSWER_KEY_CSV) -> list[ExpectedRow]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            ExpectedRow(row["supplier"], row["item_code"],
                       _decimal_or_none(row["qty_canonical"]),
                       _decimal_or_none(row["unit_price_ex_gst"]),
                       _decimal_or_none(row["line_total_ex_gst"]),
                       row["flags"])
            for row in csv.DictReader(f)
        ]


# ---------------------------------------------------------------------------------
# Item-code matching accuracy: a 20-way "did this line land on the right RFQ item, or
# correctly land on none" question, per (supplier, item_code).
# ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchResult:
    total: int
    correct: int
    # (supplier, item_code) pairs the two sides disagree on -- kept for a human to go look
    # at, not just a bare percentage that gives no hint where to start debugging.
    mismatches: list[tuple[str, str]]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def score_item_matching(expected: list[ExpectedRow], actual: list[SupplierQuote],
                        rfq_codes: set[str] | None = None) -> MatchResult:
    """For every (supplier, RFQ item) pair, check that "was this actually quoted" agrees
    between the two sides. Answer-key rows whose item_code ISN'T one of the 20 RFQ codes
    (the lump-sum "A+B+C" bundles, the free-text "EXTRA-..." out-of-RFQ labels) are skipped
    here on purpose: our schema can never produce those exact strings (it only ever emits a
    real RFQ code or the literal OUT_OF_RFQ), so they aren't a fair target for a matching
    metric -- whether we handled them right is scored later, by issue-detection recall.
    """
    rfq_codes = rfq_codes or set(item_codes_from_rfq())
    actual_keys = {
        (quote.supplier, line.item_code)
        for quote in actual for line in quote.lines
        if line.item_code in rfq_codes
    }

    total = 0
    correct = 0
    mismatches: list[tuple[str, str]] = []
    for row in expected:
        if row.item_code not in rfq_codes:
            continue
        total += 1
        expected_quoted = row.qty_canonical is not None
        actually_matched = (row.supplier, row.item_code) in actual_keys
        if expected_quoted == actually_matched:
            correct += 1
        else:
            mismatches.append((row.supplier, row.item_code))
    return MatchResult(total, correct, mismatches)


# ---------------------------------------------------------------------------------
# Value tolerance: for rows BOTH sides agree were quoted, are the converted numbers close
# enough? ($0.01 / 0.01 -- the acceptance table's own tolerances)
# ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueMismatch:
    supplier: str
    item_code: str
    field: str
    expected: Decimal
    actual: Decimal | None


def score_values(expected: list[ExpectedRow], actual: list[SupplierQuote],
                 rfq_codes: set[str] | None = None) -> list[ValueMismatch]:
    """Only checks rows where a match exists on both sides. A row one side has and the
    other doesn't is already reported by score_item_matching() -- checking it again here
    would count the same underlying mistake twice, once as a "wrong match" and once as a
    "wrong number", which would double-punish it and make the two metrics harder to read
    independently.
    """
    rfq_codes = rfq_codes or set(item_codes_from_rfq())
    actual_lines = {
        (quote.supplier, line.item_code): line
        for quote in actual for line in quote.lines
        if line.item_code in rfq_codes
    }

    mismatches: list[ValueMismatch] = []
    for row in expected:
        if row.item_code not in rfq_codes or row.qty_canonical is None:
            continue   # not an RFQ code, or genuinely not quoted -- nothing to compare
        line = actual_lines.get((row.supplier, row.item_code))
        if line is None:
            continue   # unmatched: score_item_matching() already flags this

        if (line.qty_canonical is None
                or abs(line.qty_canonical - row.qty_canonical) > QTY_TOLERANCE):
            mismatches.append(ValueMismatch(row.supplier, row.item_code, "qty_canonical",
                                            row.qty_canonical, line.qty_canonical))
        if row.unit_price_ex_gst is not None and (
                line.unit_price_ex_gst is None
                or abs(line.unit_price_ex_gst - row.unit_price_ex_gst) > PRICE_TOLERANCE):
            mismatches.append(ValueMismatch(row.supplier, row.item_code, "unit_price_ex_gst",
                                            row.unit_price_ex_gst, line.unit_price_ex_gst))
    return mismatches


# ---------------------------------------------------------------------------------
# Issue-detection recall / precision: does rules.detect_issues() find what the answer key
# says it should, and does it ever cry wolf on an ERROR?
#
# The answer key's `flags` column is free text a human wrote, not rules.py's rule names, so
# there is no exact match between the two sides -- this is "weak supervision": we translate
# free text into rule tags via keyword search, which is itself an approximation that can be
# wrong. Any phrase this table doesn't recognise is reported, never silently dropped (see
# the conversation that led here: a dropped phrase would shrink the "expected" count and
# make recall look better than it really is -- the measuring tool lying to itself).
#
# KNOWN GAP: "total_mismatch" (the quote's own printed TOTAL vs the sum of its lines) is a
# whole-quote fact with no per-line flag text to carry it, so it can never be recovered from
# this CSV's shape. The acceptance table's "A's total" ERROR is NOT covered by this function
# and must still be checked by hand (or with a small hand-written fixture, not this parser).
# ---------------------------------------------------------------------------------

# (keyword to search for, case-insensitive) -> (rule name, severity). Checked against each
# ";"-separated chunk of the flags text, so a row like "gst:unstated (assumed ex); qty short:
# 1100 of 1200 lm" is correctly read as TWO findings, not one.
KEYWORD_RULES: list[tuple[str, str, Severity]] = [
    ("arithmetic:", "calc_error", Severity.ERROR),
    ("typo:", "typo", Severity.ERROR),
    ("price outlier", "outlier", Severity.WARNING),
    ("qty short", "qty_short", Severity.WARNING),
    ("substitution", "substitution", Severity.WARNING),
    ("missing:", "missing_item", Severity.WARNING),
    # Not "gst:unstated" -- the lump-sum row phrases this as "gst unstated" (no colon, a
    # genuinely different wording found by actually running this against the real answer
    # key). "unstated" alone is specific enough in this vocabulary to be a safe keyword.
    ("unstated", "gst_unstated", Severity.WARNING),
    ("lump sum", "lump_sum", Severity.INFO),
    ("extra charge not in rfq", "out_of_rfq", Severity.INFO),
    ("total only", "total_only", Severity.INFO),
]

# Chunks that are real, known phrases but describe a normalise.py CONVERSION, not a
# rules.py ISSUE -- recognised on purpose so they don't get reported as "unrecognised".
KNOWN_NON_ISSUES = ("unit:length", "unit:m2", "gst:incl", "rounded")

# Rules that describe the whole quote (or an out-of-RFQ charge with no real item_code to
# pin down), not one specific RFQ item -- matched on (supplier, rule) alone, ignoring
# item_code, on both the expected and the actual side.
QUOTE_LEVEL_RULES = {"gst_unstated", "total_mismatch", "lump_sum", "out_of_rfq"}

# A tag identifying one finding: (supplier, item_code -- None for a quote-level rule, rule
# name, severity). The shared shape both expected_issue_tags() and actual_issue_tags()
# produce, so the two sides can be compared with plain set operations.
IssueTag = tuple[str, str | None, str, Severity]


def expected_issue_tags(rows: list[ExpectedRow]) -> tuple[set[IssueTag], list[str]]:
    """Returns (tags, unrecognised_phrases). The second list is never silently dropped --
    see the module-level comment above for why that matters."""
    tags: set[IssueTag] = set()
    unrecognised: list[str] = []
    for row in rows:
        for chunk in (c.strip() for c in row.flags.split(";")):
            if not chunk:
                continue
            lowered = chunk.lower()
            hits = [(rule, severity) for keyword, rule, severity in KEYWORD_RULES
                   if keyword in lowered]
            for rule, severity in hits:
                item_code = None if rule in QUOTE_LEVEL_RULES else row.item_code
                tags.add((row.supplier, item_code, rule, severity))
            if not hits and not any(known in lowered for known in KNOWN_NON_ISSUES):
                unrecognised.append(f"{row.supplier}/{row.item_code}: {chunk!r}")
    return tags, unrecognised


def actual_issue_tags(issues: list[Issue]) -> set[IssueTag]:
    """Same tag shape as expected_issue_tags(), built from rules.detect_issues()'s real
    output instead of parsed free text."""
    return {
        (issue.supplier, None if issue.rule in QUOTE_LEVEL_RULES else issue.item_code,
        issue.rule, issue.severity)
        for issue in issues
    }


@dataclass(frozen=True)
class RecallResult:
    expected: set[IssueTag]
    found: set[IssueTag]

    @property
    def missed(self) -> set[IssueTag]:
        return self.expected - self.found

    @property
    def recall(self) -> float:
        return len(self.found) / len(self.expected) if self.expected else 1.0


def score_issue_recall(expected_tags: set[IssueTag], actual_tags: set[IssueTag],
                       severity: Severity) -> RecallResult:
    """Recall for one severity level: of everything the answer key says should fire at this
    severity, how much did rules.detect_issues() actually catch?
    """
    expected_at_severity = {t for t in expected_tags if t[3] == severity}
    return RecallResult(expected_at_severity, expected_at_severity & actual_tags)


def false_positive_errors(expected_tags: set[IssueTag], actual_tags: set[IssueTag]) -> set[IssueTag]:
    """ERROR-severity findings we raised that the answer key does NOT expect -- the
    acceptance table's "0 false positives" check. Only ERROR is checked: crying wolf on an
    INFO/WARNING is far less costly than a false "there's definitely a mistake here".
    """
    actual_errors = {t for t in actual_tags if t[3] == Severity.ERROR}
    expected_errors = {t for t in expected_tags if t[3] == Severity.ERROR}
    return actual_errors - expected_errors

"""Tests for src/evaluate.py. Same style as test_rules.py / test_export.py: hand-built
SupplierQuote/SupplierLine fixtures for the scoring logic, plus a couple of tests that load
the REAL answer key to pin down that the CSV parser actually understands its real shape
(blank "missing" rows, the lump-sum "+"-joined item_code, quoted commas inside notes).
"""
from decimal import Decimal as D

from src.evaluate import (
    ExpectedRow, actual_issue_tags, answer_key_supplier_label, expected_issue_tags,
    false_positives, load_expected, score_issue_recall, score_item_matching,
    score_values,
)
from src.rules import Issue, Severity, SupplierLine, SupplierQuote

RFQ_CODES = {"GIB-STD-10", "MESH-665", "FRM-9045-H12"}


def _line(item_code, qty=None, price=None, total=None) -> SupplierLine:
    return SupplierLine(item_code=item_code, raw_qty=None, raw_unit_price=None,
                        raw_line_total=None, qty_canonical=qty,
                        unit_price_ex_gst=price, line_total_ex_gst=total)


def _quote(supplier, *lines) -> SupplierQuote:
    return SupplierQuote(supplier=supplier, gst_status="excl", stated_total=None,
                         lines=list(lines))


# --- answer_key_supplier_label ----------------------------------------------

def test_answer_key_supplier_label_takes_the_first_two_filename_tokens():
    # Real filenames from data/quotes/ -- pinning down the actual convention, not a
    # made-up example, since this is the exact mismatch that made every score read ~0%.
    assert answer_key_supplier_label("A_Harbour_Timber_Q-24817.xlsx") == "A_Harbour"
    assert answer_key_supplier_label("B_Northshore_Building_Supplies_quote.pdf") == "B_Northshore"
    assert answer_key_supplier_label("C_KiwiFrame_email.txt") == "C_KiwiFrame"


# --- score_item_matching ---------------------------------------------------

def test_both_sides_agree_the_item_was_quoted_counts_as_correct():
    expected = [ExpectedRow("Harbour", "GIB-STD-10", D(180), D("27.88"), D("5018.40"))]
    actual = [_quote("Harbour", _line("GIB-STD-10", D(180), D("27.88"), D("5018.40")))]

    result = score_item_matching(expected, actual, RFQ_CODES)

    assert result.total == 1
    assert result.correct == 1
    assert result.mismatches == []


def test_both_sides_agree_the_item_was_missing_counts_as_correct():
    expected = [ExpectedRow("Harbour", "MESH-665", None, None, None)]   # blank = not quoted
    actual = [_quote("Harbour")]   # no MESH-665 line at all

    result = score_item_matching(expected, actual, RFQ_CODES)

    assert result.correct == 1


def test_a_matched_n_q_line_agrees_with_an_expected_missing_row():
    # Northshore's "BAR CHAIRS 4 pk N/Q": matched to the right code, but with no price.
    expected = [ExpectedRow("Northshore", "MESH-665", None, None, None)]
    actual = [_quote("Northshore", _line("MESH-665", qty=D(4)))]

    assert score_item_matching(expected, actual, RFQ_CODES).correct == 1


def test_a_real_item_our_system_failed_to_match_is_a_mismatch():
    # Answer key says Harbour quoted GIB-STD-10; our system produced nothing for it
    # (e.g. the AI matched it to the wrong code, or missed the line entirely).
    expected = [ExpectedRow("Harbour", "GIB-STD-10", D(180), D("27.88"), D("5018.40"))]
    actual = [_quote("Harbour")]

    result = score_item_matching(expected, actual, RFQ_CODES)

    assert result.correct == 0
    assert result.mismatches == [("Harbour", "GIB-STD-10")]


def test_a_hallucinated_match_for_an_item_that_was_never_quoted_is_a_mismatch():
    # Answer key says Harbour did NOT quote MESH-665; our system matched some line to it
    # anyway (a false match, e.g. the AI misread an unrelated line).
    expected = [ExpectedRow("Harbour", "MESH-665", None, None, None)]
    actual = [_quote("Harbour", _line("MESH-665", D(10), D("80"), D("800")))]

    result = score_item_matching(expected, actual, RFQ_CODES)

    assert result.correct == 0
    assert result.mismatches == [("Harbour", "MESH-665")]


def test_lump_sum_and_extra_charge_codes_are_excluded_from_the_denominator():
    # Neither of these strings can ever come out of our schema (it only emits a real RFQ
    # code or OUT_OF_RFQ), so they shouldn't count toward a "did we pick the right code"
    # accuracy -- whether we handled them stays a job for issue-detection recall.
    expected = [
        ExpectedRow("Kiwi", "NAIL-75-BR+NAIL-75-HDG+SEAL-123", None, None, D("980.00")),
        ExpectedRow("Northshore", "EXTRA-DELIVERY", D(1), D("160.87"), D("160.87")),
    ]
    result = score_item_matching(expected, actual=[], rfq_codes=RFQ_CODES)
    assert result.total == 0


# --- score_values ---------------------------------------------------

def test_values_within_tolerance_produce_no_mismatch():
    expected = [ExpectedRow("Harbour", "GIB-STD-10", D("180"), D("27.88"), D("5018.40"))]
    # One cent off on price, 0.01 off on qty: both exactly AT the tolerance, not over it.
    actual = [_quote("Harbour", _line("GIB-STD-10", D("180.01"), D("27.89"), D("5018.40")))]

    assert score_values(expected, actual, RFQ_CODES) == []


def test_a_price_over_tolerance_is_reported():
    expected = [ExpectedRow("Harbour", "GIB-STD-10", D("180"), D("27.88"), D("5018.40"))]
    actual = [_quote("Harbour", _line("GIB-STD-10", D("180"), D("28.50"), D("5130.00")))]

    mismatches = score_values(expected, actual, RFQ_CODES)

    fields = {m.field for m in mismatches}
    assert "unit_price_ex_gst" in fields


def test_an_unmatched_row_is_not_double_counted_here():
    # score_item_matching() already reports this as a matching failure; score_values()
    # should stay silent about it rather than also reporting a "wrong number".
    expected = [ExpectedRow("Harbour", "GIB-STD-10", D("180"), D("27.88"), D("5018.40"))]
    actual = [_quote("Harbour")]   # no line at all

    assert score_values(expected, actual, RFQ_CODES) == []


def test_a_known_error_row_is_not_double_counted_either():
    # A planted typo (price printed 10x too high) is never auto-corrected by design -- it
    # gets caught once by issue recall (the "typo" ERROR). Scoring the value here too would
    # compare our honestly-wrong extraction against the answer key's "plausible corrected"
    # number and double-punish the same root cause.
    expected = [ExpectedRow("Northshore", "GIB-AQ-10", D("24"), D("59.77"), D("1434.37"))]
    actual = [_quote("Northshore", _line("GIB-AQ-10", D("24"), D("597.65"), D("14343.70")))]

    mismatches = score_values(expected, actual, RFQ_CODES,
                              known_errors=frozenset({("Northshore", "GIB-AQ-10")}))
    assert mismatches == []


# --- load_expected: parsing the REAL answer key -----------------------------

def test_load_expected_reads_all_79_rows():
    assert len(load_expected()) == 79


def test_load_expected_parses_a_normal_priced_row():
    rows = load_expected()
    row = next(r for r in rows if r.supplier == "A_Harbour" and r.item_code == "FRM-9045-H12")
    assert row.qty_canonical == D("1200.0")
    assert row.unit_price_ex_gst == D("6.12")
    assert row.line_total_ex_gst == D("7342.0")


def test_load_expected_parses_a_missing_row_as_all_none():
    rows = load_expected()
    row = next(r for r in rows if r.supplier == "B_Northshore" and r.item_code == "CHAIR-5065")
    assert row.qty_canonical is None
    assert row.unit_price_ex_gst is None
    assert row.line_total_ex_gst is None


def test_load_expected_parses_the_lump_sum_row_without_crashing():
    rows = load_expected()
    row = next(r for r in rows if r.item_code == "NAIL-75-BR+NAIL-75-HDG+SEAL-123")
    assert row.supplier == "C_KiwiFrame"
    assert row.qty_canonical is None          # a bundle has no single quantity
    assert row.line_total_ex_gst == D("980.0")   # but it does have a total


# --- expected_issue_tags: translating free text into rule tags ---------------------

def test_a_row_with_two_findings_in_one_flags_string_produces_two_tags():
    # Real example from the answer key: one semicolon-separated string, two distinct issues.
    row = ExpectedRow("Kiwi", "FRM-9045-H12", D(1100), D("5.75"), D("6325.00"),
                      flags="gst:unstated (assumed ex); qty short: 1100 of 1200 lm")
    tags, unrecognised = expected_issue_tags([row])

    assert ("Kiwi", None, "gst_unstated", Severity.WARNING) in tags   # quote-level: no item_code
    assert ("Kiwi", "FRM-9045-H12", "qty_short", Severity.WARNING) in tags
    assert unrecognised == []


def test_a_known_conversion_note_is_not_treated_as_an_issue():
    row = ExpectedRow("Harbour", "FRM-9045-H12", D(1200), D("6.12"), D("7342.00"),
                      flags="unit:length→lm")
    tags, unrecognised = expected_issue_tags([row])
    assert tags == set()
    assert unrecognised == []   # a recognised non-issue phrase, not an unrecognised one


def test_a_genuinely_unknown_phrase_is_reported_not_dropped():
    row = ExpectedRow("Harbour", "FRM-9045-H12", D(1200), D("6.12"), D("7342.00"),
                      flags="some brand new wording nobody has seen before")
    tags, unrecognised = expected_issue_tags([row])
    assert tags == set()
    assert len(unrecognised) == 1


def test_expected_issue_tags_recognises_every_phrase_in_the_real_answer_key():
    # The integration check: run the actual keyword table against the actual file. If
    # someone edits the CSV and introduces new wording, this is the test that catches it.
    _, unrecognised = expected_issue_tags(load_expected())
    assert unrecognised == []


# --- actual_issue_tags: same tag shape, built from rules.py's real Issue objects ----

def test_actual_issue_tags_nulls_the_item_code_for_quote_level_rules():
    issues = [Issue("gst_unstated", Severity.WARNING, "no GST statement", "Kiwi"),
             Issue("calc_error", Severity.ERROR, "...", "Harbour", "MESH-665")]
    tags = actual_issue_tags(issues)
    assert ("Kiwi", None, "gst_unstated", Severity.WARNING) in tags
    assert ("Harbour", "MESH-665", "calc_error", Severity.ERROR) in tags


# --- score_issue_recall / false_positives ---------------------

def test_recall_is_one_when_every_expected_tag_was_actually_found():
    expected = {("Harbour", "MESH-665", "calc_error", Severity.ERROR)}
    actual = {("Harbour", "MESH-665", "calc_error", Severity.ERROR)}
    result = score_issue_recall(expected, actual, Severity.ERROR)
    assert result.recall == 1.0
    assert result.missed == set()


def test_recall_reports_exactly_what_was_missed():
    expected = {("Harbour", "MESH-665", "calc_error", Severity.ERROR),
               ("Northshore", "GIB-AQ-10", "typo", Severity.ERROR)}
    actual = {("Harbour", "MESH-665", "calc_error", Severity.ERROR)}   # the typo was missed
    result = score_issue_recall(expected, actual, Severity.ERROR)
    assert result.recall == 0.5
    assert result.missed == {("Northshore", "GIB-AQ-10", "typo", Severity.ERROR)}


def test_false_positives_finds_an_error_we_raised_that_was_not_expected():
    expected = {("Harbour", "MESH-665", "calc_error", Severity.ERROR)}
    actual = {("Harbour", "MESH-665", "calc_error", Severity.ERROR),
             ("Northshore", "WB-180", "calc_error", Severity.ERROR)}   # a false alarm
    assert false_positives(expected, actual, Severity.ERROR) == {
        ("Northshore", "WB-180", "calc_error", Severity.ERROR)}


def test_false_positives_only_counts_the_severity_asked_for():
    expected: set = set()
    actual = {("Harbour", "MESH-665", "qty_short", Severity.WARNING)}
    assert false_positives(expected, actual, Severity.ERROR) == set()
    assert false_positives(expected, actual, Severity.WARNING) == actual

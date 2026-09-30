"""Tests for src/rules.py. Numbers are drawn from the real mock quotes where possible."""
from decimal import Decimal as D

import pytest

from src import rules
from src.rules import Issue, Severity, SupplierLine, SupplierQuote, detect_issues
from src.schemas import OUT_OF_RFQ


# --- calc_error: RAW numbers only --------------------------------------------

def test_harbour_mesh_665_calc_error_is_caught():
    # Raw: 14 x 80.01 = 1120.14, but the file prints 11201.40 (an extra zero).
    assert rules.calc_error(D(14), D("80.01"), D("11201.40")) is True


def test_correct_arithmetic_does_not_fire():
    assert rules.calc_error(D(200), D("36.71"), D("7342.00")) is False


def test_calc_error_skips_lines_with_nothing_to_cross_check():
    # A total-only or price-only line (D's mesh, or an N/Q line) has no third number to
    # check the other two against, so there is nothing to flag.
    assert rules.calc_error(D(14), None, D("1270")) is False
    assert rules.calc_error(None, D("80.01"), D("1120.14")) is False


def test_calc_error_respects_the_dollar_tolerance():
    # $0.05 over: still fine (rounding noise, not a supplier mistake).
    assert rules.calc_error(D(3), D("10.00"), D("30.05")) is False
    assert rules.calc_error(D(3), D("10.00"), D("30.06")) is True


# --- total_mismatch: RAW line totals vs the file's own printed total --------

def test_total_mismatch_from_the_mesh_error_propagating_up():
    # Sum of Harbour's raw line totals (with the wrong MESH 11201.40) vs its own TOTAL row.
    # Numbers taken from the real A_Harbour file: Subtotal 46862.88 + GST 7029.43 = 53892.31.
    raw_totals = [D("7342.00"), D("11201.40"), D("28319.48")]   # stand-in for "everything else"
    assert rules.total_mismatch(raw_totals, D("53892.31")) is True


def test_total_mismatch_false_when_no_total_is_printed():
    assert rules.total_mismatch([D(100), D(200)], None) is False


def test_total_mismatch_within_tolerance_is_fine():
    assert rules.total_mismatch([D("100.00"), D("200.00")], D("300.04")) is False


# --- price_band: typo (>=3x or <=1/3) vs outlier (>=1.25x, under 3x) --------

def test_northshore_gib_aq_price_typo_is_caught():
    # Real GIB-AQ-10 ex-GST prices from the answer key: A=53.62, D=58.26 (median ~55.94).
    assert rules.price_band(D("59.77"), [D("53.62"), D("58.26")]) is None   # B's real (correct) price
    assert rules.price_band(D("597.7"), [D("53.62"), D("58.26")]) == "typo"   # B's actual x10 typo


def test_kiwiframe_lvl_price_outlier_is_caught_not_typo():
    # Real case: LVL ~55.43/m vs others around 36-40, roughly +40-45%: outlier, not typo.
    assert rules.price_band(D("55.43"), [D("38.26"), D("40.29")]) == "outlier"


def test_price_within_25_percent_is_not_flagged():
    assert rules.price_band(D("45.00"), [D("40.00")]) is None   # +12.5%: fine


def test_price_band_boundary_does_not_double_fire():
    median = D("100")
    assert rules.price_band(median * 3, [median]) == "typo"        # exactly 3x: typo
    assert rules.price_band(median * D("1.25"), [median]) == "outlier"   # exactly +25%: outlier


def test_price_band_with_no_other_suppliers_returns_none():
    # Caller is responsible for falling back to reference_prices.csv in this case.
    assert rules.price_band(D("100"), []) is None


# --- qty_short ----------------------------------------------------------------

def test_kiwiframe_90x45_qty_short_is_caught():
    assert rules.qty_short(D(1100), D(1200)) is True


def test_qty_meeting_or_exceeding_rfq_is_not_short():
    assert rules.qty_short(D(1200), D(1200)) is False
    assert rules.qty_short(D("182.4"), D(180)) is False   # Harbour's normal stick-rounding overage


# --- is_substitution / is_lump_sum --------------------------------------------

def test_harbour_pink_batts_substitution_is_caught():
    assert rules.is_substitution("ALT - Earthwool R2.6 n/a, Pink Batts equiv offered") is True


def test_plain_note_is_not_a_substitution():
    assert rules.is_substitution("3 week lead time on this one") is False
    assert rules.is_substitution(None) is False


def test_kiwiframe_nails_and_sika_package_is_a_lump_sum():
    assert rules.is_lump_sum(OUT_OF_RFQ, None, D("980.00")) is True


def test_northshore_delivery_has_a_real_qty_so_is_not_a_lump_sum():
    assert rules.is_lump_sum(OUT_OF_RFQ, D(1), D("160.87")) is False


def test_normal_rfq_item_is_never_a_lump_sum():
    assert rules.is_lump_sum("MESH-665", None, D("1120.14")) is False


# --- detect_issues: the orchestrator, on a small realistic dataset -----------

def line(**overrides):
    base = dict(item_code=None, raw_qty=None, raw_unit_price=None, raw_line_total=None,
                qty_canonical=None, unit_price_ex_gst=None, line_total_ex_gst=None,
                note=None, is_total_only=False)
    return SupplierLine(**{**base, **overrides})


@pytest.fixture
def rfq_qtys():
    return {"MESH-665": D(14), "LVL-24045": D(36), "INS-R26": D(28)}


@pytest.fixture
def reference_prices():
    return {"MESH-665": D("80.00"), "LVL-24045": D("43.10"), "INS-R26": D("99.56")}


def test_detect_issues_end_to_end(rfq_qtys, reference_prices):
    a = SupplierQuote("A_Harbour", "excl", D("53892.31"), [
        line(item_code="MESH-665", raw_qty=D(14), raw_unit_price=D("80.01"),
             raw_line_total=D("11201.40"), qty_canonical=D(14),
             unit_price_ex_gst=D("80.01"), line_total_ex_gst=D("11201.40")),
        line(item_code="LVL-24045", raw_qty=D(36), raw_unit_price=D("36.36"),
             raw_line_total=D("1308.96"), qty_canonical=D(36),
             unit_price_ex_gst=D("36.36"), line_total_ex_gst=D("1308.96")),
        line(item_code="INS-R26", raw_qty=D(28), raw_unit_price=D("99.56"),
             raw_line_total=D("2787.68"), qty_canonical=D(28),
             unit_price_ex_gst=D("99.56"), line_total_ex_gst=D("2787.68"),
             note="ALT - Earthwool R2.6 n/a, Pink Batts equiv offered"),
    ])
    c = SupplierQuote("C_KiwiFrame", "unstated", None, [
        line(item_code="LVL-24045", raw_qty=D(36), raw_unit_price=D("55.43"), raw_line_total=None,
             qty_canonical=D(36), unit_price_ex_gst=D("55.43"), line_total_ex_gst=D("1995.48")),
        # A lump-sum package (Kiwi Frame's nails + Sikaflex, $980, no qty): real-world data
        # shows this should be tagged "lump_sum" ONLY, not also a generic "out_of_rfq".
        line(item_code=OUT_OF_RFQ, raw_line_total=D("980.00"), line_total_ex_gst=D("980.00")),
        # A genuine single extra charge (delivery-shaped: has a real qty of 1) DOES get
        # "out_of_rfq", to prove that branch still fires when it's not a lump sum.
        line(item_code=OUT_OF_RFQ, qty_canonical=D(1), raw_line_total=D("185.00"),
             line_total_ex_gst=D("160.87")),
        # MESH-665 and INS-R26 never appear in C's lines at all -> both "missing".
    ])

    issues = detect_issues([a, c], rfq_qtys, reference_prices)
    by_rule_supplier = {(i.rule, i.supplier, i.item_code) for i in issues}

    assert ("calc_error", "A_Harbour", "MESH-665") in by_rule_supplier
    assert ("substitution", "A_Harbour", "INS-R26") in by_rule_supplier
    assert ("gst_unstated", "C_KiwiFrame", None) in by_rule_supplier
    assert ("lump_sum", "C_KiwiFrame", OUT_OF_RFQ) in by_rule_supplier
    assert ("out_of_rfq", "C_KiwiFrame", OUT_OF_RFQ) in by_rule_supplier
    assert ("missing_item", "C_KiwiFrame", "MESH-665") in by_rule_supplier
    assert ("missing_item", "C_KiwiFrame", "INS-R26") in by_rule_supplier
    # LVL: A=36.36 vs C=55.43. Median of the OTHER supplier's price for each is just that
    # one price, so C's 55.43 is checked against A's 36.36 (+52%: outlier) and vice versa
    # A's 36.36 against C's 55.43 (-34%: not flagged, price_band only flags ABOVE median).
    assert ("outlier", "C_KiwiFrame", "LVL-24045") in by_rule_supplier

    # Severity actually carries through, not just the rule name.
    calc = next(i for i in issues if i.rule == "calc_error")
    assert calc.severity is Severity.ERROR
    missing = next(i for i in issues if i.rule == "missing_item")
    assert missing.severity is Severity.WARNING
    lump = next(i for i in issues if i.rule == "lump_sum")
    assert lump.severity is Severity.INFO


def test_detect_issues_leaves_one_out_so_a_supplier_never_flags_itself(rfq_qtys, reference_prices):
    # Two suppliers agree closely; neither should be flagged against ITS OWN price.
    a = SupplierQuote("A", "excl", None, [
        line(item_code="MESH-665", unit_price_ex_gst=D("80.00"), qty_canonical=D(14),
             line_total_ex_gst=D("1120.00"))])
    b = SupplierQuote("B", "excl", None, [
        line(item_code="MESH-665", unit_price_ex_gst=D("81.00"), qty_canonical=D(14),
             line_total_ex_gst=D("1134.00"))])
    issues = detect_issues([a, b], rfq_qtys, reference_prices)
    assert [i for i in issues if i.rule in ("typo", "outlier")] == []


def test_single_quote_falls_back_to_reference_price(rfq_qtys, reference_prices):
    # Only one supplier quotes LVL-24045, at roughly +30% over the $43.10 reference price.
    only = SupplierQuote("Solo", "excl", None, [
        line(item_code="LVL-24045", unit_price_ex_gst=D("56.00"), qty_canonical=D(36),
             line_total_ex_gst=D("2016.00"))])
    issues = detect_issues([only], rfq_qtys, reference_prices)
    assert any(i.rule == "outlier" and i.item_code == "LVL-24045" for i in issues)

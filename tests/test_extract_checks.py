"""Tests for the schema (src/schemas.py) and the checks on AI output (src/validate.py).

No API calls: these run offline and pin down what we accept from the model, using the real
RFQ codes and real lines from the supplier files.
"""
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from src import validate
from src.readers import SourceLine
from src.schemas import OUT_OF_RFQ, item_codes_from_rfq, make_models

CODES = item_codes_from_rfq()
ExtractedLine, ExtractedQuote = make_models(CODES)

# Northshore line 13 exactly as the reader produced it (the GIB AQ price has a typo: x10).
GIB_AQ = SourceLine(13, "GIB AQ 10 2.4 24 sht 687.30 16,495.20")


def extracted(**overrides):
    base = dict(line_no=13, item_code="GIB-AQ-10", match_confidence=0.95, qty=24,
                unit="sht", unit_price=687.30, line_total=16495.20, note=None,
                bundle_item_codes=[])
    return ExtractedLine(**{**base, **overrides})


def quote(**overrides):
    base = dict(supplier="X", gst_status="excl", quote_date=None, valid_until=None, lines=[],
                stated_total=None, stated_total_line_no=None)
    return ExtractedQuote(**{**base, **overrides})


# --- layer 1/2: the schema ---------------------------------------------------

def test_rfq_has_twenty_codes():
    assert len(CODES) == 20


def test_schema_only_offers_rfq_codes_plus_out_of_rfq():
    enum = ExtractedLine.model_json_schema()["properties"]["item_code"]["enum"]
    assert set(enum) == set(CODES) | {OUT_OF_RFQ}


def test_invented_item_code_is_rejected():
    with pytest.raises(ValidationError):
        extracted(item_code="GIB-AQ-12")


def test_out_of_rfq_is_allowed_for_charges_like_delivery():
    assert extracted(item_code=OUT_OF_RFQ).item_code == OUT_OF_RFQ


def test_not_quoted_line_keeps_none_instead_of_zero():
    line = extracted(qty=4, unit="pk", unit_price=None, line_total=None, note="N/Q")
    assert line.unit_price is None and line.line_total is None


def test_gst_status_must_be_one_of_three_values():
    with pytest.raises(ValidationError):
        ExtractedQuote(supplier="X", gst_status="maybe", quote_date=None, valid_until=None, lines=[])


# --- the grounding check -----------------------------------------------------

def test_numbers_in_handles_thousands_commas_and_trailing_zeros():
    numbers = validate.numbers_in(GIB_AQ.text)
    assert {D("687.3"), D("16495.2"), D("24"), D("2.4"), D("10")} <= numbers


def test_genuine_line_has_no_problems():
    assert validate.find_problems([extracted()], [GIB_AQ]) == []


def test_ai_silently_fixing_the_supplier_typo_is_caught():
    problems = validate.find_problems([extracted(unit_price=68.73)], [GIB_AQ])
    assert problems == ["line 13: unit_price=68.73 does not appear in the source text"]


def test_ai_doing_its_own_unit_conversion_is_caught():
    # Conversion is code's job. 518.4 m2 -> 180 sheets is not written on the line.
    line = SourceLine(12, "GIB STD 10 2.4 518.4 m2 12.25 6,350.40")
    problems = validate.find_problems([extracted(line_no=12, qty=180, unit_price=12.25, line_total=6350.40)], [line])
    assert problems == ["line 12: qty=180.0 does not appear in the source text"]


def test_number_from_a_different_row_does_not_count():
    other = SourceLine(14, "GIB BL 10 2.4 30 sht 56.63 1,698.90")
    # 56.63 is on line 14, but the AI claims it for line 13.
    problems = validate.find_problems([extracted(unit_price=56.63)], [GIB_AQ, other])
    assert any("unit_price=56.63" in p for p in problems)


def test_none_values_are_skipped_not_checked():
    line = SourceLine(23, "BAR CHAIRS 4 pk N/Q -")
    ok = extracted(line_no=23, item_code="CHAIR-5065", qty=4, unit="pk", unit_price=None, line_total=None)
    assert validate.find_problems([ok], [line]) == []


def test_same_item_code_used_twice_is_flagged():
    lines = [extracted(line_no=17, item_code="INS-R26"), extracted(line_no=18, item_code="INS-R26")]
    source = [SourceLine(17, "x"), SourceLine(18, "y")]
    problems = validate.find_problems(lines, source)
    assert any("INS-R26" in p and "2 lines" in p for p in problems)


def test_out_of_rfq_may_legitimately_repeat():
    # qty/unit_price/line_total are None here so this isolates the duplicate-code check
    # from the grounding check (which would otherwise fail against the trivial "x"/"y" text).
    kwargs = dict(qty=None, unit_price=None, line_total=None)
    lines = [extracted(line_no=1, item_code=OUT_OF_RFQ, **kwargs),
             extracted(line_no=2, item_code=OUT_OF_RFQ, **kwargs)]
    source = [SourceLine(1, "x"), SourceLine(2, "y")]
    assert validate.find_problems(lines, source) == []


def test_stated_total_with_no_value_is_not_checked():
    assert validate.check_stated_total(quote(stated_total=None), [GIB_AQ]) == []


def test_stated_total_grounded_against_its_own_line():
    total_line = SourceLine(31, "TOTAL 53892.31")
    q = quote(stated_total=53892.31, stated_total_line_no=31)
    assert validate.check_stated_total(q, [GIB_AQ, total_line]) == []


def test_stated_total_not_on_the_claimed_line_is_caught():
    total_line = SourceLine(31, "TOTAL 53892.31")
    q = quote(stated_total=99999.99, stated_total_line_no=31)   # invented number
    problems = validate.check_stated_total(q, [GIB_AQ, total_line])
    assert len(problems) == 1 and "99999.99" in problems[0]


def test_stated_total_on_the_line_below_its_label_is_accepted():
    # B's PDF really does this: label on line 28, value on line 29, model cites line 28.
    label, value = SourceLine(28, "TOTAL (incl GST)"), SourceLine(29, "$59,589.49")
    q = quote(stated_total=59589.49, stated_total_line_no=28)
    assert validate.check_stated_total(q, [label, value]) == []


def test_stated_total_two_lines_away_is_still_caught():
    label, gap, value = SourceLine(28, "TOTAL"), SourceLine(29, "-"), SourceLine(30, "59,589.49")
    q = quote(stated_total=59589.49, stated_total_line_no=28)
    assert validate.check_stated_total(q, [label, gap, value]) != []


def test_stated_total_without_a_line_reference_is_rejected():
    q = quote(stated_total=53892.31, stated_total_line_no=None)
    assert validate.check_stated_total(q, [GIB_AQ]) != []


def test_unknown_line_no_and_bad_confidence_are_reported():
    problems = validate.find_problems([extracted(line_no=99), extracted(match_confidence=1.7)], [GIB_AQ])
    assert "line 99: no such line in the source file" in problems
    assert any("outside 0..1" in p for p in problems)


def test_bundle_may_only_list_real_rfq_codes():
    with pytest.raises(ValidationError):
        extracted(item_code=OUT_OF_RFQ, bundle_item_codes=[OUT_OF_RFQ])


def test_bundle_on_a_line_matched_to_one_item_is_flagged():
    line = extracted(bundle_item_codes=["SEAL-123"])   # still item_code=GIB-AQ-10
    assert any("bundle_item_codes" in p for p in validate.find_problems([line], [GIB_AQ]))

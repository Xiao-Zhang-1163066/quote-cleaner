"""Tests for src/normalise.py.

Expected numbers come from data/answer_key/expected_normalised.csv, so a green run means
the conversions agree with the hand-checked answers, not just with themselves.
"""
from decimal import Decimal as D

import pytest

from src import normalise as n


# --- building blocks ---------------------------------------------------------

def test_to_decimal_goes_through_str_so_float_error_is_not_copied():
    assert n.to_decimal(0.1) == D("0.1")
    assert n.to_decimal(None) is None


def test_money_rounds_half_up_where_float_round_gets_it_wrong():
    assert round(2.675, 2) == 2.67          # float: the trap
    assert n.money(D("2.675")) == D("2.68")


def test_strip_gst_only_divides_when_prices_include_gst():
    assert n.money(n.strip_gst(D("7.50"), "incl")) == D("6.52")
    assert n.strip_gst(D("7.50"), "excl") == D("7.50")
    assert n.strip_gst(D("7.50"), "unstated") == D("7.50")


@pytest.mark.parametrize("text, expected", [
    ("FRAMING 90X45 SG8 H1.2 KD 6.0M", D("6.0")),
    ("WBOARD BEVELBACK 180X18 PP FJ 6.3M", D("6.3")),
    ("PINK BATTS ULTRA R3.2 WALL 9.6M2", None),   # 9.6 m2 is an area, not a length
    ("GIB STANDARD 10MM 2400X1200 TE", None),
])
def test_length_from_description(text, expected):
    assert n.length_from_description(text) == expected


# --- whole lines, one per case in the answer key ------------------------------

def line(**overrides):
    base = dict(raw_qty=None, raw_unit=None, raw_unit_price=None, raw_line_total=None,
                gst_status="excl", item_unit="each", description="")
    return n.normalise_line(**{**base, **overrides})


def test_timber_sold_by_the_stick_becomes_metres():          # Harbour, ex GST
    r = line(raw_qty=200, raw_unit="EA", raw_unit_price=36.71, raw_line_total=7342,
             item_unit="lm", description="FRAMING 90X45 SG8 H1.2 KD 6.0M")
    assert (r.qty, r.unit_price_ex_gst, r.line_total_ex_gst) == (D("1200.00"), D("6.12"), D("7342.00"))
    assert r.flags == (n.FLAG_LENGTH,)


def test_weatherboard_uses_its_own_length():                  # 6.3 m sticks -> 422.1 m
    r = line(raw_qty=67, raw_unit="EA", raw_unit_price=81.46, raw_line_total=5457.82,
             item_unit="lm", description="WBOARD BEVELBACK 180X18 PP FJ 6.3M")
    assert (r.qty, r.unit_price_ex_gst) == (D("422.10"), D("12.93"))


def test_gst_inclusive_price_is_divided_by_1_15():            # Northshore
    r = line(raw_qty=1200, raw_unit="m", raw_unit_price=7.50, raw_line_total=9000,
             gst_status="incl", item_unit="lm")
    assert (r.qty, r.unit_price_ex_gst, r.line_total_ex_gst) == (D("1200.00"), D("6.52"), D("7826.09"))
    assert r.flags == (n.FLAG_GST_INCL,)


def test_plasterboard_priced_per_m2_becomes_sheets():         # Northshore GIB STD
    r = line(raw_qty=518.4, raw_unit="m2", raw_unit_price=12.25, raw_line_total=6350.40,
             gst_status="incl", item_unit="sheet")
    assert (r.qty, r.unit_price_ex_gst, r.line_total_ex_gst) == (D("180.00"), D("30.68"), D("5522.09"))
    assert r.flags == (n.FLAG_AREA, n.FLAG_GST_INCL)


def test_unstated_gst_is_assumed_ex_but_flagged():            # Kiwi Frame, no total given
    r = line(raw_qty=1100, raw_unit="m", raw_unit_price=5.75, gst_status="unstated", item_unit="lm")
    assert (r.unit_price_ex_gst, r.line_total_ex_gst) == (D("5.75"), D("6325.00"))
    assert r.flags == (n.FLAG_GST_UNSTATED,)


def test_total_only_line_derives_the_unit_price():            # Huaxin mesh, Chinese unit
    r = line(raw_qty=14, raw_unit="张", raw_line_total=1270, gst_status="incl", item_unit="sheet")
    assert (r.unit_price_ex_gst, r.line_total_ex_gst) == (D("78.88"), D("1104.35"))
    assert r.flags == (n.FLAG_TOTAL_ONLY, n.FLAG_GST_INCL)


def test_supplier_arithmetic_error_is_carried_through_not_fixed():   # Harbour MESH 665
    r = line(raw_qty=14, raw_unit="SHT", raw_unit_price=80.01, raw_line_total=11201.4,
             item_unit="sheet")
    assert r.line_total_ex_gst == D("11201.40")   # 14 x 80.01 is 1120.14; the rules stage must catch it


def test_ea_on_a_non_timber_item_is_a_plain_count():
    r = line(raw_qty=180, raw_unit="EA", raw_unit_price=27.88, raw_line_total=5018.4, item_unit="sheet")
    assert (r.qty, r.unit_price_ex_gst, r.flags) == (D("180.00"), D("27.88"), ())


def test_chinese_bao_fits_bag_or_pack_but_not_roll():
    assert line(raw_qty=4, raw_unit="包", raw_unit_price=50, item_unit="pack").qty == D("4.00")
    assert line(raw_qty=60, raw_unit="包", raw_unit_price=9, item_unit="bag").qty == D("60.00")
    with pytest.raises(n.UnitError):
        line(raw_qty=6, raw_unit="包", raw_unit_price=200, item_unit="roll")


# --- refusing to guess -------------------------------------------------------

@pytest.mark.parametrize("raw_unit, item_unit, description", [
    ("furlong", "lm", ""),                    # unknown unit
    (None, "lm", ""),                         # no unit at all
    ("EA", "lm", "FRAMING 90X45 (no length)"),  # per-stick price but no length to convert with
    ("roll", "box", ""),                      # known unit, wrong kind for this item
])
def test_unconvertible_units_raise_instead_of_guessing(raw_unit, item_unit, description):
    with pytest.raises(n.UnitError):
        line(raw_qty=1, raw_unit=raw_unit, raw_unit_price=1, item_unit=item_unit, description=description)

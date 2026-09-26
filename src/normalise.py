"""Unit and GST conversion. Pure functions: same input, same output, no I/O, no AI.

This module is the reason the AI is not allowed to do arithmetic. An LLM is right most of
the time and occasionally wrong, and "occasionally" is the hardest kind of bug to notice.
Here every conversion is a few lines of Decimal maths that a unit test can pin down.

Deliberate non-goal: this module does NOT fix supplier mistakes. If the supplier's line
total is wrong (Harbour's MESH 665), we carry the wrong total through untouched so that
the rules stage (Phase 5) can detect and report it. Silently "correcting" it would hide
exactly the finding the tool exists to surface.
"""
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from src import config

CENT = Decimal("0.01")

# Flag wording matches answer_key/expected_normalised.csv so the evaluation can compare.
FLAG_GST_INCL = "gst:incl→ex"
FLAG_GST_UNSTATED = "gst:unstated (assumed ex)"
FLAG_LENGTH = "unit:length→lm"
FLAG_AREA = f"unit:m2→sheet ({config.SHEET_AREA_M2} m2/sheet)"
FLAG_TOTAL_ONLY = "total only (unit price derived)"


class UnitError(ValueError):
    """A unit we cannot convert safely. The caller sends the line to a human."""


@dataclass(frozen=True)
class NormalisedLine:
    qty: Decimal | None
    unit_price_ex_gst: Decimal | None
    line_total_ex_gst: Decimal | None
    flags: tuple[str, ...]


def to_decimal(value) -> Decimal | None:
    if value is None:
        return None
    # str() first: Decimal(0.1) copies the float's binary error (0.1000000000000000055...),
    # Decimal("0.1") is exactly 0.1. This one line is the whole float-vs-Decimal lesson.
    return value if isinstance(value, Decimal) else Decimal(str(value))


def money(value: Decimal) -> Decimal:
    # ROUND_HALF_UP is what people expect from "round to cents" (2.675 -> 2.68).
    # Decimal's default is banker's rounding, which surprises in a price comparison.
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def strip_gst(amount: Decimal, gst_status: str) -> Decimal:
    # Only 'incl' is divided. 'unstated' is treated as ex-GST (industry habit per the spec);
    # the flag added by normalise_line is what stops that assumption being invisible.
    if gst_status == "incl":
        return amount / (1 + config.GST_RATE)
    return amount


def length_from_description(text: str) -> Decimal | None:
    """Read a stick length such as '6.0M' from a product name. None if there isn't one."""
    # The trailing \b matters: it rejects "9.6M2" (an AREA, 9.6 square metres of batts)
    # because a digit follows the M, so there is no word boundary.
    match = re.search(r"(\d+(?:\.\d+)?)\s*m\b", text, flags=re.IGNORECASE)
    return Decimal(match.group(1)) if match else None


def canonical_units_per_raw_unit(raw_unit: str | None, item_unit: str,
                                 description: str) -> tuple[Decimal, list[str]]:
    """How many of the RFQ's unit one raw unit is worth, plus flags describing the trick used."""
    if raw_unit is None:
        raise UnitError("no unit given")
    key = raw_unit.strip().lower()
    if key not in config.UNIT_ALIASES:
        raise UnitError(f"unknown unit {raw_unit!r}")
    meaning = config.UNIT_ALIASES[key]

    if meaning is None:                       # "EA": a count of whatever the item is
        if item_unit != "lm":
            return Decimal(1), []
        # Timber is priced per stick but the RFQ wants metres, so one stick = its length.
        length = length_from_description(description)
        if length is None:
            raise UnitError(f"'{raw_unit}' on a per-metre item but no length in {description!r}")
        return length, [FLAG_LENGTH]

    if meaning == "m2" and item_unit == "sheet":
        return 1 / config.SHEET_AREA_M2, [FLAG_AREA]

    allowed = meaning if isinstance(meaning, tuple) else (meaning,)
    if item_unit in allowed:
        return Decimal(1), []

    # Refuse to guess (e.g. a quote in 'roll' for an item we buy by the box).
    raise UnitError(f"cannot convert {raw_unit!r} to {item_unit!r}")


def normalise_line(*, raw_qty, raw_unit, raw_unit_price, raw_line_total,
                   gst_status: str, item_unit: str, description: str) -> NormalisedLine:
    """Convert one raw quote line to the RFQ's unit and to ex-GST prices."""
    qty = to_decimal(raw_qty)
    price = to_decimal(raw_unit_price)
    total = to_decimal(raw_line_total)
    flags: list[str] = []

    factor, unit_flags = canonical_units_per_raw_unit(raw_unit, item_unit, description)
    flags += unit_flags

    # Some suppliers give only a total ("一共 $1270"). Derive the unit price from it and say so.
    if price is None and total is not None and qty:
        price = total / qty
        flags.append(FLAG_TOTAL_ONLY)
    # Others give only a unit price (Kiwi Frame's email), so the total is qty x price.
    if total is None and price is not None and qty is not None:
        total = qty * price

    if gst_status == "incl":
        flags.append(FLAG_GST_INCL)
    elif gst_status == "unstated":
        flags.append(FLAG_GST_UNSTATED)

    # Keep full Decimal precision through every step and round ONCE at the end. Rounding
    # in between (say, the ex-GST price before multiplying) accumulates cents of error.
    return NormalisedLine(
        qty=money(qty * factor) if qty is not None else None,
        unit_price_ex_gst=money(strip_gst(price, gst_status) / factor) if price is not None else None,
        line_total_ex_gst=money(strip_gst(total, gst_status)) if total is not None else None,
        flags=tuple(flags),
    )

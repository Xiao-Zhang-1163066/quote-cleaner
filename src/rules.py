"""FR-6: issue detection. Pure functions, no AI, no I/O. Thresholds live in config.py.

Two different "versions" of the numbers feed this module, and mixing them up would cause
real bugs, not just style issues:
  - RAW numbers (the supplier's own qty/unit_price/line_total, exactly as extracted, BEFORE
    Phase 3's unit/GST conversion) check the SUPPLIER'S OWN arithmetic (calc_error,
    total_mismatch). Using the converted numbers here would run the check against values we
    have already rounded to cents, and that rounding can itself manufacture a few cents of
    "error" that isn't the supplier's fault -- exactly the false positive the acceptance
    criteria (0 false-positive ERROR findings) forbid.
  - NORMALISED numbers (Phase 3's ex-GST, canonical-unit output) are for anything that
    COMPARES across suppliers or against the RFQ (typo/outlier pricing, qty vs RFQ, missing
    items), because only normalised numbers are on the same footing -- comparing one
    supplier's GST-inclusive raw price against another's GST-exclusive raw price would be
    comparing two different things.
"""
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from src import config
from src.schemas import OUT_OF_RFQ


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    rule: str
    severity: Severity
    message: str
    supplier: str
    item_code: str | None = None


# ---------------------------------------------------------------------------------
# RAW-number rules: does the supplier's OWN arithmetic check out, in their own file?
# ---------------------------------------------------------------------------------

def calc_error(qty: Decimal | None, unit_price: Decimal | None, line_total: Decimal | None) -> bool:
    """True if qty x unit_price disagrees with the printed line_total by more than tolerance.

    All three arguments must be RAW (pre-conversion) numbers -- see the module docstring.
    """
    if qty is None or unit_price is None or line_total is None:
        return False   # a total-only or price-only line has nothing to cross-check
    return abs(qty * unit_price - line_total) > config.CALC_TOLERANCE


def total_mismatch(raw_line_totals: list[Decimal], stated_total: Decimal | None) -> bool:
    """True if the file's own printed grand total disagrees with the sum of its own lines."""
    if stated_total is None:
        return False   # no printed total in this file to check against
    return abs(sum(raw_line_totals, Decimal(0)) - stated_total) > config.CALC_TOLERANCE


# ---------------------------------------------------------------------------------
# Normalised-number rules: cross-supplier and RFQ comparisons
# ---------------------------------------------------------------------------------

def _median(values: list[Decimal]) -> Decimal:
    s = sorted(values)
    mid = len(s) // 2
    # Even count: average the two middle values. Decimal / int stays exact (no float involved).
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def price_band(price: Decimal, other_prices: list[Decimal]) -> str | None:
    """Classify a normalised unit price against OTHER suppliers' price for the same item
    (this supplier excluded -- "leave-one-out", so a supplier is never compared to itself).

    Returns "typo", "outlier", or None. The two bands are defined not to overlap (outlier
    is explicitly "under 3x"), so a price is classified into at most one band, never both.
    """
    if not other_prices:
        return None   # caller falls back to reference_prices.csv when only one quote exists
    median = _median(other_prices)
    if median == 0:
        return None   # can't compute a meaningful ratio against a zero price
    ratio = price / median
    if ratio >= config.TYPO_FACTOR or ratio <= 1 / config.TYPO_FACTOR:
        return "typo"
    if ratio >= 1 + config.OUTLIER_PCT:
        return "outlier"
    return None


def qty_short(qty_canonical: Decimal | None, rfq_qty: Decimal) -> bool:
    return qty_canonical is not None and qty_canonical < rfq_qty


def is_substitution(note: str | None) -> bool:
    """Per spec: note mentions ALT/equiv/替代. A brand differing from the RFQ's own brand
    would also qualify, but detecting that needs the AI's understanding, not code -- Phase
    4's extraction is expected to say so in `note`, which is what this checks."""
    if not note:
        return False
    lowered = note.lower()
    return any(kw in lowered for kw in ("alt", "equiv", "替代"))


def is_lump_sum(item_code: str | None, qty_canonical: Decimal | None,
               line_total_ex_gst: Decimal | None) -> bool:
    """A priced charge with NO quantity, outside the RFQ, looks like several items bundled
    into one price (Kiwi Frame's nails+Sikaflex package: qty null, total $980) rather than a
    single identifiable extra charge (Northshore's delivery: qty=1, a normal count).
    A heuristic, not a certainty -- this is exactly the kind of line low confidence should
    also route to a human, which Phase 4's match_confidence already does independently.
    """
    return item_code == OUT_OF_RFQ and qty_canonical is None and line_total_ex_gst is not None


# ---------------------------------------------------------------------------------
# Orchestration: run every rule over every supplier's lines for one RFQ
# ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class SupplierLine:
    """One supplier's line: both the raw numbers (arithmetic checks) and the normalised
    numbers (comparison checks) it needs, plus the context those checks require."""
    item_code: str | None            # an RFQ code, OUT_OF_RFQ, or None if truly unmatched
    raw_qty: Decimal | None
    raw_unit_price: Decimal | None
    raw_line_total: Decimal | None
    qty_canonical: Decimal | None
    unit_price_ex_gst: Decimal | None
    line_total_ex_gst: Decimal | None
    note: str | None = None
    is_total_only: bool = False      # True when normalise.py set FLAG_TOTAL_ONLY on this line
    bundle_item_codes: tuple[str, ...] = ()   # RFQ items this line's single price covers


def is_priced(line: SupplierLine) -> bool:
    """Did this line actually put a price on its item? "BAR CHAIRS 4 pk N/Q" is correctly
    matched to CHAIR-5065 -- knowing WHICH item wasn't priced is useful -- but it is not a
    quote for it. The one definition of "quoted", shared by the missing_item rule and by
    evaluate.py so the two can never disagree about it."""
    return line.unit_price_ex_gst is not None or line.line_total_ex_gst is not None


@dataclass(frozen=True)
class SupplierQuote:
    supplier: str
    gst_status: str                  # "incl" / "excl" / "unstated"
    stated_total: Decimal | None
    lines: list[SupplierLine]


def detect_issues(quotes: list[SupplierQuote], rfq_qtys: dict[str, Decimal],
                  reference_prices: dict[str, Decimal]) -> list[Issue]:
    """Run every rule across all suppliers for one RFQ. Order doesn't matter downstream
    (Phase 6's export groups by severity/item), so issues are appended as each is found."""
    issues: list[Issue] = []

    for quote in quotes:
        if quote.gst_status == "unstated":
            issues.append(Issue("gst_unstated", Severity.WARNING,
                                "no GST statement anywhere in this quote", quote.supplier))

        raw_totals = [l.raw_line_total for l in quote.lines if l.raw_line_total is not None]
        if total_mismatch(raw_totals, quote.stated_total):
            issues.append(Issue("total_mismatch", Severity.ERROR,
                                f"printed total {quote.stated_total} != sum of the quote's "
                                f"own line totals ({sum(raw_totals, Decimal(0))})", quote.supplier))

        seen_codes: set[str] = set()
        for line in quote.lines:
            code = line.item_code
            if code and code != OUT_OF_RFQ and is_priced(line):
                seen_codes.add(code)
            # Items inside a bundle WERE quoted, just not one by one -- reporting them as
            # missing as well would tell the reader the same thing twice, and wrongly.
            seen_codes.update(line.bundle_item_codes)

            if calc_error(line.raw_qty, line.raw_unit_price, line.raw_line_total):
                issues.append(Issue(
                    "calc_error", Severity.ERROR,
                    f"{line.raw_qty} x {line.raw_unit_price} = "
                    f"{None if line.raw_qty is None or line.raw_unit_price is None else line.raw_qty * line.raw_unit_price}"
                    f", but printed as {line.raw_line_total}", quote.supplier, code))

            if is_substitution(line.note):
                issues.append(Issue("substitution", Severity.WARNING, line.note,
                                    quote.supplier, code))

            if is_lump_sum(code, line.qty_canonical, line.line_total_ex_gst):
                covers = ", ".join(line.bundle_item_codes) or "unspecified items"
                issues.append(Issue("lump_sum", Severity.INFO,
                                    f"one price covers {covers}; not split per item",
                                    quote.supplier, code))
            # Only a line with a price is a CHARGE. "Delivery free to North Shore" has none,
            # so it is a remark, not a cost to flag.
            elif code == OUT_OF_RFQ and (line.line_total_ex_gst is not None
                                         or line.unit_price_ex_gst is not None):
                issues.append(Issue("out_of_rfq", Severity.INFO,
                                    "charge not in the RFQ", quote.supplier, code))

            if line.is_total_only:
                issues.append(Issue("total_only", Severity.INFO,
                                    "unit price derived from a total-only line",
                                    quote.supplier, code))

            if code and code != OUT_OF_RFQ and code in rfq_qtys:
                if qty_short(line.qty_canonical, rfq_qtys[code]):
                    issues.append(Issue(
                        "qty_short", Severity.WARNING,
                        f"{line.qty_canonical} of {rfq_qtys[code]} quoted",
                        quote.supplier, code))

        for code in rfq_qtys:
            if code not in seen_codes:
                issues.append(Issue("missing_item", Severity.WARNING,
                                    "not quoted by this supplier", quote.supplier, code))

    # Cross-supplier price comparison, one RFQ item at a time.
    by_item: dict[str, list[tuple[str, Decimal]]] = {}
    for quote in quotes:
        for line in quote.lines:
            if (line.item_code and line.item_code != OUT_OF_RFQ
                    and line.unit_price_ex_gst is not None):
                by_item.setdefault(line.item_code, []).append((quote.supplier, line.unit_price_ex_gst))

    for code, priced in by_item.items():
        for supplier, price in priced:
            others = [p for s, p in priced if s != supplier]
            if not others and code in reference_prices:
                others = [reference_prices[code]]   # only one quote: compare to public retail
            band = price_band(price, others)
            # Leave-one-out alone cannot tell WHO is wrong when there is only one other
            # quote and THAT one has the typo: the ratio is symmetric, so the honest price
            # looks exactly as extreme as the dishonest one (found by actually running this
            # against real data -- GIB-AQ-10, where B's price was 10x too high and A's
            # genuinely correct price got flagged right along with it). Break the tie with
            # the reference price as an INDEPENDENT third source -- not blended into the
            # median above, which a real typo would just drag along with it -- if this
            # supplier's own price looks ordinary next to the public retail price, the typo
            # is on the other side, not here.
            if band == "typo" and code in reference_prices:
                if price_band(price, [reference_prices[code]]) is None:
                    band = None
            if band:
                severity = Severity.ERROR if band == "typo" else Severity.WARNING
                baseline = _median(others) if others else None
                issues.append(Issue(band, severity,
                                    f"{price} vs other suppliers' median {baseline}",
                                    supplier, code))

    return issues

"""The exact shape the AI must return. Nothing outside this shape gets into the database.

Layered defence against a made-up item code:
  1. This schema declares item_code as one of the RFQ codes (a JSON-schema enum), so the API
     can constrain the model while it generates.
  2. Pydantic re-validates the reply in our own code. We never trust a vendor guarantee for
     a business rule, and this layer also protects us if the model or API is swapped.
  3. The database foreign key (Phase 1) is the last resort.
"""
import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src import config

OUT_OF_RFQ = "OUT_OF_RFQ"   # a real charge that is not an RFQ item, e.g. delivery


def item_codes_from_rfq(rfq_csv: Path = config.RFQ_CSV) -> list[str]:
    with open(rfq_csv, newline="", encoding="utf-8") as f:
        return [row["item_code"] for row in csv.DictReader(f)]


def make_models(item_codes: list[str]):
    """Build the line and quote models for a given list of allowed item codes.

    Built by a function because the allowed codes come from the RFQ file, not from source
    code: a different RFQ means a different set, with no schema edit.
    """
    # Literal[("A", "B")] is the same as Literal["A", "B"]; a tuple lets us build it from data.
    code_type = Literal[tuple(item_codes) + (OUT_OF_RFQ,)]

    class ExtractedLine(BaseModel):
        # Which line of the source file this came from. raw_text is deliberately NOT here:
        # it is copied from the reader's output by line_no, so the AI has nothing to invent.
        line_no: int
        item_code: code_type
        # 0..1. Range is checked in validate.py, not in the schema: some structured-output
        # modes accept only a subset of JSON-Schema keywords, and this way it works everywhere.
        match_confidence: float
        # Copied from the text, never computed. Floats here are fine because they are turned
        # into Decimal via str() straight away (normalise.to_decimal); None means "not quoted".
        qty: float | None
        unit: str | None
        unit_price: float | None
        line_total: float | None
        note: str | None          # e.g. "ALT - Earthwool R2.6 n/a" or "N/Q"

    class ExtractedQuote(BaseModel):
        supplier: str
        gst_status: Literal["incl", "excl", "unstated"]   # "unstated" is a finding, not a default
        quote_date: str | None    # as written; parsing dates is not the AI's job
        valid_until: str | None
        lines: list[ExtractedLine]

    return ExtractedLine, ExtractedQuote

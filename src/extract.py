"""FR-2/FR-3: turn one supplier file's raw lines into structured, checked data.

This is the ONLY module in the whole pipeline that calls an LLM. Everything it returns
still has to survive validate.find_problems before anything downstream trusts it — the
model's job here is understanding messy product names, not being the source of truth.
"""
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from src import config, validate
from src.readers import SourceLine
from src.schemas import item_codes_from_rfq, make_models, rfq_catalog

ITEM_CODES = item_codes_from_rfq()
ExtractedLine, ExtractedQuote = make_models(ITEM_CODES)

# The Literal in the schema gives the API a list of legal SYMBOLS ("JNT-COMP"), which is
# enough to block an invented code, but not enough to match against: it carries no meaning.
# Spelling that meaning out here is what actually lets the model do the matching (FR-3).
_CATALOG_TEXT = "\n".join(
    f"{r['item_code']}: {r['description']} (unit: {r['unit']})" for r in rfq_catalog()
)

# The one instruction that matters most: an LLM's instinct is to produce a complete-looking
# answer, so without this line it will invent a plausible price for a line marked "N/Q".
# Everything else here (units to pick from, examples of the tricks in this data) helps
# accuracy; this sentence is what stops confident invention.
INSTRUCTIONS = f"""\
You are extracting supplier quotation lines for a building-supplies RFQ comparison tool.

You will be given the numbered lines of ONE supplier's quote file, in order. Extract every
priced line item, plus the quote's own header/footer facts (supplier name, GST statement,
date, validity).

RFQ item catalog (the ONLY items this comparison covers; match against these descriptions,
not against the bare code):
{_CATALOG_TEXT}

Rules:
- Extract EVERY line that looks like a priced product row, even one you are unsure how to
  match. If you cannot find a good catalog match, still output the line with
  item_code=OUT_OF_RFQ and a low match_confidence. It is fine to guess wrong; it is NOT fine
  to silently skip a line, because a skipped line looks identical to "supplier didn't offer
  this item" and that is a different, wrong conclusion.
- If a value is not written on the line (e.g. marked "N/Q", or simply absent), leave that
  field as null. NEVER invent, estimate, or guess a number that is not on the page.
- item_code must be one of the {len(ITEM_CODES)} RFQ codes above, chosen by matching the
  supplier's product description to the catalog description (e.g. "GIB AQ 10", "石膏板",
  "Aqualine" all mean the same board; "CEMIX MULTICRETE 20KG" is the brand name for the
  catalog's "Cemix Multicrete 20kg bag"). If a charge is real but is not one of those items
  (e.g. a delivery fee), use OUT_OF_RFQ. Never invent a code that isn't in the allowed list.
- match_confidence is your own certainty in the item_code match, from 0 (guessing) to 1
  (certain). Abbreviations you must expand using context in the file (a footnote, a legend)
  can still be high confidence; a match with no supporting evidence should be low.
- Copy qty, unit, unit_price and line_total exactly as written (same numbers, do not convert
  units or compute a total yourself — that is done separately). If only a total is given for
  a line, still copy it into line_total and leave unit_price null.
- gst_status is "incl" only if the file states prices include GST, "excl" if it states
  exclusive, and "unstated" if it says nothing either way. Do not assume "excl" by default.
- note: copy any remark on the line (e.g. "ALT", "N/Q", a substitution, a lead time).
- line_no must be the exact line number given in the input, so results can be traced back
  to the source file.
- If the file has its own printed total (e.g. a "TOTAL" or "Subtotal" row), copy that number
  into stated_total and give stated_total_line_no as the line it appears on. If there is no
  such total anywhere in the file, leave both null — do not calculate one yourself.
"""


@dataclass(frozen=True)
class ExtractionResult:
    # Typed as Any, not ExtractedQuote: that class is built at RUNTIME by make_models()
    # from the RFQ file's codes, so a static type checker has no way to know its shape
    # ahead of time. This is a known, accepted gap in dynamic Pydantic models; the real
    # shape is enforced at runtime by Pydantic and pinned down by tests, not by Pylance.
    quote: Any   # an ExtractedQuote instance, or None if extraction ultimately failed
    problems: list[str]            # grounding/validation problems on the LAST attempt
    attempts: int
    model: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    error: str | None = None       # set when quote is None: why this file needs a human


def _render(source_lines: list[SourceLine]) -> str:
    # Same "line_no: text" shape as pipeline.py --read, so what the model sees is exactly
    # what you can reproduce and inspect yourself, line by line.
    return "\n".join(f"{s.line_no}: {s.text}" for s in source_lines)


def _call_openai(client: OpenAI, source_lines: list[SourceLine], prior_problems: list[str]):
    input_text = _render(source_lines)
    if prior_problems:
        # Feed the retry the SAME grounding check it failed, in plain words, so it can fix
        # the specific line instead of guessing what "try again" means.
        input_text += (
            "\n\nYour previous attempt had these problems, fix them:\n"
            + "\n".join(f"- {p}" for p in prior_problems)
        )
    return client.responses.parse(
        model=config.MODEL_NAME,
        instructions=INSTRUCTIONS,
        input=input_text,
        text_format=ExtractedQuote,
    )


def extract_quote(source_lines: list[SourceLine], client: OpenAI | None = None,
                  max_attempts: int = 2) -> ExtractionResult:
    """Extract and validate one quote file. Retries once on a validation failure (FR-2's
    "validate; on failure retry once; on a second failure, flag for human review")."""
    client = client or OpenAI(api_key=config.OPENAI_API_KEY)
    input_tokens = output_tokens = 0
    problems: list[str] = []

    for attempt in range(1, max_attempts + 1):
        start = time.monotonic()
        try:
            response = _call_openai(client, source_lines, problems)
        except Exception as exc:   # network error, rate limit, etc: not worth a retry loop here
            return ExtractionResult(None, [str(exc)], attempt, config.MODEL_NAME,
                                    input_tokens, output_tokens, time.monotonic() - start,
                                    error=f"API call failed: {exc}")
        elapsed = time.monotonic() - start
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        quote = response.output_parsed
        # The schema's Literal already blocks a code outside our list; this second pass
        # (Pydantic re-validating our own dataclass rules, not the API's) is the numbers
        # grounding check, which the schema has no way to express.
        try:
            problems = (validate.find_problems(quote.lines, source_lines)
                       + validate.check_stated_total(quote, source_lines))
        except ValidationError as exc:   # belt-and-braces: malformed even after API-side parsing
            problems = [str(exc)]

        if not problems:
            return ExtractionResult(quote, [], attempt, config.MODEL_NAME,
                                    input_tokens, output_tokens, elapsed)
        # Retry once, telling the model exactly what it got wrong, before giving up on this file.

    return ExtractionResult(None, problems, max_attempts, config.MODEL_NAME,
                            input_tokens, output_tokens, elapsed,
                            error=f"failed validation after {max_attempts} attempts: {problems}")

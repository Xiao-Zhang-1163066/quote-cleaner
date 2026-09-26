"""Every tunable number and path lives here, and nowhere else.

Rule of thumb: thresholds are *policy* (a business person might want to change them),
logic is *mechanism* (only a developer should). Keeping them apart means a demo-time
"what if the outlier limit were 20%?" is a one-line edit, not a code search.
"""
import os
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths from this file's location, not from the current working directory.
# Otherwise the tool works when run from the project root and breaks when run from
# anywhere else (e.g. by pytest or by the Streamlit app later).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The mock data is committed inside the repo so a fresh `git clone` runs with no extra
# setup. It is read-only input: nothing in the pipeline ever writes here.
# Rule: only evaluate.py may read ANSWER_KEY_CSV. If the pipeline could see the answers,
# the accuracy score would be meaningless (the model would be marking its own homework).
DATA_DIR = PROJECT_ROOT / "data"
QUOTES_DIR = DATA_DIR / "quotes"
RFQ_CSV = DATA_DIR / "rfq_items.csv"
REFERENCE_PRICES_CSV = DATA_DIR / "reference_prices.csv"
ANSWER_KEY_CSV = DATA_DIR / "answer_key" / "expected_normalised.csv"

# Generated files go in one git-ignored folder so it is obvious what is input vs output.
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_PATH = OUTPUT_DIR / "quotes.db"

# Reads .env into environment variables. It never overrides variables already set in
# the shell, so CI or a production host can inject the key without any .env file.
load_dotenv(PROJECT_ROOT / ".env")

# Only ever read from the environment: no default, so a missing key is loud, not silent.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model name is config, not code, so swapping models (cost vs accuracy) needs no edit.
# We'll confirm the exact ID against current docs in Phase 4 before first use.
MODEL_NAME = os.getenv("QUOTE_MODEL", "claude-sonnet-5")

# --- Business rules -------------------------------------------------------------
# All money arithmetic uses Decimal, never float (0.1 + 0.2 != 0.3 in float, and errors
# of a cent break comparisons against the answer key). Build Decimals from STRINGS:
# Decimal(0.15) would carry the float's binary error along with it.
GST_RATE = Decimal("0.15")   # NZ GST

# Conversion factors are data, not logic: changing the board size is a one-line edit here.
SHEET_AREA_M2 = Decimal("2.88")   # 2400 x 1200 mm plasterboard

# Raw unit spelling -> what it means. A string is one canonical unit. None means "a count
# of whatever this item is sold in" ("EA" on a GIB sheet, on a pail, or on a 6 m length).
# A tuple means "one of these" (Chinese 包 is used for both bags and packs).
# Keys are lowercase; the code lowercases the input before looking it up.
UNIT_ALIASES = {
    "lm": "lm", "m": "lm", "米": "lm",
    "m2": "m2",
    "sheet": "sheet", "sht": "sheet", "张": "sheet",
    "pack": "pack", "pk": "pack",
    "bag": "bag",
    "包": ("bag", "pack"),
    "roll": "roll", "rl": "roll", "卷": "roll",
    "box": "box", "bx": "box", "箱": "box",
    "tube": "tube", "支": "tube",
    "pail": "pail", "桶": "pail",
    "each": None, "ea": None, "根": None,
}

# --- Issue-detection thresholds (used from Phase 5) -----------------------------
CALC_TOLERANCE = 0.05          # $: qty x unit price vs line total may differ by this much
TYPO_FACTOR = 3.0              # price > 3x or < 1/3 of other suppliers' median => likely typo (error)
OUTLIER_PCT = 0.25             # price > 25% above the median => outlier (warning)

# --- AI trust boundary (used from Phase 4) --------------------------------------
# Below this, a match is queued for a human instead of being counted automatically.
MATCH_CONFIDENCE_MIN = 0.8

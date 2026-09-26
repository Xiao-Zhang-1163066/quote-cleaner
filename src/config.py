"""Every tunable number and path lives here, and nowhere else.

Rule of thumb: thresholds are *policy* (a business person might want to change them),
logic is *mechanism* (only a developer should). Keeping them apart means a demo-time
"what if the outlier limit were 20%?" is a one-line edit, not a code search.
"""
import os
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
# NZ GST. Kept as a plain number for now; Phase 3 decides how money should be
# represented (float vs Decimal) before any arithmetic depends on it.
GST_RATE = 0.15

# --- Issue-detection thresholds (used from Phase 5) -----------------------------
CALC_TOLERANCE = 0.05          # $: qty x unit price vs line total may differ by this much
TYPO_FACTOR = 3.0              # price > 3x or < 1/3 of other suppliers' median => likely typo (error)
OUTLIER_PCT = 0.25             # price > 25% above the median => outlier (warning)

# --- AI trust boundary (used from Phase 4) --------------------------------------
# Below this, a match is queued for a human instead of being counted automatically.
MATCH_CONFIDENCE_MIN = 0.8

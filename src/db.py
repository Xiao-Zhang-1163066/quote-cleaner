"""SQLite storage: schema, connection, and idempotent (re-runnable) writes.

Design rule: raw text and normalised numbers live in separate tables. Every normalised
number can be traced back to the exact line of the supplier's original file.
The stdlib sqlite3 module is enough; no ORM, so every SQL statement is visible and
explainable in an interview.

Transactions: none of these functions commit. The caller wraps a whole stage in
`with conn:` so it either fully succeeds or is rolled back (no half-written state).
"""
import csv
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from src import config
from src.normalise import FLAG_TOTAL_ONLY, to_decimal
from src.rules import SupplierLine, SupplierQuote
from src.schemas import OUT_OF_RFQ

# Money columns are REAL because SQLite has no decimal type. That is safe here only because
# all arithmetic happens in Python with Decimal (normalise.py) and values are rounded to
# cents BEFORE they are stored. Never do money maths in SQL (SUM over REAL); sum in Python.
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_code         TEXT PRIMARY KEY,   -- e.g. FRM-9045-H12: the 20 codes the AI may choose from
    description       TEXT NOT NULL,
    unit              TEXT NOT NULL,      -- canonical unit: lm, sheet, pack ...
    rfq_qty           REAL NOT NULL,
    ref_price_ex_gst  REAL NOT NULL       -- public retail benchmark, used when few quotes exist
);

CREATE TABLE IF NOT EXISTS quotes (
    quote_id     INTEGER PRIMARY KEY,
    supplier     TEXT NOT NULL,
    -- UNIQUE is what makes re-running safe: the same file always maps to the same row.
    -- Store the file NAME, not an absolute path, so the key is the same on every machine.
    source_file  TEXT NOT NULL UNIQUE,
    quote_date   TEXT,
    -- 'unstated' is its own value on purpose: silence about GST is a finding, not "excl".
    gst_status   TEXT NOT NULL CHECK (gst_status IN ('incl', 'excl', 'unstated')),
    valid_until  TEXT,
    -- The quote's OWN printed grand total, exactly as extracted (same gst_status as the
    -- lines above -- not yet ex-GST). Phase 1 had no column for this; rules.total_mismatch
    -- needs it to compare against a freshly recomputed sum of the lines, so it has to be
    -- stored somewhere, and it's a fact about the whole quote, not any one line.
    stated_total          REAL,
    stated_total_line_no  INTEGER   -- traceability: which line of the source file it came from
);

-- Exactly what the supplier wrote. Never modified by later stages.
CREATE TABLE IF NOT EXISTS quote_lines_raw (
    line_id          INTEGER PRIMARY KEY,
    quote_id         INTEGER NOT NULL REFERENCES quotes(quote_id),
    line_no          INTEGER NOT NULL,    -- position inside the file: stable, unlike AI-read text
    raw_text         TEXT NOT NULL,
    raw_qty          REAL,
    raw_unit         TEXT,
    raw_unit_price   REAL,
    raw_line_total   REAL,
    raw_note         TEXT,
    -- Natural key. quote_id already implies source_file (it is UNIQUE above), so this is
    -- the spec's (source_file, line_no) without repeating the long text on every row.
    UNIQUE (quote_id, line_no)
);

-- Our interpretation of one raw line. Rebuilt any time rules or prompts change.
CREATE TABLE IF NOT EXISTS quote_lines_norm (
    -- 1:1 with a raw line, so the raw line's id doubles as the primary key.
    line_id             INTEGER PRIMARY KEY REFERENCES quote_lines_raw(line_id),
    -- NULL means "not in the RFQ" (e.g. delivery fee). The foreign key then guarantees
    -- any non-NULL code really is one of the 20 items, even if the AI invents another.
    item_code           TEXT REFERENCES items(item_code),
    match_confidence    REAL,
    confirmed_by_human  INTEGER NOT NULL DEFAULT 0,
    qty_canonical       REAL,
    unit_price_ex_gst   REAL,
    line_total_ex_gst   REAL,
    flags               TEXT
);

-- Which RFQ items one bundled price covers (C's "nails + Sikaflex, $980 all up"). One line
-- can cover many items, so this is its own table rather than a column; the item_code FK
-- means the AI can't put an invented code in a bundle either.
CREATE TABLE IF NOT EXISTS bundle_items (
    line_id    INTEGER NOT NULL REFERENCES quote_lines_raw(line_id),
    item_code  TEXT NOT NULL REFERENCES items(item_code),
    PRIMARY KEY (line_id, item_code)
);
"""


def connect(db_path: Path = config.DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # GOTCHA: SQLite ignores foreign keys unless you switch this on for EVERY connection.
    # Without it the REFERENCES clauses above are decoration and bad codes get in silently.
    conn.execute("PRAGMA foreign_keys = ON")
    # Rows behave like dicts (row["item_code"]) so the code doesn't depend on column order.
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    # IF NOT EXISTS makes this safe to call on every run.
    conn.executescript(SCHEMA)


def load_items(conn: sqlite3.Connection,
               rfq_csv: Path = config.RFQ_CSV,
               ref_csv: Path = config.REFERENCE_PRICES_CSV) -> int:
    """Load the 20 RFQ items plus their reference prices. Returns how many were loaded."""
    with open(ref_csv, newline="", encoding="utf-8") as f:
        ref_prices = {row["item_code"]: float(row["ref_price_ex_gst_per_canonical_unit"])
                      for row in csv.DictReader(f)}

    with open(rfq_csv, newline="", encoding="utf-8") as f:
        rfq_rows = list(csv.DictReader(f))

    for row in rfq_rows:
        # Fail loudly: an item with no reference price would make every later price
        # comparison quietly wrong, so stop here rather than store a NULL.
        if row["item_code"] not in ref_prices:
            raise ValueError(f"No reference price for {row['item_code']} in {ref_csv.name}")

    # UPSERT: insert, or if the code already exists, update it. Re-running changes nothing.
    conn.executemany(
        """
        INSERT INTO items (item_code, description, unit, rfq_qty, ref_price_ex_gst)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(item_code) DO UPDATE SET
            description = excluded.description,
            unit = excluded.unit,
            rfq_qty = excluded.rfq_qty,
            ref_price_ex_gst = excluded.ref_price_ex_gst
        """,
        [(r["item_code"], r["description"], r["unit"], float(r["qty"]),
          ref_prices[r["item_code"]]) for r in rfq_rows],
    )
    return len(rfq_rows)


def upsert_quote(conn: sqlite3.Connection, supplier: str, source_file: str,
                 gst_status: str, quote_date: str | None = None,
                 valid_until: str | None = None, stated_total: float | None = None,
                 stated_total_line_no: int | None = None) -> int:
    """Insert or update one supplier quote and return its quote_id."""
    # ON CONFLICT ... DO UPDATE keeps the existing row (and its quote_id).
    # Do NOT use INSERT OR REPLACE: it deletes the old row and inserts a new one with a
    # new id, which orphans every line that pointed at the old id.
    conn.execute(
        """
        INSERT INTO quotes (supplier, source_file, gst_status, quote_date, valid_until,
                            stated_total, stated_total_line_no)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
            supplier = excluded.supplier,
            gst_status = excluded.gst_status,
            quote_date = excluded.quote_date,
            valid_until = excluded.valid_until,
            stated_total = excluded.stated_total,
            stated_total_line_no = excluded.stated_total_line_no
        """,
        (supplier, source_file, gst_status, quote_date, valid_until,
         stated_total, stated_total_line_no),
    )
    # A separate SELECT is the most portable way to get the id: after an update-on-conflict
    # cursor.lastrowid is not reliable.
    return conn.execute("SELECT quote_id FROM quotes WHERE source_file = ?",
                        (source_file,)).fetchone()["quote_id"]


def upsert_raw_line(conn: sqlite3.Connection, quote_id: int, line_no: int, raw_text: str,
                    raw_qty: float | None = None, raw_unit: str | None = None,
                    raw_unit_price: float | None = None,
                    raw_line_total: float | None = None,
                    raw_note: str | None = None) -> int:
    """Insert or update one raw line, keyed by (quote_id, line_no). Returns line_id."""
    # If the AI reads the same line slightly differently on a re-run, the row is updated
    # in place instead of duplicated, because the key is the line's POSITION in the file,
    # not the text the AI produced.
    conn.execute(
        """
        INSERT INTO quote_lines_raw (quote_id, line_no, raw_text, raw_qty, raw_unit,
                                     raw_unit_price, raw_line_total, raw_note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(quote_id, line_no) DO UPDATE SET
            raw_text = excluded.raw_text,
            raw_qty = excluded.raw_qty,
            raw_unit = excluded.raw_unit,
            raw_unit_price = excluded.raw_unit_price,
            raw_line_total = excluded.raw_line_total,
            raw_note = excluded.raw_note
        """,
        (quote_id, line_no, raw_text, raw_qty, raw_unit, raw_unit_price,
         raw_line_total, raw_note),
    )
    return conn.execute(
        "SELECT line_id FROM quote_lines_raw WHERE quote_id = ? AND line_no = ?",
        (quote_id, line_no),
    ).fetchone()["line_id"]


# flags is a tuple in normalise.NormalisedLine but SQLite has no array column -- join it
# into one TEXT column to store, and check membership with a plain substring search to read
# it back (is_total_only below). No flag string is ever a substring of another, so this
# never produces a false positive; a real array column (JSON or a separate table) would be
# the production answer, but is overkill for 5 known flag strings in a demo.
FLAG_SEPARATOR = "|"


def upsert_norm_line(conn: sqlite3.Connection, line_id: int, item_code: str | None,
                     match_confidence: float | None, qty_canonical: float | None,
                     unit_price_ex_gst: float | None, line_total_ex_gst: float | None,
                     flags: tuple[str, ...] = ()) -> None:
    """Insert or update one line's normalised numbers. 1:1 with a raw line, so line_id
    (already unique) is reused as the primary key instead of inventing a new one."""
    # OUT_OF_RFQ is a legal match (schemas.py's Literal allows it) but it is not a row in
    # `items` -- it means "this charge is real but isn't one of the 20 RFQ items", so there
    # is no catalog entry to look up. Storing the literal string would violate the item_code
    # FK, so it is translated to NULL here (this module's own schema comment already says
    # NULL means "not in the RFQ" -- that was always the plan, Phase 4's AI schema just used
    # a different spelling for it) and folded into flags instead, so load_quotes() can still
    # recover it.
    is_out_of_rfq = item_code == OUT_OF_RFQ
    db_item_code = None if is_out_of_rfq else item_code
    db_flags = (OUT_OF_RFQ, *flags) if is_out_of_rfq else flags
    conn.execute(
        """
        INSERT INTO quote_lines_norm (line_id, item_code, match_confidence, qty_canonical,
                                      unit_price_ex_gst, line_total_ex_gst, flags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(line_id) DO UPDATE SET
            item_code = excluded.item_code,
            match_confidence = excluded.match_confidence,
            qty_canonical = excluded.qty_canonical,
            unit_price_ex_gst = excluded.unit_price_ex_gst,
            line_total_ex_gst = excluded.line_total_ex_gst,
            flags = excluded.flags
        """,
        (line_id, db_item_code, match_confidence, qty_canonical, unit_price_ex_gst,
         line_total_ex_gst, FLAG_SEPARATOR.join(db_flags)),
    )


def set_bundle_items(conn: sqlite3.Connection, line_id: int, item_codes: list[str]) -> None:
    """Replace the set of items one line's bundle covers. Delete-then-insert rather than an
    upsert: on a re-run the AI may list a different set, and a code it dropped must go too."""
    conn.execute("DELETE FROM bundle_items WHERE line_id = ?", (line_id,))
    conn.executemany("INSERT INTO bundle_items (line_id, item_code) VALUES (?, ?)",
                     [(line_id, code) for code in sorted(set(item_codes))])


@dataclass(frozen=True)
class ItemRow:
    """One row of the `items` table, read back as Decimal -- the shape pipeline.py needs to
    hand to rules.py (rfq_qtys/reference_prices) and export.py (the Item list), without
    either of those modules having to know SQL exists.
    """
    item_code: str
    description: str
    unit: str
    rfq_qty: Decimal
    ref_price_ex_gst: Decimal


def get_items(conn: sqlite3.Connection) -> list[ItemRow]:
    rows = conn.execute(
        "SELECT item_code, description, unit, rfq_qty, ref_price_ex_gst "
        "FROM items ORDER BY item_code"
    ).fetchall()
    return [ItemRow(r["item_code"], r["description"], r["unit"],
                    to_decimal(r["rfq_qty"]), to_decimal(r["ref_price_ex_gst"]))
            for r in rows]


def load_quotes(conn: sqlite3.Connection) -> list[SupplierQuote]:
    """Rebuild SupplierQuote/SupplierLine objects from what is actually stored in the
    database -- the "B" choice: rules.py/export.py/evaluate.py consume THIS, not whatever
    extract.py/normalise.py just computed in memory. A bug in the write path (wrong upsert
    key, a dropped column) then shows up here as wrong data, instead of never being
    exercised at all.
    """
    # LEFT JOIN, not JOIN: a raw line that failed normalisation (e.g. an unknown unit)
    # still needs to show up -- as a line with no normalised numbers, not be silently
    # dropped, which would make a real extraction failure look like "nothing was quoted".
    rows = conn.execute(
        """
        SELECT q.quote_id, q.supplier, q.gst_status, q.stated_total,
               r.line_id, r.raw_qty, r.raw_unit_price, r.raw_line_total, r.raw_note,
               n.item_code, n.qty_canonical, n.unit_price_ex_gst, n.line_total_ex_gst, n.flags
        FROM quotes q
        JOIN quote_lines_raw r ON r.quote_id = q.quote_id
        LEFT JOIN quote_lines_norm n ON n.line_id = r.line_id
        ORDER BY q.quote_id, r.line_no
        """
    ).fetchall()
    bundles: dict[int, list[str]] = {}
    for b in conn.execute("SELECT line_id, item_code FROM bundle_items ORDER BY item_code"):
        bundles.setdefault(b["line_id"], []).append(b["item_code"])

    # SQL can't return a tree of objects, only flat rows -- group consecutive rows by
    # quote_id back into one SupplierQuote per supplier, each holding its own line list.
    quotes: dict[int, SupplierQuote] = {}
    order: list[int] = []
    for row in rows:
        qid = row["quote_id"]
        if qid not in quotes:
            order.append(qid)
            quotes[qid] = SupplierQuote(
                supplier=row["supplier"], gst_status=row["gst_status"],
                stated_total=to_decimal(row["stated_total"]), lines=[])
        flags = row["flags"] or ""
        # The write side folded OUT_OF_RFQ into flags instead of the item_code column (FK
        # safety -- see upsert_norm_line) -- undo that here so every downstream consumer
        # (rules.py, export.py, evaluate.py) sees the same OUT_OF_RFQ string it always did.
        item_code = OUT_OF_RFQ if OUT_OF_RFQ in flags else row["item_code"]
        # Mutating .lines is fine even though SupplierQuote is frozen: frozen only blocks
        # reassigning the attribute itself (quote.lines = ...), not mutating the list object
        # it already points at.
        quotes[qid].lines.append(SupplierLine(
            item_code=item_code,
            raw_qty=to_decimal(row["raw_qty"]),
            raw_unit_price=to_decimal(row["raw_unit_price"]),
            raw_line_total=to_decimal(row["raw_line_total"]),
            qty_canonical=to_decimal(row["qty_canonical"]),
            unit_price_ex_gst=to_decimal(row["unit_price_ex_gst"]),
            line_total_ex_gst=to_decimal(row["line_total_ex_gst"]),
            note=row["raw_note"],
            is_total_only=FLAG_TOTAL_ONLY in flags,
            bundle_item_codes=tuple(bundles.get(row["line_id"], ())),
        ))
    return [quotes[qid] for qid in order]

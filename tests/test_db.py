"""Tests for src/db.py.

Every test uses a fresh in-memory SQLite connection (":memory:") -- no file on disk, no
cleanup needed, and fast enough that this file costs nothing to run on every save. This is
the first test file for db.py: Phases 1-7 only ever exercised it manually via `--init-db`.
"""
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from src import db
from src.schemas import OUT_OF_RFQ


@pytest.fixture
def conn():
    # db.connect() calls db_path.parent.mkdir(), so the path has to be a real Path object,
    # not a bare ":memory:" string -- Path(":memory:").parent is just "." and already exists,
    # and sqlite3 still recognises the special in-memory name once it's stringified back.
    connection = db.connect(Path(":memory:"))
    db.init_db(connection)
    yield connection
    connection.close()


def _seed_item(conn, item_code="FRM-9045-H12"):
    # quote_lines_norm.item_code REFERENCES items(item_code) -- any test writing a norm
    # line with a real code needs that code to already exist, or SQLite's FK check (turned
    # on in db.connect) rejects it. That rejection is the point of the foreign key: it is
    # the last-resort defence (see schemas.py) against a code that shouldn't be possible.
    conn.execute(
        "INSERT INTO items (item_code, description, unit, rfq_qty, ref_price_ex_gst) "
        "VALUES (?, 'Framing 90x45', 'lm', 1000, 5.2)", (item_code,)
    )


def test_get_items_reads_back_as_decimal(conn):
    conn.execute(
        "INSERT INTO items (item_code, description, unit, rfq_qty, ref_price_ex_gst) "
        "VALUES ('FRM-9045-H12', 'Framing 90x45', 'lm', 1000, 5.2)"
    )
    [item] = db.get_items(conn)
    assert item.item_code == "FRM-9045-H12"
    # REAL columns come back as Decimal, not float -- the whole point of get_items() existing
    # instead of callers reading conn.execute() rows directly.
    assert item.rfq_qty == Decimal("1000")
    assert item.ref_price_ex_gst == Decimal("5.2")


def test_upsert_quote_is_idempotent(conn):
    first = db.upsert_quote(conn, "A_Harbour", "a.xlsx", "excl", stated_total=1270.0)
    second = db.upsert_quote(conn, "A_Harbour", "a.xlsx", "excl", stated_total=1270.0)
    # Same source_file -> same row updated in place, not a second row with a new id.
    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] == 1


def test_upsert_raw_line_is_idempotent(conn):
    qid = db.upsert_quote(conn, "A_Harbour", "a.xlsx", "excl")
    first = db.upsert_raw_line(conn, qid, line_no=3, raw_text="90x45 framing")
    second = db.upsert_raw_line(conn, qid, line_no=3, raw_text="90x45 framing (re-read)")
    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM quote_lines_raw").fetchone()[0] == 1
    # The update actually happened, not just the key -- re-running with a different reading
    # of the same line should replace the text, not silently keep the stale version.
    row = conn.execute("SELECT raw_text FROM quote_lines_raw WHERE line_id = ?",
                       (first,)).fetchone()
    assert row["raw_text"] == "90x45 framing (re-read)"


def test_load_quotes_reconstructs_one_line(conn):
    _seed_item(conn)
    qid = db.upsert_quote(conn, "A_Harbour", "a.xlsx", "excl",
                          stated_total=1270.0, stated_total_line_no=9)
    lid = db.upsert_raw_line(conn, qid, line_no=3, raw_text="90x45 framing",
                             raw_qty=100, raw_unit="lm", raw_unit_price=12.5,
                             raw_line_total=1250.0)
    db.upsert_norm_line(conn, lid, item_code="FRM-9045-H12", match_confidence=0.95,
                        qty_canonical=100, unit_price_ex_gst=12.5, line_total_ex_gst=1250.0,
                        flags=("total only (unit price derived)",))

    [quote] = db.load_quotes(conn)
    assert quote.supplier == "A_Harbour"
    assert quote.stated_total == Decimal("1270")
    [line] = quote.lines
    assert line.item_code == "FRM-9045-H12"
    assert line.raw_line_total == Decimal("1250")       # the RAW number, untouched
    assert line.line_total_ex_gst == Decimal("1250")    # the NORMALISED number
    # Flags round-trip through one TEXT column via substring search, not a real array column.
    assert line.is_total_only is True


def test_load_quotes_keeps_a_line_with_no_norm_row(conn):
    """A raw line that extraction/normalisation never got to (e.g. it failed) must still
    appear -- as a line with no normalised numbers -- never be silently dropped, which
    would make a real failure look identical to "nothing was quoted here"."""
    qid = db.upsert_quote(conn, "A_Harbour", "a.xlsx", "excl")
    db.upsert_raw_line(conn, qid, line_no=3, raw_text="some unreadable row")

    [quote] = db.load_quotes(conn)
    [line] = quote.lines
    assert line.item_code is None
    assert line.qty_canonical is None
    assert line.is_total_only is False


def test_load_quotes_groups_lines_under_the_right_supplier(conn):
    """Two suppliers' lines must not bleed into each other after the grouping step --
    the exact bug a wrong GROUP BY or join condition would produce."""
    qid_a = db.upsert_quote(conn, "A_Harbour", "a.xlsx", "excl")
    qid_b = db.upsert_quote(conn, "B_Northshore", "b.pdf", "incl")
    db.upsert_raw_line(conn, qid_a, line_no=5, raw_text="second")
    db.upsert_raw_line(conn, qid_a, line_no=2, raw_text="first")
    db.upsert_raw_line(conn, qid_b, line_no=1, raw_text="only line")

    quotes = {q.supplier: q for q in db.load_quotes(conn)}
    assert len(quotes) == 2
    assert len(quotes["A_Harbour"].lines) == 2
    assert len(quotes["B_Northshore"].lines) == 1


def test_upsert_norm_line_handles_out_of_rfq_without_violating_the_fk(conn):
    """OUT_OF_RFQ is a legal match but is not a row in `items` -- writing it as the literal
    string used to trip the item_code FK (a real bug the first --run caught). It must round
    -trip through load_quotes() as OUT_OF_RFQ, not NULL and not crash.
    """
    qid = db.upsert_quote(conn, "B_Northshore", "b.pdf", "incl")
    lid = db.upsert_raw_line(conn, qid, line_no=7, raw_text="Delivery")
    db.upsert_norm_line(conn, lid, item_code=OUT_OF_RFQ, match_confidence=0.9,
                        qty_canonical=1, unit_price_ex_gst=160.87, line_total_ex_gst=160.87)

    [quote] = db.load_quotes(conn)
    assert quote.lines[0].item_code == OUT_OF_RFQ
    # The real column underneath should be NULL, matching this file's own schema comment
    # ("NULL means not in the RFQ") -- OUT_OF_RFQ only exists again after load_quotes decodes it.
    raw_item_code = conn.execute(
        "SELECT item_code FROM quote_lines_norm WHERE line_id = ?", (lid,)
    ).fetchone()["item_code"]
    assert raw_item_code is None


def test_upsert_norm_line_is_idempotent(conn):
    _seed_item(conn)
    qid = db.upsert_quote(conn, "A_Harbour", "a.xlsx", "excl")
    lid = db.upsert_raw_line(conn, qid, line_no=1, raw_text="x")
    db.upsert_norm_line(conn, lid, item_code="FRM-9045-H12", match_confidence=0.5,
                        qty_canonical=1, unit_price_ex_gst=1, line_total_ex_gst=1)
    db.upsert_norm_line(conn, lid, item_code="FRM-9045-H12", match_confidence=0.9,
                        qty_canonical=2, unit_price_ex_gst=2, line_total_ex_gst=2)
    assert conn.execute("SELECT COUNT(*) FROM quote_lines_norm").fetchone()[0] == 1
    [quote] = db.load_quotes(conn)
    assert quote.lines[0].qty_canonical == Decimal("2")   # second write won, not duplicated


def test_bundle_items_round_trip_and_a_rerun_replaces_the_set(conn):
    for code in ("NAIL-75-BR", "NAIL-75-HDG", "SEAL-123"):
        _seed_item(conn, code)
    qid = db.upsert_quote(conn, "C_KiwiFrame", "c.txt", "unstated")
    lid = db.upsert_raw_line(conn, qid, line_no=22, raw_text="Nails and Sikaflex, $980 all up")
    db.upsert_norm_line(conn, lid, item_code=OUT_OF_RFQ, match_confidence=0.6,
                        qty_canonical=None, unit_price_ex_gst=None, line_total_ex_gst=980)
    db.set_bundle_items(conn, lid, ["SEAL-123", "NAIL-75-BR", "NAIL-75-HDG"])
    db.set_bundle_items(conn, lid, ["NAIL-75-BR", "SEAL-123"])   # re-run lists fewer items

    [quote] = db.load_quotes(conn)
    assert quote.lines[0].bundle_item_codes == ("NAIL-75-BR", "SEAL-123")


def test_bundle_item_with_an_invented_code_is_rejected_by_the_fk(conn):
    qid = db.upsert_quote(conn, "C_KiwiFrame", "c.txt", "unstated")
    lid = db.upsert_raw_line(conn, qid, line_no=22, raw_text="package")
    with pytest.raises(sqlite3.IntegrityError):
        db.set_bundle_items(conn, lid, ["NOT-A-CODE"])

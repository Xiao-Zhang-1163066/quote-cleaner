"""Command-line entry point: `python -m src.pipeline`.

`--run` is the full pipeline in the order the spec describes it: read -> extract ->
normalise -> store -> detect issues -> export. This is the only module that calls every
other module -- it is the "imperative shell" the rest of the functional core sits inside.
"""
import argparse
import sqlite3

from pathlib import Path

from src import config, db, export, extract, normalise, readers, rules


def check_setup() -> bool:
    """Report what the pipeline can see. Returns True if all required inputs exist."""
    required = [config.RFQ_CSV, config.REFERENCE_PRICES_CSV, config.QUOTES_DIR]
    ok = True
    for path in required:
        found = path.exists()
        ok = ok and found
        print(f"  [{'ok' if found else 'MISSING'}] {path}")

    # Only report whether a key is present, never print it: logs get pasted into
    # chats and tickets, and a printed secret is a leaked secret.
    print(f"  [{'ok' if config.OPENAI_API_KEY else 'not set'}] OPENAI_API_KEY"
          " (needed from Phase 4)")
    print(f"  [{'ok' if config.MODEL_NAME else 'not set'}] QUOTE_MODEL"
          " (needed from Phase 4)")
    return ok


def ingest_quote_file(conn: sqlite3.Connection, path: Path,
                      items_by_code: dict[str, db.ItemRow]) -> extract.ExtractionResult:
    """Read one file, extract it with AI, normalise every line, and write all of it to the
    database. Returns the raw ExtractionResult so the caller can report tokens/time/errors --
    this function's job is the side effects, not deciding what to print about them.
    """
    source_lines = readers.read_quote(path)
    result = extract.extract_quote(source_lines)
    if result.quote is None:
        return result   # caller reports the failure; nothing to store for this file

    quote = result.quote
    # One transaction per file: either the whole quote (header + every line) lands, or
    # none of it does. A half-written quote (header saved, lines not) would be worse than
    # not having tried -- the file would look "done" on a re-run check but have no lines.
    with conn:
        quote_id = db.upsert_quote(conn, quote.supplier, path.name, quote.gst_status,
                                   quote.quote_date, quote.valid_until,
                                   quote.stated_total, quote.stated_total_line_no)
        for line in quote.lines:
            raw_text = next(s.text for s in source_lines if s.line_no == line.line_no)
            line_id = db.upsert_raw_line(
                conn, quote_id, line.line_no, raw_text=raw_text,
                raw_qty=line.qty, raw_unit=line.unit, raw_unit_price=line.unit_price,
                raw_line_total=line.line_total, raw_note=line.note)

            # OUT_OF_RFQ has no RFQ catalog entry, so there is no real "item_unit" to convert
            # to -- "" is a placeholder that can never equal a real unit, so every conversion
            # branch in canonical_units_per_raw_unit() falls through to "no conversion needed"
            # for a plain count ("EA"), which is the only unit these lines use in this data.
            item = items_by_code.get(line.item_code)
            item_unit = item.unit if item else ""
            # BUG FOUND BY RUNNING THIS FOR REAL: length_from_description() has to search the
            # SUPPLIER'S OWN line text ("FRAMING 90X45 SG8 H1.2 KD 6.0M" -- the stick length
            # is only written here), not the RFQ catalog's generic description ("90x45 SG8
            # H1.2 KD radiata framing" -- never has a length). Passing the catalog text made
            # every per-metre timber line silently fail unit conversion.
            try:
                normalised = normalise.normalise_line(
                    raw_qty=line.qty, raw_unit=line.unit, raw_unit_price=line.unit_price,
                    raw_line_total=line.line_total, gst_status=quote.gst_status,
                    item_unit=item_unit, description=raw_text)
            except normalise.UnitError as exc:
                # No usable unit at all (e.g. C's lump-sum "fixings package $980" -- no
                # qty, no unit, just a dollar figure). There is no per-unit price to compute,
                # but the total still needs its GST stripped so rules.is_lump_sum (which
                # requires a non-None line_total_ex_gst) and the Issues sheet have a number.
                total = normalise.to_decimal(line.line_total)
                normalised = normalise.NormalisedLine(
                    qty=None, unit_price_ex_gst=None,
                    line_total_ex_gst=(normalise.money(normalise.strip_gst(total, quote.gst_status))
                                       if total is not None else None),
                    flags=(f"unit_error:{exc}",))

            db.upsert_norm_line(
                conn, line_id, line.item_code, line.match_confidence,
                qty_canonical=float(normalised.qty) if normalised.qty is not None else None,
                unit_price_ex_gst=(float(normalised.unit_price_ex_gst)
                                   if normalised.unit_price_ex_gst is not None else None),
                line_total_ex_gst=(float(normalised.line_total_ex_gst)
                                   if normalised.line_total_ex_gst is not None else None),
                flags=normalised.flags)
    return result


def run_pipeline(conn: sqlite3.Connection, quotes_dir: Path = config.QUOTES_DIR) -> None:
    """Ingest every supported file in quotes_dir, printing a line of stats (or the error)
    per file. Unsupported files (D's .png -- Phase 9's job) are skipped, not fatal: readers.py
    raises ValueError for a type it has no reader for, and that is exactly the signal to skip.
    """
    items_by_code = {item.item_code: item for item in db.get_items(conn)}
    for path in sorted(quotes_dir.iterdir()):
        if path.suffix.lower() not in readers.READERS:
            print(f"  skip {path.name}: no reader for {path.suffix!r} yet")
            continue

        result = ingest_quote_file(conn, path, items_by_code)
        print(f"  {path.name}: model={result.model} attempts={result.attempts} "
              f"tokens={result.input_tokens}in/{result.output_tokens}out "
              f"time={result.elapsed_seconds:.1f}s")
        if result.quote is None:
            print(f"    FAILED: {result.error} -- needs human review, skipped")


def export_report(conn: sqlite3.Connection,
                  output_path: Path = config.OUTPUT_DIR / "comparison.xlsx") -> list[rules.Issue]:
    """Read everything back out of the database (never trust the in-memory copy -- see
    Phase 8's design note in DEV_LOG.md) and produce the Excel + the issue list."""
    item_rows = db.get_items(conn)
    items = [export.Item(i.item_code, i.description, i.unit) for i in item_rows]
    rfq_qtys = {i.item_code: i.rfq_qty for i in item_rows}
    reference_prices = {i.item_code: i.ref_price_ex_gst for i in item_rows}

    quotes = db.load_quotes(conn)
    issues = rules.detect_issues(quotes, rfq_qtys, reference_prices)
    export.export_workbook(items, quotes, issues, output_path)
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Turn messy supplier quotes into one comparable, flagged table.",
    )
    # argparse gives us --help for free; the flags we add later (e.g. --confirm) will
    # show up in it automatically, which is our built-in usage documentation.
    parser.add_argument("--check", action="store_true",
                        help="verify input files and settings, then exit")
    parser.add_argument("--init-db", action="store_true",
                        help="create the database tables and load the RFQ items")
    parser.add_argument("--read", metavar="FILE",
                        help="print the lines the reader extracts from one quote file")
    parser.add_argument("--extract", metavar="FILE",
                        help="read one quote file and run AI extraction + validation on it")
    parser.add_argument("--run", action="store_true",
                        help="run every stage end-to-end on every file in data/quotes/ "
                             "and write output/comparison.xlsx")
    args = parser.parse_args()

    if args.read:
        for line in readers.read_quote(Path(args.read)):
            print(f"{line.line_no:>3}: {line.text}")
        return 0

    if args.extract:
        source_lines = readers.read_quote(Path(args.extract))
        result = extract.extract_quote(source_lines)
        print(f"model={result.model} attempts={result.attempts} "
              f"tokens={result.input_tokens}in/{result.output_tokens}out "
              f"time={result.elapsed_seconds:.1f}s")
        if result.quote is None:
            print(f"FAILED: {result.error}")
            return 1
        print(f"supplier={result.quote.supplier!r} gst={result.quote.gst_status} "
              f"lines={len(result.quote.lines)} "
              f"stated_total={result.quote.stated_total} (line {result.quote.stated_total_line_no})")
        for line in result.quote.lines:
            print(f"  {line.line_no:>3}: {line.item_code:<14} conf={line.match_confidence:.2f} "
                  f"qty={line.qty} unit={line.unit} price={line.unit_price} "
                  f"total={line.line_total} note={line.note}")
        return 0

    if args.run:
        conn = db.connect()
        with conn:
            db.init_db(conn)
            db.load_items(conn)
        print("Ingesting:")
        run_pipeline(conn)
        print("Exporting:")
        issues = export_report(conn)
        by_severity = {s: sum(1 for i in issues if i.severity == s) for s in rules.Severity}
        print(f"  {config.OUTPUT_DIR / 'comparison.xlsx'} "
              f"({by_severity[rules.Severity.ERROR]} error, "
              f"{by_severity[rules.Severity.WARNING]} warning, "
              f"{by_severity[rules.Severity.INFO]} info)")
        return 0

    if args.init_db:
        conn = db.connect()
        # `with conn` = one transaction: schema and items are both saved, or neither is.
        with conn:
            db.init_db(conn)
            count = db.load_items(conn)
        print(f"Database ready at {config.DB_PATH} ({count} RFQ items loaded)")
        return 0

    if args.check:
        print("Setup check:")
        # Exit code 1 on failure so scripts and CI can detect a bad setup.
        return 0 if check_setup() else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

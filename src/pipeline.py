"""Command-line entry point: `python -m src.pipeline`.

For now it only checks that the project is wired up correctly (paths exist, API key
status). Each later phase plugs its stage into main() in the order of the pipeline:
read -> extract -> normalise -> store + detect issues -> export.
"""
import argparse

from pathlib import Path

from src import config, db, extract, readers


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

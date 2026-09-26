"""Command-line entry point: `python -m src.pipeline`.

For now it only checks that the project is wired up correctly (paths exist, API key
status). Each later phase plugs its stage into main() in the order of the pipeline:
read -> extract -> normalise -> store + detect issues -> export.
"""
import argparse

from src import config, db


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
    print(f"  [{'ok' if config.ANTHROPIC_API_KEY else 'not set'}] ANTHROPIC_API_KEY"
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
    args = parser.parse_args()

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

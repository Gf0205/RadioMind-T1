from .load import MODULATIONS, SNRS, DatasetStore, load_store
from .split import build_splits, load_or_create_splits

__all__ = ["MODULATIONS", "SNRS", "DatasetStore", "load_store", "build_splits", "load_or_create_splits"]

def main() -> None:
    import argparse
    from .check import run_check
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.check:
        run_check(args.data_root)
    else:
        parser.error("use --check")

if __name__ == "__main__":
    main()

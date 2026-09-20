"""P8: main table (spec 13.1) with S, prediction check and verdicts; appendix; optional action table (spec 14.5).

  python scripts/p8_tables.py [--models cjepa lpwm] [--split test] [--actions]
"""
import argparse

from owm.analysis.tables import action_table, main_table
from owm.config import output_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["cjepa", "lpwm"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--actions", action="store_true")
    a = ap.parse_args()
    main_table(tuple(a.models), a.split)
    print((output_dir() / "tables" / "main_table.md").read_text())
    print((output_dir() / "tables" / "appendix.md").read_text())
    if a.actions:
        print(action_table(tuple(a.models), a.split))

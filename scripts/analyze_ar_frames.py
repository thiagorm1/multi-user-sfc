"""Summarize measured AR motion/display timestamps without simulating comfort."""

import argparse
import json
from pathlib import Path

from muar_sfc.core.ar_metrics import ar_frame_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="JSON list of motion_s/display_s or dropped frames"
    )
    parser.add_argument("--budget-ms", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = ar_frame_summary(json.loads(args.input.read_text()), args.budget_ms)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

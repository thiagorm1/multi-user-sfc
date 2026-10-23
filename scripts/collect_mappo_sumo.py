"""Collect policy-independent real SUMO trajectories for mobile MAPPO training."""

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from muar_sfc.core.scenarios.muvr_reference import MUVRProfile
from muar_sfc.core.scenarios.sumo_replay import (
    DATASET_SCHEMA,
    collect_trace,
    document_sha256,
    validate_seed_splits,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumo-net", type=Path, required=True)
    parser.add_argument("--sumo-binary", default=".venv/bin/sumo")
    parser.add_argument("--train-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--validation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--test-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sla-ms", type=float, default=6.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, default=Path("results/sumo_training_traces"))
    parser.add_argument("--check", action="store_true", help="Validate only; no SUMO or writes.")
    args = parser.parse_args(argv)
    splits = validate_seed_splits(
        {"train": args.train_seeds, "validation": args.validation_seeds, "test": args.test_seeds}
    )
    profile = MUVRProfile(requests=1, players=4, sla=args.sla_ms)
    if not args.sumo_net.is_file() or not shutil.which(args.sumo_binary):
        parser.error("An existing road network and Eclipse SUMO executable are required")
    if args.step_s <= 0 or not (profile.arrival_interval / args.step_s).is_integer():
        parser.error("The positive trace step must divide the arrival interval")
    specification = {
        "schema": DATASET_SCHEMA,
        "profile": asdict(profile),
        "seed_splits": splits,
        "step_s": args.step_s,
        "sumo_net": str(args.sumo_net.resolve()),
        "sumo_binary": args.sumo_binary,
    }
    if args.check:
        print(json.dumps({"check": "passed", **specification}))
        return
    directory = args.output_root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**specification, "complete": False, "trace_hashes": {}}
    for split, seeds in splits.items():
        for seed in seeds:
            trace = collect_trace(profile, seed, args.sumo_net, args.sumo_binary, args.step_s)
            with (directory / f"seed_{seed}.json").open("x") as output:
                json.dump(trace, output, indent=2, allow_nan=False)
            manifest["trace_hashes"][str(seed)] = trace["trace_sha256"]
            print(json.dumps({"split": split, "seed": seed, "trace": str(directory)}), flush=True)
    manifest["complete"] = True
    manifest["manifest_sha256"] = document_sha256(manifest)
    with (directory / "manifest.json").open("x") as output:
        json.dump(manifest, output, indent=2, allow_nan=False)
    print(f"Dataset: {directory}")


if __name__ == "__main__":
    main()

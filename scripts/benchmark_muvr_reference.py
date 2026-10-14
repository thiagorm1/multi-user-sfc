"""Run the explicit 25-edge-server MUVR profile in virtual time."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 75)))
    parser.add_argument(
        "--destinations", choices=["same_access", "scattered"], default="same_access"
    )
    parser.add_argument("--strategy", choices=["joint", "anchor"], default="joint")
    parser.add_argument(
        "--output", type=Path, default=Path("results/muvr_50_reference/benchmark.json")
    )
    args = parser.parse_args()
    profile = MUVRProfile(requests=args.requests, destination_mode=args.destinations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        row = simulate(profile, seed, args.strategy)
        rows.append(row)
        document = {
            "profile": asdict(profile),
            "rows": rows,
            "assumptions": [
                "25 edge servers as requested; thesis Table 9 says 25 total nodes and 30% edge.",
                "Original nominal CPU/cache 100 units per server; IPS uniform(0.1,0.2)*1e11.",
                "1 Gbps links; no added edges, graph resampling or capacity multipliers.",
                "One request is one complete four-player session, atomic immediate admission.",
                "Entire session consumes 20 CPU / 15 memory, as requested (not per player).",
                "SF CPU/memory proportional to input traffic then normalized per session.",
                "Predicted frames still consume processing resources; no free pre-rendering.",
                "Explicit destinations: one shared regional access, or scattered users.",
                "Link delays plus processing with the existing 60fps/10cycles-bit model.",
                "No unprovided additional access delay or queueing delay is invented.",
                "Joint heuristic and MAPPO feasibility fallback; no trained policy or Qwen.",
                "Not an exact reproduction or verified superiority over the published algorithm.",
            ],
            "source": "https://webarchive.inf.unibe.ch/cds/publications/files/Hugo_Santos.pdf",
        }
        args.output.write_text(json.dumps(document, indent=2, allow_nan=False))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("topology", "records", "utilization_samples")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

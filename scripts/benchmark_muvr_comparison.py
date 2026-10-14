"""Paired comparison with explicit adapters and common physical validation."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from muar_sfc.algorithms.muvr_baselines import MUVRBaseline
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument(
        "--models", nargs="+", default=["joint", "replic", "musfico", "greedy", "darsppo"]
    )
    parser.add_argument("--output", type=Path, default=Path("results/muvr_comparison"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    profile = MUVRProfile(requests=args.requests)
    for model in args.models:
        adapter = None if model == "joint" else MUVRBaseline(model)
        document = {
            "profile": asdict(profile),
            "model": model,
            "rows": [],
            "method": "Paired immediate session admission, shared physical resources/SLA audit.",
            "adaptation": "Joint is the current model. REPLIC/DARSPPO use frozen legacy "
            "MaskablePPO checkpoints and original observation functions, reverse full-DAG "
            "traversal, shared instance coordination and physical DAG validation. "
            "Greedy uses existing DAGGreedyAlgorithm plus physical validation. "
            "MuSFiCO is a generalized bounded DP DAG adapter (beam width 4), "
            "not the unmodified implementation or published algorithm.",
            "checkpoint_sha256": hashlib.sha256(adapter.model_path.read_bytes()).hexdigest()
            if adapter is not None and adapter.model_path
            else None,
        }
        for seed in args.seeds:
            row = simulate(profile, seed, model, admitter=adapter)
            document["rows"].append(row)
            (args.output / f"{model}.json").write_text(
                json.dumps(document, indent=2, allow_nan=False)
            )
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in [
                            "seed",
                            "strategy",
                            "accepted",
                            "acceptance_percent",
                            "mean_latency_ms",
                            "mean_decision_s",
                        ]
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

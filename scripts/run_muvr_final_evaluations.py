"""Explicitly authorized D1/D2/D3 live evaluations in exclusive directories.

This command calls the existing local Qwen; it never trains or downloads models.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


def write_new(path, document):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=["D1", "D2", "D3"],
                        default=["D1", "D2", "D3"])
    args = parser.parse_args()
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "environment.json", {
        "platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(), "hip": torch.version.hip,
        "checkpoint": str(args.model), "checkpoint_sha256": model_hash,
        "qwen_mode": "live_async", "seeds": [42, 43, 44], "requests": 50,
        "threads": 4, "load_balance_scale": 0.0,
    })
    for label, attempts, extra in [
        ("D1", 16, []),
        ("D2_exploratory_toporag", 16, [
            "--toporag-enabled", "--toporag-kappa", "0.2", "--toporag-top-k", "4",
            "--toporag-prior-scale", "1.0", "--toporag-confidence-mode", "fixed",
        ]),
        ("D3_exploratory_attempts32", 32, []),
    ]:
        if label.split("_")[0] not in args.conditions:
            continue
        output = args.output / label
        output.mkdir(exist_ok=False)
        command = [
            sys.executable, "-u", "scripts/benchmark_mappo_qwen_muvr.py",
            "--requests", "50", "--seeds", "42", "43", "44",
            "--variants", "mappo_qwen", "--model", str(args.model),
            "--threads", "4", "--search-attempts", str(attempts),
            "--load-balance-scale", "0.0", "--qwen-model", "qwen2.5:1.5b",
            "--api-base-url", "http://localhost:11434/v1", "--output", str(output),
            *extra,
        ]
        start = time.perf_counter()
        started_at = datetime.now().astimezone().isoformat()
        with (output / "execution.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        write_new(output / "execution.json", {
            "command": command, "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "wall_seconds": time.perf_counter() - start, "exit_code": result.returncode,
            "checkpoint_sha256": model_hash,
        })
        if result.returncode:
            raise RuntimeError(f"{label} failed; preserve logs at {output}")
        document = json.loads((output / "mappo_qwen.json").read_text(encoding="utf-8"))
        (output / "seeds").mkdir()
        for row in document["rows"]:
            write_new(output / "seeds" / f"seed_{row['seed']}.json", row)
        print(label, [(r["seed"], r["accepted"], r["mean_decision_s"])
                      for r in document["rows"]], flush=True)


if __name__ == "__main__":
    main()

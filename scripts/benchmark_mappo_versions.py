"""Compare MAPPO versions on one host, with paired, counterbalanced CPU runs."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path

import torch
from loguru import logger

from muar_sfc.algorithms.mappo.muvr import TrainedMUVRMAPPO
from muar_sfc.config import SimulationSettings
from muar_sfc.core.ar_metrics import distribution
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--old-model", required=True, type=Path)
    parser.add_argument("--new-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory to preserve previous measurements")
    if min(args.requests, args.repeats, args.threads) < 1:
        parser.error("Requests, repeats and threads must be positive")
    revision = subprocess.check_output(["git", "rev-parse", args.baseline_ref], text=True).strip()
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/muar_sfc/algorithms/mappo/muvr.py"], text=True
    )
    namespace = {}
    exec(compile(source, f"muvr_at_{revision}.py", "exec"), namespace)
    old_class = namespace["TrainedMUVRMAPPO"]
    variants = {
        "original_8": (old_class, args.old_model, 8),
        "current_same_model_8": (TrainedMUVRMAPPO, args.old_model, 8),
        "current_same_model_16": (TrainedMUVRMAPPO, args.old_model, 16),
        "current_new_model_16": (TrainedMUVRMAPPO, args.new_model, 16),
    }
    for _, model, _ in variants.values():
        if not model.is_file():
            parser.error(f"Checkpoint missing: {model}")
    torch.set_num_threads(args.threads)
    logger.remove()
    args.output.mkdir(parents=True)
    (args.output / "baseline_muvr.py").write_text(source, encoding="utf-8")
    manifest = {
        "baseline_revision": revision,
        "requests_per_seed": args.requests,
        "seeds": [42, 43, 44],
        "repeats": args.repeats,
        "threads": args.threads,
        "qwen": False,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "current_source_sha256": hashlib.sha256(
            Path("src/muar_sfc/algorithms/mappo/muvr.py").read_bytes()
        ).hexdigest(),
        "models": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.old_model, args.new_model)
        },
        "method": "Serial CPU runs; variant order reversed on alternate repeats; "
        "one unmeasured warmup per variant; no training, downloads or Qwen calls.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def run(name, seed, requests):
        cls, model, attempts = variants[name]
        settings = SimulationSettings(
            _env_file=None,
            seed=seed,
            n_mappo_regions=4,
            mappo_model_path=str(model),
            mappo_search_attempts=attempts,
            cognitive_planner=False,
        )
        adapter = cls(settings, qwen=False)
        try:
            row = simulate(MUVRProfile(requests=requests), seed, name, admitter=adapter)
        finally:
            adapter.close()
        row["model_sha256"] = adapter.model_sha256
        return row

    for name in variants:
        run(name, 100042, 1)
    print(json.dumps({"stage": "warmup_complete"}), flush=True)
    measured = []
    for repeat in range(args.repeats):
        order = list(variants)
        if repeat % 2:
            order.reverse()
        for seed in manifest["seeds"]:
            for name in order:
                row = run(name, seed, args.requests)
                row.update(variant=name, repeat=repeat + 1)
                measured.append(row)
                (args.output / "measurements.json").write_text(json.dumps(measured, indent=2))
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in ("variant", "repeat", "seed", "accepted", "mean_decision_s")
                        }
                    ),
                    flush=True,
                )
    for seed in manifest["seeds"]:
        paired = [r for r in measured if r["seed"] == seed]
        for key in ("workload_sha256", "topology_sha256"):
            assert len({r[key] for r in paired}) == 1, (seed, key)
        for name in variants:
            runs = [r for r in paired if r["variant"] == name]
            assert len({tuple(x["accepted_immediate"] for x in r["records"]) for r in runs}) == 1
    summary = {}
    for name in variants:
        rows = [r for r in measured if r["variant"] == name]
        records = [r for row in rows for r in row["records"]]
        assert all(lat <= 6 for r in records for lat in r["latencies_ms"])
        summary[name] = {
            "accepted_unique": sum(r["accepted"] for r in rows if r["repeat"] == 1),
            "unique_sessions": args.requests * len(manifest["seeds"]),
            "decision_ms": distribution([r["decision_s"] * 1000 for r in records]),
            "run_means_ms": [r["mean_decision_s"] * 1000 for r in rows],
            "max_latency_ms": max(r["max_latency_ms"] for r in rows),
        }
    reference = summary["original_8"]["decision_ms"]["mean"]
    lines = [
        "# MAPPO: comparação de versões na mesma máquina",
        "",
        manifest["method"],
        "",
        f"CPU: {manifest['processor']}; PyTorch {manifest['torch']}; "
        f"{args.threads} threads. Sementes 42–44, {args.requests} sessões por semente, "
        f"{args.repeats} repetições. Qwen desativado em todas as variantes.",
        "",
        "| Variante | Aceitas únicas | Média (ms) | p95 (ms) | Variação vs. original |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in summary.items():
        d = s["decision_ms"]
        s["difference_vs_original_percent"] = (d["mean"] / reference - 1) * 100
        lines.append(
            f"| {name} | {s['accepted_unique']}/{s['unique_sessions']} "
            f"| {d['mean']:.2f} | {d['p95']:.2f} "
            f"| {s['difference_vs_original_percent']:+.2f}% |"
        )
    lines += [
        "",
        "Original_8 usa a implementação do commit " + revision + " e o checkpoint "
        "recriado com o treinamento anterior (não é o checkpoint histórico de 347 ms). "
        "Current_same_model_8 isola a implementação; "
        "current_same_model_16 acrescenta o orçamento de busca; "
        "current_new_model_16 acrescenta o novo checkpoint.",
        "",
        "As repetições medem variação de tempo, não acrescentam sessões independentes. "
        "As contagens de aceitação únicas usam apenas a primeira repetição. "
        "O SLA de 6 ms foi verificado para todas as cadeias aceitas.",
        "",
        "O valor histórico de 347 ms veio de outra máquina e não é uma referência "
        "controlada para estimar a regressão desta implementação.",
        "",
        f"Média das execuções da referência: "
        f"{statistics.mean(summary['original_8']['run_means_ms']):.2f} ms.",
    ]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verified": True, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Paired ranking tradeoff with recorded Qwen guidance, not fresh LLM calls."""

import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from muar_sfc.algorithms.mappo.muvr import TrainedMUVRMAPPO
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.cloud_llm_planner import CognitiveGuidance
from muar_sfc.core.ar_metrics import distribution
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, simulate


class ReplayedGuidance(TrainedMUVRMAPPO):
    def __init__(self, settings, source):
        super().__init__(settings, qwen=False)
        self.source = source
        self.index = 0
        self.node_samples = []
        self.peak_samples = []
        self.imbalance_samples = []

    def prepare(self, net, batch):
        evidence = self.source["inference_evidence"]
        request = evidence["request_evidence"][self.index]
        guidance = None
        if request["guidance_used"]:
            for record in evidence["planning_records"][: request["guidance_cycle"]]:
                if record.get("qwen_success"):
                    guidance = record["guidance"]
            if guidance is None:
                raise ValueError("Source says guidance was used but has no successful guidance")
        self.guidance = CognitiveGuidance.model_validate(guidance) if guidance else None
        self.guidance_ready_at = None
        self.index += 1

    def __call__(self, net, batch, strategy):
        result = super().__call__(net, batch, strategy)
        graph = (result[0] if result[0] is not None else net).graph
        utilization = [100 * d["cpu_used"] / d["cpu_capacity"] for _, d in graph.nodes(data=True)]
        self.node_samples.extend(utilization)
        self.peak_samples.append(max(utilization))
        self.imbalance_samples.append(statistics.pstdev(utilization))
        return result


def create_run_directory(base, stamp=None):
    """Atomically reserve a new run directory, including concurrent launches."""
    base = Path(base)
    stamp = stamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    base.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        suffix = f"_{index:02d}" if index else ""
        candidate = base.with_name(f"{base.name}_{stamp}{suffix}")
        try:
            candidate.mkdir()
        except FileExistsError:
            index += 1
        else:
            return candidate


def write_json_new(path, value):
    """Never replace a result, even if another process created the file first."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def scale_tag(scale):
    return str(scale).removesuffix(".0")


def limit_source_requests(source, requests):
    """Select a verified workload prefix without modifying the recorded run."""
    if not 1 <= requests <= source["requests"]:
        raise ValueError("Requests must be positive and cannot exceed the source run")
    if requests == source["requests"]:
        return source
    records = source["records"][:requests]
    evidence = source["inference_evidence"]
    if len(records) != requests or len(evidence["request_evidence"]) < requests:
        raise ValueError("Source workload or guidance trace is incomplete")
    trace = [
        {
            key: record[key]
            for key in (
                "arrival_s",
                "h",
                "f",
                "source",
                "destinations",
                "lifetime_s",
                "distinct_sfs",
            )
        }
        for record in records
    ]
    return {
        **source,
        "requests": requests,
        "accepted": sum(r["accepted_immediate"] for r in records),
        "records": records,
        "workload_sha256": hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
        "inference_evidence": {
            **evidence,
            "request_evidence": evidence["request_evidence"][:requests],
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guidance-source", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("rl_saved_models/MAPPO_muvr_meu_treino.pt"))
    p.add_argument("--scales", type=float, nargs="+", default=[2.0, 1.8, 1.5, 1.0])
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Prefixo da pasta; cada execução recebe data/hora e nunca sobrescreve resultados",
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--requests", type=int, help="Requisições por semente; usa um prefixo da fonte")
    args = p.parse_args()
    if args.threads < 1:
        p.error("Threads must be positive")
    if len(set(args.scales)) != len(args.scales):
        p.error("Scales must not be repeated")
    if any(not 0 <= v <= 2 for v in args.scales):
        p.error("This experiment supports scales from zero to the original 2.0")
    doc = json.loads(args.guidance_source.read_text())
    sources = doc["rows"]
    expected_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if not all(r["inference_evidence"]["model_sha256"] == expected_hash for r in sources):
        p.error("Guidance and evaluation must use the same checkpoint")
    if len({r["seed"] for r in sources}) != len(sources):
        p.error("Source seeds must not be repeated")
    source_requests = sources[0]["requests"]
    if any(r["requests"] != source_requests for r in sources):
        p.error("Source runs must have equal request counts")
    requests = args.requests if args.requests is not None else source_requests
    try:
        sources = [limit_source_requests(row, requests) for row in sources]
    except (ValueError, KeyError) as exc:
        p.error(str(exc))
    requested_output = args.output
    args.output = create_run_directory(requested_output)
    print(f"Resultados desta execução: {args.output.resolve()}", flush=True)
    print(
        f"{len(sources)} sementes × {requests} requisições × {len(args.scales)} pesos "
        f"= {len(sources) * requests * len(args.scales)} requisições no total",
        flush=True,
    )
    torch.set_num_threads(args.threads)
    logger.remove()
    manifest = {
        "requested_output_prefix": str(requested_output),
        "output_directory": str(args.output.resolve()),
        "output_policy": "unique_timestamped_directory; exclusive_file_creation; per_seed_results",
        "guidance_source": str(args.guidance_source),
        "guidance_source_sha256": hashlib.sha256(args.guidance_source.read_bytes()).hexdigest(),
        "model_sha256": expected_hash,
        "requests": requests,
        "source_requests_per_seed": source_requests,
        "workload_selection": "same recorded prefix per seed",
        "seeds": [r["seed"] for r in sources],
        "scales": args.scales,
        "method": "Recorded real Qwen guidance replayed per request for every scale. "
        "No new LLM calls, no training, same nominal physical profile, SLA and search budget. "
        "CPU utilization below refers to simulated edge servers, not host process CPU.",
    }
    write_json_new(args.output / "manifest.json", manifest)
    summaries = []
    for scale in args.scales:
        seed_output = args.output / "seeds" / f"scale_{scale_tag(scale)}"
        seed_output.mkdir(parents=True)
        rows = []
        nodes = []
        peaks = []
        imbalances = []
        for source in sources:
            settings = SimulationSettings(
                _env_file=None,
                seed=source["seed"],
                mappo_model_path=str(args.model),
                mappo_search_attempts=16,
                mappo_load_balance_scale=scale,
            )
            adapter = ReplayedGuidance(settings, source)
            try:
                row = simulate(
                    MUVRProfile(requests=requests),
                    source["seed"],
                    "replayed_qwen",
                    admitter=adapter,
                )
            finally:
                adapter.close()
            assert row["topology_sha256"] == source["topology_sha256"]
            assert row["workload_sha256"] == source["workload_sha256"]
            assert row["max_latency_ms"] <= 6 + 1e-9
            row["inference_evidence"] = adapter.evidence()
            rows.append(row)
            nodes += adapter.node_samples
            peaks += adapter.peak_samples
            imbalances += adapter.imbalance_samples
            print(
                json.dumps({"scale": scale, "seed": source["seed"], "accepted": row["accepted"]}),
                flush=True,
            )
            write_json_new(seed_output / f"seed_{source['seed']}.json", row)
        total = requests * len(rows)
        summary = {
            "scale": scale,
            "accepted": sum(r["accepted"] for r in rows),
            "total_requests": total,
            "acceptance_percent": sum(r["accepted"] for r in rows) / total * 100,
            "per_seed": {str(r["seed"]): r["accepted"] for r in rows},
            "cpu_global_mean_percent": statistics.mean(
                s["cpu_percent"] for r in rows for s in r["utilization_samples"]
            ),
            "cpu_node_percent": distribution(nodes),
            "cpu_peak_node_percent": distribution(peaks),
            "cpu_imbalance_mean_pp": statistics.mean(imbalances),
            "latency_ms": distribution(
                v for r in rows for s in r["records"] for v in s["latencies_ms"]
            ),
            "decision_ms": distribution(
                s["decision_s"] * 1000 for r in rows for s in r["records"]
            ),
        }
        if scale == 2:
            summary["reproduces_source_acceptance"] = all(
                [s["accepted_immediate"] for s in a["records"]]
                == [s["accepted_immediate"] for s in b["records"]]
                for a, b in zip(rows, sources, strict=True)
            )
            assert summary["reproduces_source_acceptance"], "Replay baseline failed"
        write_json_new(args.output / f"scale_{scale_tag(scale)}.json", {"rows": rows})
        write_json_new(args.output / f"summary_scale_{scale_tag(scale)}.json", summary)
        summaries.append(summary)
    write_json_new(args.output / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()

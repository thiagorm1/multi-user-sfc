"""Paired CPU benchmark: original/optimized policy, isolated Ollama load and Qwen guidance."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
from loguru import logger

from muar_sfc.algorithms.mappo.muvr import TrainedMUVRMAPPO
from muar_sfc.config import SimulationSettings
from muar_sfc.core.ar_metrics import distribution
from muar_sfc.core.controller_metrics import ControllerCPUSampler
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, simulate


def baseline_class(revision):
    env_source = subprocess.check_output(
        ["git", "show", f"{revision}:src/muar_sfc/algorithms/mappo/env_mappo.py"], text=True
    )
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/muar_sfc/algorithms/mappo/muvr.py"], text=True
    )
    env = {"__name__": "_baseline_env"}
    exec(compile(env_source, "baseline_env.py", "exec"), env)
    namespace = {
        "__name__": "_baseline_muvr",
        "_env_class": env["MAPPO_SFC_Env"],
        "_comp": env["safe_comp_latency"],
    }
    source = source.replace(
        "from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env, safe_comp_latency",
        "MAPPO_SFC_Env, safe_comp_latency = _env_class, _comp",
    )
    exec(compile(source, "baseline_muvr.py", "exec"), namespace)
    return namespace["TrainedMUVRMAPPO"]


class OllamaLoadOnly(TrainedMUVRMAPPO):
    """Real background Qwen calls, but discard guidance to isolate CPU contention."""

    def _collect_background(self):
        super()._collect_background()
        self.guidance = None


class TimedAdmitter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.samples = []

    def prepare(self, *args):
        return self.adapter.prepare(*args)

    def __call__(self, *args):
        start, cpu = time.perf_counter(), time.thread_time()
        result = self.adapter(*args)
        self.samples.append(
            {
                "wall_ms": (time.perf_counter() - start) * 1000,
                "decision_thread_cpu_ms": (time.thread_time() - cpu) * 1000,
            }
        )
        return result


def signature(row):
    return {
        "topology": row["topology_sha256"],
        "workload": row["workload_sha256"],
        "requests": [
            {
                "accepted": r["accepted_immediate"],
                "latency": [round(v, 12) for v in r["latencies_ms"]],
            }
            for r in row["records"]
        ],
        "choices": [
            [(d["vnf"], d["selected_host"], d["search_attempt"]) for d in r["actor_decisions"]]
            for r in row["inference_evidence"]["request_evidence"]
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-ref", default="HEAD")
    p.add_argument("--model", default="rl_saved_models/MAPPO_muvr_meu_treino.pt", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--requests", type=int, default=50)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "original",
            "original_qwen",
            "optimized",
            "optimized_ollama_load",
            "optimized_qwen",
        ],
    )
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new output directory")
    if min(args.requests, args.threads) < 1:
        p.error("Requests and threads must be positive")
    original = baseline_class(args.baseline_ref)
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    logger.remove()
    logger.add(args.output / "qwen.log")
    variants = {
        "original": (original, False),
        "original_qwen": (original, True),
        "optimized": (TrainedMUVRMAPPO, False),
        "optimized_ollama_load": (OllamaLoadOnly, True),
        "optimized_qwen": (TrainedMUVRMAPPO, True),
    }
    if args.variants:
        variants = {name: variants[name] for name in args.variants}
    manifest = {
        "baseline_revision": subprocess.check_output(
            ["git", "rev-parse", args.baseline_ref], text=True
        ).strip(),
        "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "requests": args.requests,
        "seeds": args.seeds,
        "threads": args.threads,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "method": "Serial variants, reverse order on alternating seeds. Same trained checkpoint. "
        "Ollama-load mode discards guidance to isolate competing work. CPU sampled every 0.5s; "
        "100% process CPU = one core. CPU samples include the final worker drain; "
        "per-decision timings exclude it. "
        "Finite sample timing is not a causal performance guarantee.",
        "optimized_source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "src/muar_sfc/algorithms/mappo/env_mappo.py",
                "src/muar_sfc/algorithms/mappo/muvr.py",
            )
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = {name: [] for name in variants}
    # One unmeasured policy warmup per implementation before the measured runs.
    for cls in (original, TrainedMUVRMAPPO):
        adapter = cls(SimulationSettings(_env_file=None, mappo_model_path=str(args.model)))
        try:
            simulate(MUVRProfile(requests=1), 100042, "warmup", admitter=adapter)
        finally:
            adapter.close()
    for index, seed in enumerate(args.seeds):
        order = list(variants)
        if index % 2:
            order.reverse()
        for name in order:
            cls, qwen = variants[name]
            adapter = cls(
                SimulationSettings(
                    _env_file=None,
                    seed=seed,
                    mappo_model_path=str(args.model),
                    mappo_search_attempts=16,
                    cognitive_planner=qwen,
                    cognitive_llm_model="qwen2.5:1.5b",
                    cognitive_api_base_url="http://localhost:11434/v1",
                    cognitive_api_timeout=120,
                    cognitive_max_tokens=256,
                    cognitive_planner_interval=60,
                    cognitive_guidance_max_age=120,
                ),
                qwen=qwen,
            )
            timed = TimedAdmitter(adapter)
            with ControllerCPUSampler() as sampler:
                try:
                    row = simulate(MUVRProfile(requests=args.requests), seed, name, admitter=timed)
                finally:
                    adapter.close()
            row["inference_evidence"] = adapter.evidence()
            row["timing_samples"] = timed.samples
            row["cpu_samples"] = sampler.samples
            rows[name].append(row)
            (args.output / f"{name}.json").write_text(
                json.dumps({"rows": rows[name]}, allow_nan=False)
            )
            print(
                json.dumps(
                    {
                        "variant": name,
                        "seed": seed,
                        "accepted": row["accepted"],
                        "decision_ms": row["mean_decision_s"] * 1000,
                    }
                ),
                flush=True,
            )
    summary = {}
    for name, runs in rows.items():
        summary[name] = {
            "accepted": sum(r["accepted"] for r in runs),
            "total_requests": args.requests * len(runs),
            "acceptance_percent": sum(r["accepted"] for r in runs)
            / args.requests
            / len(runs)
            * 100,
            "decision_ms": distribution([s["wall_ms"] for r in runs for s in r["timing_samples"]]),
            "decision_thread_cpu_ms": distribution(
                [s["decision_thread_cpu_ms"] for r in runs for s in r["timing_samples"]]
            ),
            "admission_response_ms": distribution(
                [s["admission_response_s"] * 1000 for r in runs for s in r["records"]]
            ),
            "latency_ms": distribution(
                [v for r in runs for s in r["records"] for v in s["latencies_ms"]]
            ),
            "cpu": {
                field: distribution([s[field] for r in runs for s in r["cpu_samples"]])
                for field in ("controller_cpu_percent", "ollama_cpu_percent", "host_busy_percent")
            },
            "qwen_successful_calls": sum(
                r["inference_evidence"]["qwen_successful_calls"] for r in runs
            ),
            "qwen_failed_calls": sum(r["inference_evidence"]["qwen_failed_calls"] for r in runs),
        }
    for name in ("optimized", "optimized_ollama_load"):
        if name not in rows or "original" not in rows:
            continue
        summary[name]["same_policy_results_as_original"] = all(
            signature(a) == signature(b) for a, b in zip(rows["original"], rows[name], strict=True)
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    # Full traces are preserved per seed, not just the final seed of each variant.
    for name, runs in rows.items():
        (args.output / f"{name}.json").write_text(json.dumps({"rows": runs}, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

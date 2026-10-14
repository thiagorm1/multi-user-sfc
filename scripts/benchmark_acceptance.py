"""Seeded virtual-time replay of admission, deployment, bounded retries and expiry."""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import random
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["before", "improved", "heuristic", "qwen"], required=True
    )
    parser.add_argument(
        "--profile", choices=["basic", "sharing", "dag", "scaled", "full"], default="basic"
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--arrival-mean", type=float, default=3)
    parser.add_argument("--planner-interval", type=float, default=30)
    parser.add_argument("--max-wait", type=float, default=30)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    source_root = args.source_root or workspace
    sys.path.insert(0, str(source_root / "src"))
    # The replay's retry policy is identical even when loading a preserved baseline.
    spec = importlib.util.spec_from_file_location(
        "benchmark_admission_queue", workspace / "src/muar_sfc/controllers/admission_queue.py"
    )
    queue_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = queue_module
    spec.loader.exec_module(queue_module)
    AdmissionQueue = queue_module.AdmissionQueue

    import networkx as nx
    import numpy as np
    import torch
    from loguru import logger

    from muar_sfc.algorithms.mappo import MAPPOAlgorithm
    from muar_sfc.config import SimulationSettings
    from muar_sfc.controllers.cloud_llm_planner import CloudLLMPlanner
    from muar_sfc.controllers.modules.mobility_manager import MobilityManager
    from muar_sfc.controllers.modules.sfcs_instatiator import SFCInstatiator
    from muar_sfc.controllers.modules.sfcs_manager import SFCManager
    from muar_sfc.controllers.sfc_queue import SFCQueue
    from muar_sfc.core.poisson_emitter import PoissonEmitter
    from muar_sfc.core.scenarios.muar import MuarScenario
    from muar_sfc.topology.instantiator import TopologyInstantiator

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(args.output.with_suffix(".log"), level="WARNING")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    scaled = args.profile in ("scaled", "full")
    settings = SimulationSettings(
        _env_file=None,
        alg="mappo",
        topology="luxembourgv2",
        mobility=False,
        n_sessions=args.sessions,
        n_players=args.players,
        sfc_lifetime=120,
        latency=50.0,
        share=args.profile != "basic",
        shareband=args.profile != "basic",
        dag=args.profile in ("dag", "scaled", "full"),
        allow_delay=args.profile == "full",
        admission_max_wait=args.max_wait,
        backup=False,
        number_of_fails=0,
        verbose=False,
        cpu_capacity_factor=3.6 if scaled else 1,
        cache_capacity_factor=3.6 if scaled else 1,
        link_capacity_factor=8 if scaled else 1,
        wireless_capacity_factor=4 if scaled else 1,
        source_bandwidth_mbps=40000 if scaled else 0,
        cognitive_planner=args.variant in ("heuristic", "qwen"),
        cognitive_llm_model="qwen2.5:1.5b" if args.variant == "qwen" else "mock",
        cognitive_api_base_url="http://localhost:11434/v1",
        cognitive_api_timeout=60,
        cognitive_planner_interval=args.planner_interval,
    )
    rows, records = [], []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        topology = TopologyInstantiator().instantiate_topology("luxembourgv2", 0.7)
        network = topology.generate_substrate_network()
        if args.profile != "basic":
            from muar_sfc.core.capacity import apply_capacity_profile

            apply_capacity_profile(network, settings)
            network.set_bandwidth_sharing(settings.shareband)
        network.set_sharing_params(settings.share)
        network.verbose = False
        initial = copy.deepcopy(network.graph)
        source_queue = SFCQueue()
        scenario = MuarScenario(
            settings, source_queue, topology, PoissonEmitter(args.arrival_mean)
        )
        trace = []
        simulated_time = 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(args.sessions):
                simulated_time += float(np.random.poisson(args.arrival_mean))
                scenario.generate_sfc_session(None)
                for _ in range(args.players):
                    trace.append((simulated_time, source_queue.peek_sfc()))
        signature = [
            (
                t,
                [
                    (
                        s.id,
                        s.closer_router,
                        s.duration,
                        s.vnfs_dict,
                        list(s.graph.edges(data=True)),
                    )
                    for s in batch
                ],
            )
            for t, batch in trace
        ]
        arrival_signature = [
            (t, batch[0].dst_node, batch[0].duration, batch[0].closer_router) for t, batch in trace
        ]
        graph_signature = [(n, d) for n, d in initial.nodes(data=True)] + [
            (u, v, d) for u, v, d in initial.edges(data=True)
        ]

        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

        algorithm = MAPPOAlgorithm(model_path=args.model_path)
        admission = SFCInstatiator(algorithm, args=settings)
        admission.verbose = False
        manager = SFCManager(settings, None, algorithm)
        mobility = MobilityManager(settings)
        planner = CloudLLMPlanner(settings, network, algorithm)
        retry = AdmissionQueue(settings.allow_delay, args.max_wait, 1)
        planning_times, guidance_records, decision_times = [], [], []
        llm_success = llm_fallback = 0
        last_plan = None
        cursor = 0
        now = trace[0][0] if trace else 0
        wall_start = time.perf_counter()
        while cursor < len(trace) or retry.pending:
            next_arrival = trace[cursor][0] if cursor < len(trace) else float("inf")
            now = min(
                next_arrival, retry.next_time if retry.next_time is not None else float("inf")
            )
            manager.cleanup_network_resources(network, now)
            while cursor < len(trace) and trace[cursor][0] <= now:
                arrival, batch = trace[cursor]
                request = retry.register(batch, now, arrival)
                request.index = cursor
                cursor += 1
            if settings.cognitive_planner and (
                last_plan is None or now - last_plan >= args.planner_interval
            ):
                tick = time.perf_counter()
                if args.variant == "qwen":
                    original_call = planner._call_openai_compatible_api
                    observations = []

                    def observed_call(telemetry, call=original_call, seen=observations):
                        result = call(telemetry)
                        seen.append(result is not None)
                        return result

                    planner._call_openai_compatible_api = observed_call
                guidance = planner.generate_guidance()
                if args.variant == "qwen":
                    planner._call_openai_compatible_api = original_call
                    llm_success += sum(observations)
                    llm_fallback += len(observations) - sum(observations)
                planner.apply_guidance_to_algorithm()
                planning_times.append(time.perf_counter() - tick)
                guidance_records.append({"time": now, **guidance.model_dump()})
                last_plan = now
            while True:
                request = retry.next_ready(now)
                if request is None:
                    break
                random.seed(seed * 10000 + request.index)
                np.random.seed(seed * 10000 + request.index)

                def deploy(
                    batch,
                    mobile=mobility,
                    admitter=admission,
                    timings=decision_times,
                    macro=planner,
                    virtual_now=now,
                ):
                    nonlocal network, manager
                    mobile.register_mobile_user_in_network(batch, network)
                    dry_network = copy.deepcopy(network)
                    tick = time.perf_counter()
                    try:
                        solution, success = admitter.search_solution(batch, dry_network)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        solution, success = {}, False
                    timings.append(time.perf_counter() - tick)
                    if success:
                        trial_network = copy.deepcopy(network)
                        trial_manager = copy.deepcopy(manager)
                        result = trial_manager.submit_solution(batch, solution, trial_network)
                        success = result["is_success"]
                        if success:
                            assert all(
                                0 <= d.get("cpu_used", 0) <= d.get("cpu_capacity", 0) + 1e-6
                                and 0
                                <= d.get("cache_used", 0)
                                <= d.get("cache_capacity", 0) + 1e-6
                                and d.get("w_channel_used", 0)
                                <= d.get("w_channel_capacity", 0) + 1e-6
                                for graph in (trial_network.graph, trial_network.md_graph)
                                for _, d in graph.nodes(data=True)
                            )
                            assert all(
                                0
                                <= d["bandwidth_used"] + d.get("bandwidth_reserved", 0)
                                <= d["bandwidth_capacity"] + 1e-6
                                for _, _, d in trial_network.graph.edges(data=True)
                            )
                            assert all(
                                solution[s.id]["latency"] <= s.get_latency_request() for s in batch
                            )
                            network, manager = trial_network, trial_manager
                            macro.substrate_network = network
                            manager.tracker.sfcs_tracker[batch[0].dst_node]["timer"] = virtual_now
                            return True, [solution[s.id]["latency"] for s in batch]
                    network.remove_node(batch[0].dst_node)
                    mobile.remove_player(batch[0].dst_node)
                    return False, []

                retry.attempt(request, now, deploy)
        snapshot = retry.snapshot()
        assert snapshot["pending"] == 0
        dimensions = {
            "obs_dim": algorithm.env.obs_dim,
            "action_dim": algorithm.env.action_dim_per_agent,
            "n_agents": algorithm.env.n_regions,
            "state_dim": algorithm.env.state_dim,
        }
        row = {
            **{k: v for k, v in snapshot.items() if k != "records"},
            "variant": args.variant,
            "profile": args.profile,
            "seed": seed,
            "sfc_requests": sum(len(batch) for _, batch in trace),
            "accepted_sfcs": sum(
                len(r.batch) for r in retry.requests if r.status.startswith("accepted")
            ),
            "mean_decision_ms": float(np.mean(decision_times)) * 1000,
            "planning_calls": len(planning_times),
            "planning_seconds": sum(planning_times),
            "llm_success": llm_success,
            "llm_fallback": llm_fallback,
            "wall_seconds": time.perf_counter() - wall_start,
            "trace_sha256": digest(signature),
            "arrival_sha256": digest(arrival_signature),
            "topology_sha256": digest(graph_signature),
            "checkpoint_dimensions_match": all(
                checkpoint.get(k) == v for k, v in dimensions.items()
            ),
            "environment_dimensions": dimensions,
            "guidance": guidance_records,
            "final_links": [
                {
                    "link": [u, v],
                    "used": d["bandwidth_used"],
                    "capacity": d["bandwidth_capacity"]
                    if math.isfinite(d["bandwidth_capacity"])
                    else None,
                }
                for u, v, d in network.graph.edges(data=True)
            ],
        }
        rows.append(row)
        records.extend({"seed": seed, **r} for r in snapshot["records"])
        args.output.write_text(
            json.dumps(
                {
                    "settings": settings.model_dump(),
                    "profile": args.profile,
                    "arrival_mean": args.arrival_mean,
                    "model_sha256": hashlib.sha256(args.model_path.read_bytes()).hexdigest(),
                    "rows": rows,
                    "requests": records,
                    "method": "Paired virtual-time MUAR replay; transactional admission, "
                    "physical deployment with capacity/SLA audit, bounded retries and expiry.",
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("guidance", "final_links", "environment_dimensions")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

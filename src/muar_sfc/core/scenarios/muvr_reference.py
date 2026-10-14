"""Explicit MUVR reference profile; immediate transactional session admission."""

import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass

import networkx as nx

from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env, safe_comp_latency
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.ar_metrics import ar_frame_summary, distribution
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.orchestration.placement import available_path, stream_key


@dataclass(frozen=True)
class MUVRProfile:
    servers: int = 25
    probability: float = 0.20
    cpu: float = 100.0
    memory: float = 100.0
    bandwidth: float = 1000.0
    sla: float = 6.0
    requests: int = 50
    players: int = 4
    arrival_interval: float = 10.0
    lifetime: float = 120.0
    destination_mode: str = "same_access"
    cpu_per_session: float = 20.0
    memory_per_session: float = 15.0

    def __post_init__(self):
        if not (self.servers >= 2 and 0 <= self.probability <= 1):
            raise ValueError("Invalid graph parameters")
        if not (1 <= self.players <= 4 and self.requests > 0):
            raise ValueError("Use 1 to 4 players and positive request count")
        if min(self.cpu, self.memory, self.bandwidth, self.sla, self.arrival_interval) <= 0:
            raise ValueError("Capacities, SLA and arrival interval must be positive")
        if self.destination_mode not in ("same_access", "scattered"):
            raise ValueError("Invalid destination mode")
        if self.lifetime < 0:
            raise ValueError("Lifetime cannot be negative")


def build_network(profile, seed):
    rng = random.Random(seed)
    # No connectivity conditioning, resampling or added edges.
    graph = nx.erdos_renyi_graph(profile.servers, profile.probability, seed=seed)
    net = Net2()
    net.verbose = False
    net.set_sharing_params(True)
    net.set_bandwidth_sharing(True)
    for node in graph:
        net.add_node(node + 1, "server_b", profile.cpu, profile.memory, ips=rng.uniform(0.1, 0.2))
    for u, v in graph.edges:
        net.add_edge(u + 1, v + 1, profile.bandwidth, rng.uniform(1.0, 2.0))
    net.total_cpu_capacity = profile.servers * profile.cpu
    net.total_cache_capacity = profile.servers * profile.memory
    net.total_bandwidth_capacity = graph.number_of_edges() * profile.bandwidth
    return net


def generate_session(profile, index, rng):
    h, f = rng.uniform(0.65, 0.90), rng.uniform(0.17, 0.25)
    src = rng.randrange(1, profile.servers + 1)
    destinations = rng.sample(range(1, profile.servers + 1), profile.players)
    if profile.destination_mode == "same_access":
        destinations = [destinations[0]] * profile.players
    duration = rng.uniform(0, profile.lifetime)
    common_pred, common_view = 25 * h * f, 25 * h * (1 - f)
    unique_pred, unique_view = 25 * (1 - h) * f, 25 * (1 - h) * (1 - f)
    names = {
        k: f"MA_region_MUVR_{k}_{index}"
        for k in ("IA", "BE_pred", "BE_view", "pred_renderer", "view_renderer", "codec")
    }
    # Hypothesis: SF CPU/storage split proportional to processing input traffic.
    # Normalize the distinct SFs of the entire session to exactly 20 CPU / 15 memory.
    renderer_pred = common_pred + profile.players * unique_pred
    renderer_view = common_view + profile.players * (unique_view + 0.266)
    rates = {
        names["IA"]: common_pred
        + common_view
        + profile.players * (unique_pred + unique_view + 0.266),
        names["BE_pred"]: common_pred,
        names["BE_view"]: common_view,
        names["pred_renderer"]: renderer_pred,
        names["view_renderer"]: renderer_view,
        names["codec"]: renderer_pred + renderer_view,
    }
    for player in range(profile.players):
        rates.update(
            {
                f"FI_p{player}_{index}": 0.266,
                f"BE_pred_p{player}_{index}": unique_pred,
                f"BE_view_p{player}_{index}": unique_view,
            }
        )
    total = sum(rates.values())
    batch = []
    for player, dst in enumerate(destinations):
        fi, up, uv = (
            f"FI_p{player}_{index}",
            f"BE_pred_p{player}_{index}",
            f"BE_view_p{player}_{index}",
        )
        local = [
            names["IA"],
            names["BE_pred"],
            names["BE_view"],
            fi,
            up,
            uv,
            names["pred_renderer"],
            names["view_renderer"],
            names["codec"],
        ]
        ia, bp, bv, pr, vr, codec = (
            names[k]
            for k in ("IA", "BE_pred", "BE_view", "pred_renderer", "view_renderer", "codec")
        )
        dependencies = [
            ("src", ia, rates[ia]),
            (ia, bp, common_pred),
            (ia, bv, common_view),
            (ia, fi, 0.266),
            (ia, up, unique_pred),
            (ia, uv, unique_view),
            (bp, pr, common_pred),
            (up, pr, unique_pred),
            (bv, vr, common_view),
            (uv, vr, unique_view),
            (fi, vr, 0.266),
            (pr, codec, renderer_pred),
            (vr, codec, renderer_view),
            (codec, "dst", 25.266),
        ]
        definitions = [
            {
                "type": 2,
                "name": name,
                "CPU": profile.cpu_per_session * rates[name] / total,
                "cache": profile.memory_per_session * rates[name] / total,
                "in_bw": rates[name],
                "out_bw": rates[name],
            }
            for name in local
        ]
        sfc = SFCGenerator(
            {
                "name": f"muvr_p{player}_{index}",
                "vnf_list": definitions,
                "dependencies": dependencies,
                "bandwidth": 25.266,
                "src_node": src,
                "dst_node": dst,
                "closer_router": dst,
                "duration": duration,
                "latency": profile.sla,
            }
        ).generate()
        sfc.session_id = str(index)
        # Aggregate shared renderer input over all players for compute accounting.
        for name in (ia, pr, vr, codec):
            sfc.get_vnf_by_id(name).set_income_interface_bandwidth(rates[name])
        batch.append(sfc)
    return batch, {
        "h": h,
        "f": f,
        "source": src,
        "destinations": destinations,
        "lifetime_s": duration,
        "distinct_sfs": len(rates),
    }


def plan_colocated(net, batch, host):
    placement = host if isinstance(host, dict) else {v: host for s in batch for v in s.vnfs}
    trial = copy.deepcopy(net)
    latencies = []
    for original in batch:
        sfc = copy.deepcopy(original)
        sfc.edge_routes, sfc.edge_latencies, sfc.comp_latencies = {}, {}, {}
        route_info = {"src": [sfc.src.get_substrate_node()], "dst": [sfc.dst_node]}
        for name, vnf in sfc.vnfs.items():
            node = placement[name]
            trial.allocate_microservice(sfc, vnf, node)
            route_info[name] = [node]
            sfc.comp_latencies[name] = safe_comp_latency(trial.graph, node, vnf)
        for u, v, data in sfc.graph.edges(data=True):
            source = sfc.src.get_substrate_node() if u == "src" else placement[u]
            target = sfc.dst_node if v == "dst" else placement[v]
            key = stream_key(sfc, u, v, True)
            path = available_path(trial.graph, source, target, data["bandwidth"], key, True)
            if not path:
                raise ValueError("bandwidth_or_connectivity")
            latency = 0.0
            for a, b in zip(path[:-1], path[1:], strict=True):
                # Use the physical allocator for metrics and reference counts.
                trial.allocate_bandwidth(a, b, data["bandwidth"], key)
                latency += trial.graph.edges[a, b]["latency"]
            sfc.edge_routes[u, v] = path
            sfc.edge_latencies[u, v] = latency
        latency = sfc.get_critical_path_latency(sfc.edge_latencies, sfc.comp_latencies)
        if latency > sfc.get_latency_request() + 1e-9:
            raise ValueError("sla")
        trial.sfc_dict[sfc.id] = sfc
        trial.sfc_route_info[sfc.id] = route_info
        latencies.append(latency)
    return trial, latencies


def latency_lower_bound(net, batch):
    bounds = []
    for sfc in batch:
        try:
            communication = nx.shortest_path_length(
                net.graph, sfc.src.get_substrate_node(), sfc.dst_node, weight="latency"
            )
        except nx.NetworkXNoPath:
            return float("inf")
        compute = {
            name: min(safe_comp_latency(net.graph, node, vnf) for node in net.graph)
            for name, vnf in sfc.vnfs.items()
        }
        bounds.append(
            communication
            + sfc.get_critical_path_latency({edge: 0.0 for edge in sfc.graph.edges}, compute)
        )
    return max(bounds)


def admit(net, batch, strategy="joint"):
    if latency_lower_bound(net, batch) > batch[0].get_latency_request():
        return None, [], "physical_latency_lower_bound"
    # Exact enumeration of feasible whole-session hosts; no capacity/SLA relaxation.
    hosts = list(net.graph)
    if strategy == "anchor":
        hosts = [batch[0].src.get_substrate_node()]
    candidates = []
    for host in hosts:
        try:
            trial, latencies = plan_colocated(net, batch, host)
        except ValueError:
            continue
        load = trial.graph.nodes[host]["cpu_used"] / trial.graph.nodes[host]["cpu_capacity"]
        candidates.append(((max(latencies), load), trial, latencies))
    if candidates:
        _, trial, latencies = min(candidates, key=lambda candidate: candidate[0])
        return trial, latencies, "colocated"
    if strategy == "anchor":
        return None, [], "no_feasible_anchor"
    # Joint two-stage search: object handling upstream; rendering/coding downstream.
    # Rank all pairs by critical-path lower cost, then physically validate candidates.
    distances = dict(nx.all_pairs_dijkstra_path_length(net.graph, weight="latency"))
    ranked = []
    for front in hosts:
        for back in hosts:
            mapping = {
                name: (back if any(tag in name for tag in ("renderer", "codec")) else front)
                for s in batch
                for name in s.vnfs
            }
            worst = 0.0
            for sfc in batch:
                edge_latencies = {}
                for u, v in sfc.graph.edges:
                    a = sfc.src.get_substrate_node() if u == "src" else mapping[u]
                    b = sfc.dst_node if v == "dst" else mapping[v]
                    edge_latencies[u, v] = distances.get(a, {}).get(b, float("inf"))
                comp = {
                    name: safe_comp_latency(net.graph, mapping[name], vnf)
                    for name, vnf in sfc.vnfs.items()
                }
                worst = max(worst, sfc.get_critical_path_latency(edge_latencies, comp))
            if worst <= batch[0].get_latency_request():
                ranked.append((worst, front, back, mapping))
    for _, _, _, mapping in sorted(ranked, key=lambda value: value[:3]):
        try:
            trial, latencies = plan_colocated(net, batch, mapping)
            return trial, latencies, "two_stage"
        except ValueError:
            continue
    # Three stages allow IA, object handling, and coding to use separate hosts.
    # Cheap exact latency ranking avoids thousands of speculative graph copies.
    representative = batch[0]
    ids = list(representative.vnfs)
    ia = next(name for name in ids if "_IA_" in name)
    codec = next(name for name in ids if "_codec_" in name)
    pr = next(name for name in ids if "_pred_renderer_" in name)
    vr = next(name for name in ids if "_view_renderer_" in name)
    objects = [name for name in ids if name not in (ia, codec, pr, vr)]
    comps = {
        node: {
            name: safe_comp_latency(net.graph, node, vnf)
            for name, vnf in representative.vnfs.items()
        }
        for node in hosts
    }
    ranked = []
    source = representative.src.get_substrate_node()
    for front in hosts:
        for middle in hosts:
            for back in hosts:
                comm = (
                    distances.get(source, {}).get(front, float("inf"))
                    + distances.get(front, {}).get(middle, float("inf"))
                    + distances.get(middle, {}).get(back, float("inf"))
                )
                branch = max(
                    comps[middle][name] + comps[back][pr if "pred" in name else vr]
                    for name in objects
                )
                finish = (
                    comm
                    + comps[front][ia]
                    + branch
                    + comps[back][codec]
                    + max(distances.get(back, {}).get(s.dst_node, float("inf")) for s in batch)
                )
                if finish <= representative.get_latency_request():
                    ranked.append((finish, front, middle, back))
    for _, front, middle, back in sorted(ranked):
        mapping = {
            name: (
                front
                if "_IA_" in name
                else back
                if any(tag in name for tag in ("renderer", "codec"))
                else middle
            )
            for s in batch
            for name in s.vnfs
        }
        try:
            trial, latencies = plan_colocated(net, batch, mapping)
            return trial, latencies, "three_stage"
        except ValueError:
            continue
    # Distributed fallback uses the same hard feasibility engine as MAPPO,
    # with deterministic resource/latency ranking; no incompatible learned weights.
    trial = copy.deepcopy(net)
    latencies = []
    for original in batch:
        sfc = copy.deepcopy(original)
        env = MAPPO_SFC_Env(
            trial.graph, sfc, cloud_node=-1, share=True, shareband=True, policy_weight=0
        )
        for _ in env.vnf_sequence:
            _, _, _, done, _ = env.step({})
            if done:
                break
        if not env.success:
            return None, [], "no_feasible_joint_plan"
        sfc.edge_routes = env.edge_routes
        sfc.edge_latencies = env.edge_latencies
        sfc.comp_latencies = env.comp_latencies
        trial.deploy_sfc(sfc, env.route_info)
        latencies.append(env.total_latency)
    return trial, latencies, "distributed"


def audit(net):
    for _, node in net.graph.nodes(data=True):
        for resource in ("cpu", "cache"):
            if not -1e-7 <= node[f"{resource}_used"] <= node[f"{resource}_capacity"] + 1e-7:
                raise AssertionError(f"Invalid {resource} allocation")
    for _, _, edge in net.graph.edges(data=True):
        if not -1e-7 <= edge["bandwidth_used"] <= edge["bandwidth_capacity"] + 1e-7:
            raise AssertionError("Invalid bandwidth allocation")


def simulate(
    profile, seed, strategy="joint", admitter=None, frame_samples=None, ar_budget_ms=None
):
    net = build_network(profile, seed)
    topology = {
        "nodes": list(net.graph.nodes(data=True)),
        "edges": list(net.graph.edges(data=True)),
    }
    topology_hash = hashlib.sha256(json.dumps(topology, sort_keys=True).encode()).hexdigest()
    rng = random.Random(seed + 100000)
    records, active, samples = [], [], []
    previous_time = 0.0
    integrals = {"cpu": 0.0, "memory": 0.0, "bandwidth": 0.0}

    def integrate(until):
        nonlocal previous_time
        dt = until - previous_time
        integrals["cpu"] += dt * net.metrics.total_cpu_used / net.total_cpu_capacity
        integrals["memory"] += dt * net.metrics.total_cache_used / net.total_cache_capacity
        integrals["bandwidth"] += (
            dt * net.metrics.total_bandwidth_used / max(1, net.total_bandwidth_capacity)
        )
        previous_time = until

    for index in range(profile.requests):
        arrival = index * profile.arrival_interval
        for expiry, ids in sorted(active):
            if expiry <= arrival:
                integrate(expiry)
                for sfc_id in ids:
                    net.undeploy_sfc(sfc_id)
        active = [(expiry, ids) for expiry, ids in active if expiry > arrival]
        integrate(arrival)
        batch, metadata = generate_session(profile, index, rng)
        for sfc in batch:
            sfc.arrival_time = arrival
        lower_bound = latency_lower_bound(net, batch)
        prepare = getattr(admitter, "prepare", None)
        preparation_start = time.perf_counter()
        if prepare:
            prepare(net, batch)
        preparation_seconds = time.perf_counter() - preparation_start
        tick = time.perf_counter()
        candidate, latencies, reason = (admitter or admit)(net, batch, strategy)
        decision = time.perf_counter() - tick
        if candidate is not None:
            net = candidate
            active.append((arrival + batch[0].duration, [s.id for s in batch]))
        audit(net)
        records.append(
            {
                "index": index,
                "arrival_s": arrival,
                **metadata,
                "accepted_immediate": candidate is not None,
                "attempts": 1,
                "wait_s": 0,
                "decision_s": decision,
                "background_submission_s": preparation_seconds,
                "admission_response_s": decision + preparation_seconds,
                "latency_lower_bound_ms": lower_bound if lower_bound != float("inf") else None,
                "latencies_ms": latencies,
                "placement": reason,
            }
        )
        samples.append(
            {
                "time_s": arrival,
                "cpu_percent": 100 * net.metrics.total_cpu_used / net.total_cpu_capacity,
                "memory_percent": 100 * net.metrics.total_cache_used / net.total_cache_capacity,
                "bandwidth_percent": 100
                * net.metrics.total_bandwidth_used
                / max(1, net.total_bandwidth_capacity),
            }
        )
    # Audit full release and include the observation tail in time-weighted utilization.
    for expiry, ids in sorted(active):
        integrate(expiry)
        for sfc_id in ids:
            net.undeploy_sfc(sfc_id)
    audit(net)
    assert all(
        abs(n["cpu_used"]) < 1e-7 and abs(n["cache_used"]) < 1e-7
        for _, n in net.graph.nodes(data=True)
    )
    assert all(abs(e["bandwidth_used"]) < 1e-7 for _, _, e in net.graph.edges(data=True))
    accepted = sum(r["accepted_immediate"] for r in records)
    latencies = [lat for r in records for lat in r["latencies_ms"]]
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
    workload_hash = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    return {
        "workload_sha256": workload_hash,
        "seed": seed,
        "strategy": strategy,
        "requests": profile.requests,
        "accepted": accepted,
        "physical_sla_upper_bound_percent": 100
        * sum(
            r["latency_lower_bound_ms"] is not None and r["latency_lower_bound_ms"] <= profile.sla
            for r in records
        )
        / profile.requests,
        "acceptance_percent": 100 * accepted / profile.requests,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "max_latency_ms": max(latencies, default=None),
        "accumulated_latency_ms": sum(latencies),
        "latency_samples": len(latencies),
        "mean_decision_s": sum(r["decision_s"] for r in records) / profile.requests,
        "latency_distribution_ms": distribution(latencies),
        "decision_distribution_ms": distribution([1000 * r["decision_s"] for r in records]),
        "admission_response_distribution_ms": distribution(
            [1000 * r["admission_response_s"] for r in records]
        ),
        "ar_telemetry": ar_frame_summary(frame_samples, ar_budget_ms),
        "time_weighted_utilization_percent": {
            k: 100 * v / max(1, previous_time) for k, v in integrals.items()
        },
        "observation_seconds": previous_time,
        "topology_sha256": topology_hash,
        "topology": topology,
        "records": records,
        "utilization_samples": samples,
    }


def run_configured(settings):
    from dataclasses import asdict
    from pathlib import Path

    if (
        any(
            value != 1
            for value in (
                settings.cpu_capacity_factor,
                settings.cache_capacity_factor,
                settings.link_capacity_factor,
                settings.wireless_capacity_factor,
            )
        )
        or settings.allow_delay
    ):
        raise ValueError("MUVR reference requires unscaled resources and immediate admission")
    if settings.alg not in ("joint", "mappo"):
        raise ValueError("MUVR reference supports joint or trained mappo")
    if settings.alg == "joint" and settings.cognitive_planner:
        raise ValueError("Enable alg=mappo to use Qwen with the trained policy")
    if (
        settings.source_bandwidth_mbps
        or settings.mobility
        or settings.backup
        or settings.number_of_fails
        or settings.number_of_link_fails
    ):
        raise ValueError("MUVR reference requires static, failure-free, unscaled infrastructure")
    profile = MUVRProfile(
        servers=settings.number_of_nodes,
        requests=settings.n_sessions,
        players=settings.n_players,
        sla=settings.latency,
        lifetime=settings.sfc_lifetime,
        arrival_interval=settings.official_rate,
        destination_mode=settings.muvr_destination_mode,
    )
    if settings.alg == "mappo":
        import torch

        from muar_sfc.algorithms.mappo.muvr import TrainedMUVRMAPPO

        torch.set_num_threads(4)
        adapter = TrainedMUVRMAPPO(settings, qwen=settings.cognitive_planner)
        variant = "mappo_qwen" if settings.cognitive_planner else "mappo"
        try:
            row = simulate(
                profile,
                settings.seed,
                variant,
                admitter=adapter,
                ar_budget_ms=settings.ar_motion_to_photon_budget_ms,
            )
        finally:
            adapter.close()
        row["inference_evidence"] = adapter.evidence()
        directory = "mappo_qwen_muvr_async"
        if profile.requests != 50:
            directory += f"_{profile.requests}"
        output = Path("results") / directory / f"configured_{variant}_seed_{settings.seed}.json"
    else:
        row = simulate(profile, settings.seed)
        output = Path("results/muvr_50_reference") / f"configured_seed_{settings.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"profile": asdict(profile), "rows": [row]}, indent=2, allow_nan=False)
    )
    display = {
        k: v
        for k, v in row.items()
        if k not in ("topology", "records", "utilization_samples", "inference_evidence")
    }
    if "inference_evidence" in row:
        evidence = row["inference_evidence"]
        display.update(
            {
                key: evidence[key]
                for key in ("checkpoint_loaded", "qwen_successful_calls", "qwen_fallback_calls")
            }
        )
    print(json.dumps(display, indent=2))
    print(f"Results: {output}")

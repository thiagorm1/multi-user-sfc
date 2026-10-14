import copy
import random

import networkx as nx
import pytest

from muar_sfc.core.scenarios.muvr_reference import (
    MUVRProfile,
    admit,
    audit,
    build_network,
    generate_session,
    plan_colocated,
    simulate,
)


def test_topology_keeps_nominal_capacity_and_exact_er_draw():
    profile = MUVRProfile()
    net = build_network(profile, 42)
    expected = nx.erdos_renyi_graph(25, 0.2, seed=42)
    assert {(u - 1, v - 1) for u, v in net.graph.edges} == set(expected.edges)
    assert len(net.graph) == 25
    assert all(
        n["cpu_capacity"] == n["cache_capacity"] == 100 and 1e10 <= n["ips"] <= 2e10
        for _, n in net.graph.nodes(data=True)
    )
    assert all(
        e["bandwidth_capacity"] == 1000 and 1 <= e["latency"] <= 2
        for _, _, e in net.graph.edges(data=True)
    )


def test_full_session_has_18_functions_exact_cpu_memory_and_unique_fi():
    batch, metadata = generate_session(MUVRProfile(), 7, random.Random(42))
    unique = {v.id: v for s in batch for v in s.vnfs.values()}
    assert len(unique) == metadata["distinct_sfs"] == 18
    assert sum(v.get_cpu_request() for v in unique.values()) == pytest.approx(20)
    assert sum(v.get_cache_request() for v in unique.values()) == pytest.approx(15)
    assert len([name for name in unique if name.startswith("FI_")]) == 4
    for sfc in batch:
        assert sfc.get_latency_request() == 6
        fi = next(name for name in sfc.vnfs if name.startswith("FI_"))
        assert sfc.graph.edges[next(sfc.graph.predecessors(fi)), fi]["bandwidth"] == 0.266
    assert 0.65 <= metadata["h"] <= 0.9
    assert 0.17 <= metadata["f"] <= 0.25
    assert 0 <= metadata["lifetime_s"] <= 120


def test_failed_session_is_atomic_and_sla_not_relaxed():
    profile = MUVRProfile(sla=0.00001)
    net = build_network(profile, 42)
    before = copy.deepcopy(net.graph)
    batch, _ = generate_session(profile, 0, random.Random(42))
    trial, _, _ = admit(net, batch)
    assert trial is None
    assert nx.utils.graphs_equal(before, net.graph)
    assert not net.sfc_dict


def test_shared_session_deployment_and_release_in_reverse_order():
    profile = MUVRProfile(sla=100)
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(42))
    trial, _ = plan_colocated(net, batch, 1)
    assert trial.metrics.total_cpu_used == pytest.approx(20)
    assert trial.metrics.total_cache_used == pytest.approx(15)
    for sfc in reversed(batch):
        trial.undeploy_sfc(sfc.id)
        audit(trial)
    assert trial.metrics.total_cpu_used == pytest.approx(0)
    assert trial.metrics.total_cache_used == pytest.approx(0)
    assert trial.metrics.total_bandwidth_used == pytest.approx(0)


def test_immediate_replay_has_no_retries_or_wait():
    row = simulate(MUVRProfile(requests=3), 42)
    assert row["requests"] == 3
    assert all(r["attempts"] == 1 and r["wait_s"] == 0 for r in row["records"])
    assert all(lat <= 6 for r in row["records"] for lat in r["latencies_ms"])
    assert [r["arrival_s"] for r in row["records"]] == [0, 10, 20]


@pytest.mark.parametrize("players", [1, 2, 3, 4])
def test_sf_count_grows_only_in_unique_object_view_branches(players):
    batch, metadata = generate_session(MUVRProfile(players=players), 3, random.Random(5))
    assert metadata["distinct_sfs"] == 6 + 3 * players
    assert len({v.id for s in batch for v in s.vnfs.values()}) == 6 + 3 * players


def test_rejected_request_diagnoses_physical_delay_lower_bound():
    profile = MUVRProfile(requests=2, sla=0.001)
    row = simulate(profile, 42)
    assert row["accepted"] == 0
    assert row["physical_sla_upper_bound_percent"] == 0
    assert all(r["placement"] == "physical_latency_lower_bound" for r in row["records"])

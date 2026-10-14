import random

import networkx as nx
import pytest

from muar_sfc.algorithms.muvr_baselines import MUVRBaseline
from muar_sfc.core.scenarios.muvr_reference import (
    MUVRProfile,
    audit,
    build_network,
    generate_session,
)


@pytest.mark.parametrize("strategy", ["greedy", "musfico"])
def test_dag_baselines_validate_full_physical_plan_without_mutating_live_network(strategy):
    profile = MUVRProfile()
    network = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    baseline = MUVRBaseline(strategy, beam_width=2)
    trial, latencies, _ = baseline(network, batch, strategy)
    assert trial is not None
    assert len(latencies) == 4 and max(latencies) <= 6
    assert len(trial.sfc_dict) == 4
    audit(trial)
    assert all(set(sfc.edge_routes) == set(sfc.graph.edges) for sfc in trial.sfc_dict.values())
    assert all(
        node["cpu_used"] == node["cache_used"] == 0 for _, node in network.graph.nodes(data=True)
    )
    assert not network.sfc_dict
    for sfc in batch:
        trial.undeploy_sfc(sfc.id)
    assert trial.metrics.total_cpu_used == pytest.approx(0)
    assert trial.metrics.total_cache_used == pytest.approx(0)
    assert trial.metrics.total_bandwidth_used == pytest.approx(0)


@pytest.mark.parametrize("strategy", ["greedy", "musfico"])
def test_dag_baselines_reject_infeasible_sla_and_preserve_capacity(strategy):
    profile = MUVRProfile(sla=0.001)
    network = build_network(profile, 42)
    before = network.graph.copy()
    batch, _ = generate_session(profile, 0, random.Random(100042))
    trial, latencies, _ = MUVRBaseline(strategy)(network, batch, strategy)
    assert trial is None and not latencies
    assert nx.utils.graphs_equal(before, network.graph)
    assert not network.sfc_dict

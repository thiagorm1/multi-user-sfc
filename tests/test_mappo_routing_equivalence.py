"""Fixed-seed reference routing: exact hosts, routes, resources and latencies."""

import copy
import random

import numpy as np
import pytest

from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env
from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv
from muar_sfc.core.orchestration.placement import available_path, available_paths
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, build_network, generate_session


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_cached_routing_matches_uncached_fixed_seed(seed):
    profile = MUVRProfile()
    net = build_network(profile, seed)
    batch, _ = generate_session(profile, 0, random.Random(100000 + seed))
    for sfc in batch:
        optimized = MUVRMAPPOEnv(net.graph, sfc)
        reference = MUVRMAPPOEnv(net.graph, sfc)
        reference._input_path = MAPPO_SFC_Env._input_path.__get__(reference)
        rng = random.Random(seed)
        done = False
        while not done:
            masks = optimized.get_action_masks()
            reference_masks = reference.get_action_masks()
            for region in masks:
                np.testing.assert_array_equal(masks[region], reference_masks[region])
            actions = {
                region: rng.choice(list(np.flatnonzero(mask)) or [0])
                for region, mask in masks.items()
            }
            result = optimized.step(actions)
            expected = reference.step(actions)
            done = result[3]
            assert result[2:] == expected[2:]
            for region in result[0]:
                np.testing.assert_array_equal(result[0][region], expected[0][region])
            np.testing.assert_array_equal(result[1], expected[1])
            assert optimized.actor_decisions == reference.actor_decisions
            assert optimized.edge_routes == reference.edge_routes
            assert optimized.total_latency == reference.total_latency
            assert dict(optimized.graph.nodes(data=True)) == dict(reference.graph.nodes(data=True))
            assert dict(optimized.graph.edges) == dict(reference.graph.edges)
        assert optimized.success == reference.success


@pytest.mark.parametrize("saturated", [False, True])
def test_single_source_paths_match_target_search_with_residual_bandwidth(saturated):
    graph = build_network(MUVRProfile(), 42).graph
    rng = random.Random(137)
    for _, _, data in graph.edges(data=True):
        data["bandwidth_used"] = 999.9 if saturated else rng.uniform(0, 1000)
        data["services_in_transit"] = {
            "fixture": {"bw_per_copy": 25, "bw_used": 25, "copys": 1}
        } if not saturated and rng.random() < 0.25 else {}
    # Saturated links must not become feasible via cached propagation distances.
    for source in graph:
        paths = available_paths(graph, source, 25, "fixture", True)
        for target in graph:
            assert paths.get(target, []) == available_path(
                graph, source, target, 25, "fixture", True
            )


def test_input_paths_invalidate_on_graph_function_exclusions_and_reset(monkeypatch):
    import muar_sfc.algorithms.mappo.muvr as module

    profile = MUVRProfile()
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    env = MUVRMAPPOEnv(net.graph, batch[0])
    calls = []
    original = module.available_paths

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "available_paths", counted)

    def query():
        return env._input_path(env._copy_trial(), env.src_node, env.cloud_node, 1, "fixture", True)

    query()
    query()
    assert len(calls) == 1
    env.excluded_nodes[env.vnf_sequence[0].id] = {env.cloud_node}
    query()
    assert len(calls) == 2
    env.graph = copy.deepcopy(env.graph)
    query()
    assert len(calls) == 3
    env.current_vnf_idx += 1
    query()
    assert len(calls) == 4
    env.reset_sfc(env.sfc)
    assert env._input_paths_cache == {}
    query()
    assert len(calls) == 5

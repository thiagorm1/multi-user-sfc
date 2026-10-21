import copy
import random

import numpy as np

from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv
from muar_sfc.core.orchestration.placement import reserve_bandwidth, reserve_node
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, build_network, generate_session


def environment():
    p = MUVRProfile(sla=100)
    net = build_network(p, 42)
    batch, _ = generate_session(p, 0, random.Random(100042))
    return MUVRMAPPOEnv(net.graph, batch[0])


def test_trial_detaches_host_and_edge_without_mutating_parent_or_sibling():
    env = environment()
    node = env.cloud_node
    reserve_node(env.graph, env.sfc, env.vnf_sequence[0], node, True)
    u, v = next(iter(env.graph.edges))
    reserve_bandwidth(env.graph.edges[u, v], "test", 1, True)
    before = copy.deepcopy(env.graph)
    a, b = env._copy_trial(), env._copy_trial()
    env._detach_node(a, node)
    reserve_node(a, env.sfc, env.vnf_sequence[0], node, True)
    u, v = next(iter(a.edges))
    env._detach_edge(a, u, v)
    reserve_bandwidth(a.edges[u, v], "test", 1, True)
    # Repeated reservations detach once and retain their own cumulative state.
    env._detach_edge(a, v, u)
    reserve_bandwidth(a.edges[v, u], "test", 1, True)
    assert a.edges[u, v]["services_in_transit"]["test"]["copys"] == 3
    assert a.edges[u, v] is a.edges[v, u]
    assert dict(env.graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert dict(b.nodes(data=True)) == dict(before.nodes(data=True))
    assert dict(env.graph.edges) == dict(before.edges)
    assert dict(b.edges) == dict(before.edges)


def test_masks_reuse_candidates_and_invalidate_on_transition_exclusion_and_reset(monkeypatch):
    env = environment()
    compute = env._candidate
    calls = []

    def counted(node):
        calls.append(node)
        return compute(node)

    monkeypatch.setattr(env, "_candidate", counted)
    masks = env.get_action_masks()
    n = len(calls)
    env.get_action_masks()
    assert len(calls) == n
    node = env.cloud_node
    env.excluded_nodes[env.vnf_sequence[0].id] = {node}
    assert env._evaluated_candidate(node) is None
    assert len(calls) == n + 1
    env.excluded_nodes = {}
    actions = {r: int(np.flatnonzero(m)[0]) for r, m in masks.items()}
    env.get_action_masks()
    n = len(calls)
    env.step(actions)
    assert len(calls) == n  # ranking reused the masked candidates
    env.get_action_masks()
    assert len(calls) > n
    env.reset_sfc(env.sfc)
    assert env._candidate_cache == {}


def test_state_uses_same_observation_without_repeating_feature_extraction(monkeypatch):
    env = environment()
    original = env.get_observations
    calls = []

    def counted():
        calls.append(1)
        return original()

    monkeypatch.setattr(env, "get_observations", counted)
    obs, state = env._observation_state()
    assert len(calls) == 1
    assert np.array_equal(state[:192], np.concatenate(list(obs.values())))

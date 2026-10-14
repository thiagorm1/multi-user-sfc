import random

import pytest

from muar_sfc.algorithms.mappo.mappo_core import MAPPOTrainer
from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv, TrainedMUVRMAPPO, dimensions
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.cloud_llm_planner import CognitiveGuidance
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, build_network, generate_session


def test_four_regions_are_stable_and_original_capacities_unchanged():
    for seed in (42, 43, 44):
        net = build_network(MUVRProfile(), seed)
        batch, _ = generate_session(MUVRProfile(), 0, random.Random(seed + 100000))
        env = MUVRMAPPOEnv(net.graph, batch[0])
        assert dimensions(env) == dict(n_agents=4, obs_dim=48, action_dim=7, state_dim=196)
        assert all(len(nodes) == 6 for nodes in env.region_nodes.values())
        assert all(len(obs) == 48 for obs in env.get_observations().values())
        assert len(env.get_global_state()) == 196
        assert len(net.graph) == 25
        assert all(
            data["cpu_capacity"] == data["cache_capacity"] == 100
            for _, data in net.graph.nodes(data=True)
        )
        assert all(data["position"] == (0, 0) for _, data in net.graph.nodes(data=True))


def test_only_actor_proposals_can_be_selected():
    profile = MUVRProfile(sla=100)
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    env = MUVRMAPPOEnv(net.graph, batch[0])
    env.step({region: env.max_nodes_per_region for region in range(4)})
    assert env.actor_decisions[0]["selected_host"] == env.cloud_node
    assert set(env.actor_decisions[0]["proposals"].values()) == {env.cloud_node}


def test_untrained_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "untrained.pt"
    MAPPOTrainer(4, 48, 7, 196).save(path)
    settings = SimulationSettings(_env_file=None, mappo_model_path=str(path))
    with pytest.raises(ValueError, match="training"):
        TrainedMUVRMAPPO(settings)


def test_qwen_cannot_be_reported_if_disabled():
    settings = SimulationSettings(_env_file=None, cognitive_planner=False)
    with pytest.raises(ValueError, match="Qwen must be explicitly enabled"):
        TrainedMUVRMAPPO(settings, qwen=True)


def test_qwen_preferences_change_ranking_only_within_actor_proposals():
    net = Net2()
    for node in (1, 2):
        net.add_node(node, "server", cpu_capacity=100, cache_capacity=100, ips=0.1)
    net.add_edge(1, 2, bandwidth_capacity=1000, latency=1)
    sfc = SFCGenerator(
        {
            "name": "test_p1_1",
            "vnf_list": [
                {"type": 2, "name": "vnf", "CPU": 10, "cache": 10, "in_bw": 1, "out_bw": 1}
            ],
            "bandwidth": 1,
            "src_node": 1,
            "dst_node": 2,
            "closer_router": 2,
            "duration": 120,
            "latency": 6,
        }
    ).generate()
    plain = MUVRMAPPOEnv(net.graph, sfc, n_regions=2)
    guided = MUVRMAPPOEnv(
        net.graph, sfc, n_regions=2, guidance=CognitiveGuidance(priority_cache_nodes=[2])
    )
    actions = {0: 0, 1: plain.max_nodes_per_region}
    plain.step(actions)
    guided.step(actions)
    assert plain.allocated_nodes["vnf"] == 1
    assert guided.allocated_nodes["vnf"] == 2
    assert guided.total_latency <= 6
    assert set(guided.actor_decisions[0]["proposals"].values()) == {1, 2}


def test_llm_sync_factor_cannot_relax_sla():
    profile = MUVRProfile(sla=0.001)
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    env = MUVRMAPPOEnv(net.graph, batch[0], guidance=CognitiveGuidance(sync_tolerance_factor=3))
    env.step({region: env.max_nodes_per_region for region in range(4)})
    assert not env.success
    assert not env.allocated_nodes
    assert env.sync_tolerance == batch[0].sync_tolerance


def test_compact_qwen_telemetry_preserves_every_node_field():
    from muar_sfc.algorithms.mappo.muvr import MUVRQwenPlanner

    profile = MUVRProfile()
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    planner = MUVRQwenPlanner(SimulationSettings(_env_file=None), net)
    planner.current_batch = batch
    snapshot = planner.extract_telemetry_snapshot()
    compact = planner.compact_telemetry(snapshot)
    reconstructed = [dict(zip(compact["columns"], row, strict=True)) for row in compact["nodes"]]
    assert reconstructed == snapshot["nodes_telemetry"]
    assert len(compact["nodes"]) == 25
    assert compact["sla_ms"] == 6
    schema = planner.response_schema(snapshot)
    assert schema["properties"]["priority_cache_nodes"]["maxItems"] == 3

import pytest
from pydantic import ValidationError

from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.net_v2 import Net2
from muar_sfc.experiment_config import ExperimentConfig, command_plan


@pytest.mark.parametrize("scale", [1.5, 0.0])
def test_lower_penalty_can_choose_nearby_loaded_host_without_relaxing_sla(scale):
    net = Net2()
    for node in (1, 2):
        net.add_node(node, "server", cpu_capacity=100, cache_capacity=100, ips=1e10)
    net.add_edge(1, 2, bandwidth_capacity=1000, latency=0.9)
    net.graph.nodes[1]["cpu_used"] = net.graph.nodes[1]["cache_used"] = 50
    sfc = SFCGenerator(
        {
            "name": "test_p1_1",
            "vnf_list": [
                {"type": 2, "name": "vnf", "CPU": 10, "cache": 10, "in_bw": 1, "out_bw": 1}
            ],
            "bandwidth": 1,
            "src_node": 1,
            "dst_node": 1,
            "closer_router": 1,
            "duration": 120,
            "latency": 6,
        }
    ).generate()
    original = MUVRMAPPOEnv(net.graph, sfc, n_regions=2)
    tradeoff = MUVRMAPPOEnv(net.graph, sfc, n_regions=2)
    tradeoff.load_balance_scale = scale
    actions = {0: 0, 1: original.max_nodes_per_region}
    original.step(actions)
    tradeoff.step(actions)
    assert original.allocated_nodes["vnf"] == 2
    assert tradeoff.allocated_nodes["vnf"] == 1
    assert tradeoff.total_latency < original.total_latency <= 6
    assert tradeoff.graph.nodes[1]["cpu_used"] == 60
    assert net.graph.nodes[1]["cpu_used"] == 50
    assert all(d["cpu_capacity"] == 100 for _, d in tradeoff.graph.nodes(data=True))


def test_tradeoff_is_forwarded_to_evaluation_without_changing_training():
    config = ExperimentConfig.model_validate(
        {"execution": {"stage": "evaluate"}, "evaluation": {"load_balance_scale": 1.5}}
    )
    command = command_plan(config, ".", "/python")[0]
    assert command[command.index("--load-balance-scale") + 1] == "1.5"
    assert config.training.episodes == 300
    assert SimulationSettings(_env_file=None).mappo_load_balance_scale == 2


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_tradeoff_is_rejected(value):
    with pytest.raises(ValidationError):
        SimulationSettings(_env_file=None, mappo_load_balance_scale=value)
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"evaluation": {"load_balance_scale": value}})


def test_zero_penalty_still_masks_hosts_without_cpu():
    import random

    from muar_sfc.core.scenarios.muvr_reference import (
        MUVRProfile,
        build_network,
        generate_session,
    )

    profile = MUVRProfile()
    net = build_network(profile, 42)
    for _, data in net.graph.nodes(data=True):
        data["cpu_used"] = data["cpu_capacity"]
    batch, _ = generate_session(profile, 0, random.Random(100042))
    env = MUVRMAPPOEnv(net.graph, batch[0])
    env.load_balance_scale = 0
    assert all(not mask.any() for mask in env.get_action_masks().values())
    env.step({region: 0 for region in range(env.n_regions)})
    assert not env.success
    assert not env.allocated_nodes

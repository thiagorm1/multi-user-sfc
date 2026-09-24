import pytest

from muar_sfc.algorithms.mappo import MAPPOAlgorithm
from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.cloud_llm_planner import CloudLLMPlanner, CognitiveGuidance
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.net_v2 import Net2


@pytest.fixture
def sample_network():
    net = Net2()
    # Cria uma topologia simples com 5 servidores
    for i in range(5):
        ntype = "server_cloud" if i == 0 else "server_edge"
        net.add_node(i, ntype, cpu_capacity=100.0, cache_capacity=500.0)

    # Conecta em anel com enlaces bidirecionais
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, v in edges:
        net.add_edge(u, v, bandwidth_capacity=1000.0, latency=2.0)
    return net


@pytest.fixture
def sample_sfc():
    vnf_list = [
        {
            "type": 2, "name": "IA_1", "CPU": 15, "cache": 0,
            "in_bw": 50, "out_bw": 50, "latency": 2.0,
        },
        {
            "type": 2, "name": "MA_1", "CPU": 10, "cache": 100,
            "in_bw": 50, "out_bw": 50, "latency": 2.0,
        },
        {
            "type": 2, "name": "RE_1", "CPU": 20, "cache": 0,
            "in_bw": 50, "out_bw": 50, "latency": 2.0,
        },
    ]
    dependencies = [
        ("src", "IA_1", 50),
        ("IA_1", "MA_1", 50),
        ("MA_1", "RE_1", 50),
        ("RE_1", "dst", 50),
    ]
    sfc_dict = {
        "name": "sfc_test_cog_1",
        "vnf_list": vnf_list,
        "dependencies": dependencies,
        "bandwidth": 50,
        "closer_router": 1,
        "src_node": 1,
        "dst_node": 4,
        "latency": 100,
        "duration": 60,
    }
    return SFCGenerator(sfc_dict).generate()


def test_cognitive_guidance_defaults():
    guidance = CognitiveGuidance()
    assert guidance.avoid_nodes == []
    assert guidance.priority_cache_nodes == []
    assert guidance.latency_weight == 0.5
    assert guidance.load_balance_weight == 0.5
    assert guidance.sync_tolerance_factor == 1.0


def test_telemetry_snapshot_extraction(sample_network):
    # Simula saturação no nó 2
    sample_network.graph.nodes[2]["cpu_used"] = 75.0
    sample_network.graph.nodes[2]["cpu_capacity"] = 100.0

    settings = SimulationSettings(cognitive_planner=True)
    planner = CloudLLMPlanner(settings, substrate_network=sample_network)

    telemetry = planner.extract_telemetry_snapshot()
    assert "net_cpu_util_pct" in telemetry
    assert 2 in telemetry["high_cpu_nodes"]
    assert len(telemetry["cache_ready_nodes"]) > 0


def test_cognitive_heuristic_guidance(sample_network):
    # Simula saturação em nós específicos
    sample_network.graph.nodes[3]["cpu_used"] = 80.0
    sample_network.graph.nodes[3]["cpu_capacity"] = 100.0

    settings = SimulationSettings(cognitive_planner=True, cognitive_llm_model="mock")
    planner = CloudLLMPlanner(settings, substrate_network=sample_network)

    guidance = planner.generate_guidance()
    assert isinstance(guidance, CognitiveGuidance)
    assert 3 in guidance.avoid_nodes
    assert len(guidance.reasoning) > 0
    assert 0.0 <= guidance.latency_weight <= 1.0


def test_action_mask_pruning(sample_network, sample_sfc):
    env = MAPPO_SFC_Env(
        graph=sample_network.graph,
        sfc=sample_sfc,
        n_regions=2,
        cloud_node=0,
        is_training=False,
    )
    env.reset_sfc(sample_sfc)

    # Sem orientação cognitiva
    masks_before = env.get_action_masks()
    assert masks_before[0].sum() > 0

    # Aplica diretriz para evitar o nó 1
    guidance = CognitiveGuidance(avoid_nodes=[1])
    env.set_cognitive_guidance(guidance)
    masks_after = env.get_action_masks()

    # O nó 1 (se presente na região e com alternativas) deve ser podado
    for r in range(env.n_regions):
        nodes_in_r = env.region_nodes[r]
        if 1 in nodes_in_r:
            idx = nodes_in_r.index(1)
            # Se havia alternativas, o índice foi mascarado
            if masks_before[r].sum() > 1:
                assert masks_after[r][idx] == 0.0


def test_dynamic_reward_shaping(sample_network, sample_sfc):
    env = MAPPO_SFC_Env(
        graph=sample_network.graph,
        sfc=sample_sfc,
        n_regions=2,
        cloud_node=0,
        is_training=False,
    )
    env.reset_sfc(sample_sfc)
    env.total_latency = 25.0  # Simula latência acumulada para testar sensibilidade do peso

    # Recompensa neutra
    r_neutral = env._compute_reward(done=False)

    # Alta prioridade em latência
    guidance_lat = CognitiveGuidance(latency_weight=0.9, load_balance_weight=0.1)
    env.set_cognitive_guidance(guidance_lat)
    r_lat_prioritized = env._compute_reward(done=False)

    # Recompensas devem diferir e ser mais severas com maior peso na latência
    assert r_neutral != r_lat_prioritized
    assert r_lat_prioritized < r_neutral


def test_mappo_algorithm_integration(sample_network, sample_sfc):
    alg = MAPPOAlgorithm(n_regions=2, cloud_node=0)
    alg.install_substrate_network(sample_network.graph)
    alg.install_SFC(sample_sfc)

    settings = SimulationSettings(cognitive_planner=True)
    planner = CloudLLMPlanner(settings, substrate_network=sample_network, algorithm=alg)

    guidance = planner.generate_guidance()
    planner.apply_guidance_to_algorithm(alg)

    assert alg.cognitive_guidance == guidance

    # Executa algoritmo com a diretriz cognitiva aplicada
    success = alg.start_algorithm()
    assert success is True
    assert alg.get_latency() > 0.0

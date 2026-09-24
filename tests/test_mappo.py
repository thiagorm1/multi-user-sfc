import networkx as nx
import numpy as np
import pytest
import torch

from muar_sfc.algorithms.instantiator import AlgorithmInstantiator
from muar_sfc.algorithms.mappo import (
    MAPPO_SFC_Env,
    MAPPOAlgorithm,
    MAPPOBuffer,
    MAPPOTrainer,
    RegionalActorNetwork,
    RegionalTopologyPartitioner,
)
from muar_sfc.controllers.sfc_generator import SFCGenerator


@pytest.fixture
def sample_substrate():
    """Gera uma topologia substrata simples com nós de borda e 1 nó de nuvem."""
    graph = nx.Graph()
    # Nó 0: Cloud
    graph.add_node(
        0,
        type="server",
        cpu_capacity=5000,
        cpu_used=0,
        cache_capacity=5000,
        cache_used=0,
        position=(5000, 5000),
    )

    # 8 nós de borda distribuídos geograficamente
    positions = [
        (100, 100), (200, 150),   # Região 0 (Sudoeste)
        (800, 100), (900, 150),   # Região 1 (Sudeste)
        (100, 800), (200, 850),   # Região 2 (Noroeste)
        (800, 800), (900, 850),   # Região 3 (Nordeste)
    ]

    for idx, pos in enumerate(positions, start=1):
        graph.add_node(
            idx,
            type="server",
            cpu_capacity=200,
            cpu_used=0,
            cache_capacity=200,
            cache_used=0,
            ips=1000000000,
            position=pos,
        )

    # Conexões internas e para cloud
    graph.add_edge(0, 1, latency=2.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(0, 3, latency=2.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(0, 5, latency=2.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(0, 7, latency=2.0, bandwidth_capacity=1000, bandwidth_used=0)

    graph.add_edge(1, 2, latency=1.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(3, 4, latency=1.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(5, 6, latency=1.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(7, 8, latency=1.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(2, 4, latency=3.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(6, 8, latency=3.0, bandwidth_capacity=1000, bandwidth_used=0)

    return graph


@pytest.fixture
def sample_dag_sfc():
    """Gera uma SFC em formato DAG com bifurcação e sincronização."""
    vnf_list = [
        {
            "type": 2, "name": "IA_DET_FT_1", "CPU": 10, "cache": 0,
            "in_bw": 150, "out_bw": 150, "latency": 1.0,
        },
        {
            "type": 2, "name": "MA_region_1", "CPU": 20, "cache": 50,
            "in_bw": 80, "out_bw": 80, "latency": 2.0,
        },
        {
            "type": 2, "name": "UNI_p1_1", "CPU": 20, "cache": 0,
            "in_bw": 160, "out_bw": 160, "latency": 2.0,
        },
        {
            "type": 2, "name": "RE_p1_1", "CPU": 15, "cache": 0,
            "in_bw": 240, "out_bw": 240, "latency": 2.0,
        },
        {
            "type": 2, "name": "EC_TC_p1_1", "CPU": 10, "cache": 0,
            "in_bw": 100, "out_bw": 100, "latency": 1.0,
        },
    ]

    dependencies = [
        ("src", "IA_DET_FT_1", 150),
        ("IA_DET_FT_1", "MA_region_1", 80),
        ("IA_DET_FT_1", "UNI_p1_1", 160),
        ("MA_region_1", "RE_p1_1", 80),
        ("UNI_p1_1", "RE_p1_1", 160),
        ("RE_p1_1", "EC_TC_p1_1", 100),
        ("EC_TC_p1_1", "dst", 100),
    ]

    sfc_dict = {
        "name": "sfc_dag_test_mappo",
        "vnf_list": vnf_list,
        "dependencies": dependencies,
        "bandwidth": 100,
        "closer_router": 1,
        "src_node": 0,
        "dst_node": 8,
        "latency": 100,
        "duration": 60,
    }
    return SFCGenerator(sfc_dict).generate()


def test_partitioner(sample_substrate):
    """Valida o particionamento topológico regional."""
    partitioner = RegionalTopologyPartitioner(sample_substrate, n_regions=4, cloud_node=0)
    assert partitioner.get_region_for_node(0) == -1  # Cloud

    # Todos os nós de borda (1 a 8) devem ter regiões atribuídas entre 0 e 3
    for n in range(1, 9):
        reg = partitioner.get_region_for_node(n)
        assert 0 <= reg < 4

    # Verifica se os clusters não estão vazios
    for r in range(4):
        nodes = partitioner.get_nodes_in_region(r)
        assert len(nodes) > 0


def test_actor_masking():
    """Valida que a máscara de ação anula probabilidades de ações proibidas."""
    obs_dim = 16
    action_dim = 5
    actor = RegionalActorNetwork(obs_dim, action_dim)

    obs = torch.randn(1, obs_dim)
    # Permite apenas as ações 1 e 3
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]])

    dist = actor(obs, action_mask=mask)
    probs = dist.probs.detach().numpy()[0]

    assert probs[0] < 1e-6
    assert probs[2] < 1e-6
    assert probs[4] < 1e-6
    assert probs[1] > 0.0
    assert probs[3] > 0.0
    assert np.isclose(probs[1] + probs[3], 1.0)


def test_env_mappo_step(sample_substrate, sample_dag_sfc):
    """Testa reset e passos de decisão no ambiente CTDE."""
    env = MAPPO_SFC_Env(sample_substrate, sample_dag_sfc, n_regions=4, cloud_node=0)
    obs = env.get_observations()
    state = env.get_global_state()

    assert len(obs) == 4
    assert len(state) > 0

    masks = env.get_action_masks()
    assert len(masks) == 4
    actions = {r: 0 for r in range(4)}

    next_obs, next_state, reward, done, info = env.step(actions)
    assert isinstance(reward, float)
    assert "allocated_node" in info


def test_mappo_algorithm_execution(sample_substrate, sample_dag_sfc):
    """Valida a execução completa do MAPPOAlgorithm com geração de rota e latência."""
    alg = MAPPOAlgorithm(n_regions=4, cloud_node=0)
    alg.install_substrate_network(sample_substrate)
    alg.install_SFC(sample_dag_sfc)

    success = alg.start_algorithm()
    assert success is True
    assert alg.get_latency() > 0.0

    route_info = alg.get_route_info()
    assert "src" in route_info
    assert "dst" in route_info
    assert "IA_DET_FT_1" in route_info
    assert "RE_p1_1" in route_info


def test_instantiator_integration():
    """Valida que o instanciador do projeto instancia MAPPOAlgorithm corretamente."""
    instantiator = AlgorithmInstantiator()
    alg = instantiator.instantiate_algorithm("mappo")
    assert isinstance(alg, MAPPOAlgorithm)
    assert alg.name == "MAPPO"


def test_mappo_trainer_rollout(sample_substrate, sample_dag_sfc):
    """Valida um ciclo completo de rollout, GAE e cálculo de gradiente no MAPPOTrainer."""
    env = MAPPO_SFC_Env(sample_substrate, sample_dag_sfc, n_regions=4, cloud_node=0)
    trainer = MAPPOTrainer(
        n_agents=env.n_regions,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim_per_agent,
        state_dim=env.state_dim,
    )
    buffer = MAPPOBuffer(n_agents=env.n_regions)

    obs, state = env.reset_sfc(sample_dag_sfc)
    done = False

    while not done:
        masks = env.get_action_masks()
        actions, action_log_probs = trainer.select_actions(obs, masks, deterministic=False)
        val = trainer.get_value(state)
        next_obs, next_state, reward, done, info = env.step(actions)
        buffer.insert(obs, state, actions, action_log_probs, masks, reward, val, done)
        obs, state = next_obs, next_state

    metrics = trainer.train(buffer)
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics

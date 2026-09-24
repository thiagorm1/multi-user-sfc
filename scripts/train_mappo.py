import argparse
import contextlib

from loguru import logger

from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env
from muar_sfc.algorithms.mappo.mappo_core import MAPPOBuffer, MAPPOTrainer
from muar_sfc.config import ROOT_DIR, SimulationSettings
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.topology.instantiator import TopologyInstantiator


def generate_sample_sfc(sfc_id: int = 1, is_dag: bool = True):
    """Gera uma SFC representativa (DAG ou linear) para treino."""
    vnf_list = [
        {
            "type": 2, "name": "IA_DET_FT_1", "CPU": 10, "cache": 0,
            "in_bw": 150, "out_bw": 150, "latency": 1.0,
        },
        {
            "type": 2, "name": "MA_region_1", "CPU": 20, "cache": 80,
            "in_bw": 80, "out_bw": 80, "latency": 2.0,
        },
        {
            "type": 2, "name": "UNI_p1_1", "CPU": 25, "cache": 0,
            "in_bw": 160, "out_bw": 160, "latency": 2.0,
        },
        {
            "type": 2, "name": "RE_p1_1", "CPU": 20, "cache": 0,
            "in_bw": 240, "out_bw": 240, "latency": 2.5,
        },
        {
            "type": 2, "name": "EC_TC_p1_1", "CPU": 15, "cache": 0,
            "in_bw": 100, "out_bw": 100, "latency": 1.5,
        },
    ]

    if is_dag:
        dependencies = [
            ("src", "IA_DET_FT_1", 150),
            ("IA_DET_FT_1", "MA_region_1", 80),
            ("IA_DET_FT_1", "UNI_p1_1", 160),
            ("MA_region_1", "RE_p1_1", 80),
            ("UNI_p1_1", "RE_p1_1", 160),
            ("RE_p1_1", "EC_TC_p1_1", 100),
            ("EC_TC_p1_1", "dst", 100),
        ]
    else:
        dependencies = [
            ("src", "IA_DET_FT_1", 150),
            ("IA_DET_FT_1", "MA_region_1", 80),
            ("MA_region_1", "UNI_p1_1", 160),
            ("UNI_p1_1", "RE_p1_1", 160),
            ("RE_p1_1", "EC_TC_p1_1", 100),
            ("EC_TC_p1_1", "dst", 100),
        ]

    sfc_dict = {
        "name": f"sfc_mappo_train_{sfc_id}",
        "vnf_list": vnf_list,
        "dependencies": dependencies,
        "bandwidth": 100,
        "closer_router": 1,
        "src_node": 0,
        "dst_node": 5,
        "latency": 50,
        "duration": 60,
    }
    return SFCGenerator(sfc_dict).generate()


def train_mappo(
    episodes: int = 50,
    n_regions: int = 4,
    topology_name: str = "paloalto",
    save_path: str = "rl_saved_models/MAPPO_allocation_model.pt",
    lr_actor: float = 3e-4,
    lr_critic: float = 5e-4,
):
    logger.info(
        f"Iniciando treinamento MAPPO com {n_regions} regiões na topologia '{topology_name}'."
    )

    settings = SimulationSettings()
    topology = TopologyInstantiator().instantiate_topology(topology_name, settings.eco_effi_ratio)
    network = topology.generate_substrate_network()
    with contextlib.suppress(Exception):
        network.set_reliability_params(settings)
    graph = network.graph if hasattr(network, "graph") else network

    # Instancia ambiente base com uma SFC inicial
    sample_sfc = generate_sample_sfc(1, is_dag=True)
    env = MAPPO_SFC_Env(
        graph=graph,
        sfc=sample_sfc,
        n_regions=n_regions,
        cloud_node=0,
        is_training=True,
    )

    trainer = MAPPOTrainer(
        n_agents=env.n_regions,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim_per_agent,
        state_dim=env.state_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
    )

    buffer = MAPPOBuffer(n_agents=env.n_regions)

    all_rewards = []
    successes = 0

    for ep in range(1, episodes + 1):
        # Alterna entre linear e DAG para treinamento generalista
        is_dag = (ep % 2 == 0)
        sfc = generate_sample_sfc(ep, is_dag=is_dag)
        obs, state = env.reset_sfc(sfc)
        done = False
        ep_reward = 0.0

        while not done:
            action_masks = env.get_action_masks()
            actions, action_log_probs = trainer.select_actions(
                obs, action_masks, deterministic=False
            )
            val = trainer.get_value(state)

            next_obs, next_state, reward, done, info = env.step(actions)
            buffer.insert(obs, state, actions, action_log_probs, action_masks, reward, val, done)

            obs = next_obs
            state = next_state
            ep_reward += reward

        if env.success:
            successes += 1

        all_rewards.append(ep_reward)

        # Atualiza a cada 5 episódios ou no final
        if ep % 5 == 0 or ep == episodes:
            metrics = trainer.train(buffer)
            buffer.clear()
            logger.info(
                f"Episódio {ep}/{episodes} | Recompensa: {ep_reward:.2f} | "
                f"Sucesso: {env.success} | Latência: {env.total_latency:.2f} ms | "
                f"Perda Ator: {metrics.get('actor_loss', 0.0):.4f} | "
                f"Perda Crítico: {metrics.get('critic_loss', 0.0):.4f}"
            )

    full_save_path = ROOT_DIR / save_path
    trainer.save(full_save_path)
    logger.success(f"Modelo MAPPO treinado e salvo com sucesso em: {full_save_path}")
    pct_success = (successes / episodes) * 100
    logger.info(f"Taxa de sucesso no treino: {pct_success:.1f}% ({successes}/{episodes})")


def main():
    parser = argparse.ArgumentParser(description="Treinador do MAPPO para Orquestração Cloud-Edge")
    parser.add_argument("--episodes", type=int, default=30, help="Número de episódios")
    parser.add_argument("--n-regions", type=int, default=4, help="Número de regiões de borda")
    parser.add_argument("--topology", type=str, default="paloalto", help="Topologia para treino")
    parser.add_argument(
        "--save-path",
        type=str,
        default="rl_saved_models/MAPPO_allocation_model.pt",
        help="Caminho do checkpoint",
    )
    args = parser.parse_args()

    train_mappo(
        episodes=args.episodes,
        n_regions=args.n_regions,
        topology_name=args.topology,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

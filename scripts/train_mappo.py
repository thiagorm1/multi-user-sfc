"""Train MAPPO on varied, seeded MUAR workloads without changing the existing model."""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import random

import numpy as np
import torch
from loguru import logger

from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env
from muar_sfc.algorithms.mappo.mappo_core import MAPPOBuffer, MAPPOTrainer
from muar_sfc.config import ROOT_DIR, SimulationSettings
from muar_sfc.controllers.modules.mobility_manager import MobilityManager
from muar_sfc.controllers.sfc_queue import SFCQueue
from muar_sfc.core.capacity import apply_capacity_profile
from muar_sfc.core.poisson_emitter import PoissonEmitter
from muar_sfc.core.scenarios.muar import MuarScenario
from muar_sfc.topology.instantiator import TopologyInstantiator


def sampled_environment(base_network, topology, settings, rng, episode):
    network = copy.deepcopy(base_network)
    # Background traffic occupies actual CPU, cache, link and wireless budgets.
    upper_load = float(rng.choice([0.25, 0.60, 0.90, 0.995]))
    for _, data in network.graph.nodes(data=True):
        if data.get("type") == "server":
            data["cpu_used"] = data["cpu_capacity"] * float(rng.uniform(0, upper_load))
            data["cache_used"] = data["cache_capacity"] * float(rng.uniform(0, upper_load))
        elif data.get("type") == "router":
            data["w_channel_used"] = data["w_channel_capacity"] * float(rng.uniform(0, upper_load))
    for _, _, data in network.graph.edges(data=True):
        if np.isfinite(data["bandwidth_capacity"]):
            data["bandwidth_used"] = data["bandwidth_capacity"] * float(rng.uniform(0, upper_load))
    sample_settings = settings.model_copy(
        update={"dag": bool(rng.random() < 0.7), "n_sessions": 2}
    )
    queue = SFCQueue()
    scenario = MuarScenario(sample_settings, queue, topology, PoissonEmitter(10))
    random.seed(int(rng.integers(0, 2**31)))
    np.random.seed(int(rng.integers(0, 2**31)))
    with contextlib.redirect_stdout(io.StringIO()):
        scenario.generate_sfc_session(None)
    batch = queue.peek_sfc()
    sfc = batch[int(rng.integers(0, len(batch)))]
    # Range includes tight latency SLAs; synchronization remains a hard constraint.
    sfc.set_latency_request(float(rng.choice([10.0, 25.0, 50.0, 100.0])))
    mobility = MobilityManager(settings)
    mobility.register_mobile_user_in_network([sfc], network)
    graph = network.graph.copy()
    graph.add_node(sfc.dst_node, **copy.deepcopy(network.md_graph.nodes[sfc.dst_node]))
    router = graph.nodes[sfc.closer_router]
    graph.add_edge(
        sfc.dst_node,
        sfc.closer_router,
        bandwidth_capacity=router["w_channel_capacity"] - router["w_channel_used"],
        bandwidth_used=0.0,
        bandwidth_reserved=0.0,
        latency=1.0,
        services_in_transit={},
    )
    return MAPPO_SFC_Env(
        graph,
        sfc,
        n_regions=settings.n_mappo_regions,
        share=settings.share,
        shareband=settings.shareband,
        is_training=True,
    )


def rollout(env, trainer, buffer=None, deterministic=False):
    obs, state = env.reset_sfc(env.sfc)
    done = False
    total_reward = 0.0
    while not done:
        masks = env.get_action_masks()
        actions, log_probs = trainer.select_actions(obs, masks, deterministic=deterministic)
        value = trainer.get_value(state) if buffer is not None else 0.0
        next_obs, next_state, reward, done, _ = env.step(actions)
        if buffer is not None:
            buffer.insert(obs, state, actions, log_probs, masks, reward, value, done)
        obs, state = next_obs, next_state
        total_reward += reward
    return {
        "success": env.success,
        "reward": total_reward,
        "latency_ms": env.total_latency,
        "sync_ms": env.sync_differential,
    }


def train_mappo(
    episodes=200,
    n_regions=4,
    topology_name="luxembourgv2",
    save_path="rl_saved_models/MAPPO_acceptance_model.pt",
    seed=137,
    warm_start="rl_saved_models/MAPPO_allocation_model.pt",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    settings = SimulationSettings(
        _env_file=None,
        topology=topology_name,
        mobility=False,
        n_mappo_regions=n_regions,
        n_players=4,
        share=True,
        shareband=True,
        cpu_capacity_factor=3.6,
        cache_capacity_factor=3.6,
        link_capacity_factor=8,
        wireless_capacity_factor=4,
        source_bandwidth_mbps=40000,
    )
    topology = TopologyInstantiator().instantiate_topology(topology_name, settings.eco_effi_ratio)
    network = topology.generate_substrate_network()
    apply_capacity_profile(network, settings)
    rng = np.random.default_rng(seed)
    sample = sampled_environment(network, topology, settings, rng, 0)
    trainer = MAPPOTrainer(
        sample.n_regions,
        sample.obs_dim,
        sample.action_dim_per_agent,
        sample.state_dim,
        device="cpu",
    )
    warm_path = ROOT_DIR / warm_start if warm_start else None
    loaded = False
    if warm_path and warm_path.exists():
        checkpoint = torch.load(warm_path, map_location="cpu", weights_only=False)
        if all(
            checkpoint.get(key) == value
            for key, value in {
                "n_agents": sample.n_regions,
                "obs_dim": sample.obs_dim,
                "action_dim": sample.action_dim_per_agent,
                "state_dim": sample.state_dim,
            }.items()
        ):
            trainer.load(warm_path)
            loaded = True
    target = ROOT_DIR / save_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if warm_path and target.resolve() == warm_path.resolve():
        raise ValueError("Use outro caminho para preservar o checkpoint anterior.")
    # Identical held-out environments before and after training; seeds differ from training.
    evaluation_rng = np.random.default_rng(seed + 100000)
    held_out = [
        sampled_environment(network, topology, settings, evaluation_rng, i) for i in range(40)
    ]
    before = [rollout(env, trainer, deterministic=True) for env in held_out]
    buffer = MAPPOBuffer(n_agents=sample.n_regions)
    history = []
    for episode in range(1, episodes + 1):
        env = sampled_environment(network, topology, settings, rng, episode)
        result = rollout(env, trainer, buffer)
        history.append({"episode": episode, **result})
        if episode % 5 == 0 or episode == episodes:
            losses = trainer.train(buffer)
            buffer.clear()
            history[-1].update(losses)
        if episode % 20 == 0 or episode == episodes:
            recent = history[-20:]
            print(
                json.dumps(
                    {
                        "episode": episode,
                        "episodes": episodes,
                        "recent_success_pct": 100
                        * sum(r["success"] for r in recent)
                        / len(recent),
                        "recent_mean_reward": float(np.mean([r["reward"] for r in recent])),
                    }
                ),
                flush=True,
            )
    after = [rollout(env, trainer, deterministic=True) for env in held_out]
    temporary = target.with_suffix(".pt.tmp")
    trainer.save(temporary)
    temporary.replace(target)
    metadata = {
        "seed": seed,
        "episodes": episodes,
        "device": "cpu",
        "settings": settings.model_dump(),
        "warm_start_loaded": loaded,
        "warm_start_sha256": hashlib.sha256(warm_path.read_bytes()).hexdigest()
        if loaded
        else None,
        "checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "held_out_count": len(held_out),
        "held_out_before_pct": 100 * sum(r["success"] for r in before) / len(before),
        "held_out_after_pct": 100 * sum(r["success"] for r in after) / len(after),
        "held_out_before": before,
        "held_out_after": after,
        "history": history,
    }
    target.with_suffix(".training.json").write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {
                "checkpoint": str(target),
                "held_out_before_pct": metadata["held_out_before_pct"],
                "held_out_after_pct": metadata["held_out_after_pct"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--n-regions", type=int, default=4)
    parser.add_argument("--topology", default="luxembourgv2")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--save-path", default="rl_saved_models/MAPPO_acceptance_model.pt")
    parser.add_argument("--warm-start", default="rl_saved_models/MAPPO_allocation_model.pt")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes deve ser positivo")
    logger.remove()
    train_mappo(
        args.episodes, args.n_regions, args.topology, args.save_path, args.seed, args.warm_start
    )


if __name__ == "__main__":
    main()

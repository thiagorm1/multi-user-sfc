"""CPU training of MAPPO for the unscaled 25-server MUVR profile."""

import argparse
import copy
import hashlib
import json
import random
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from muar_sfc.algorithms.mappo.mappo_core import MAPPOBuffer, MAPPOTrainer
from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv, dimensions
from muar_sfc.core.orchestration.placement import is_shareable, reserve_node
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, build_network, generate_session


def sampled_env(seed):
    rng = random.Random(seed)
    net = build_network(MUVRProfile(), seed)
    batch, _ = generate_session(MUVRProfile(), seed, rng)
    sfc = rng.choice(batch)
    # Varied training-only background load; never increase any capacity or SLA.
    maximum = rng.choice([0.0, 0.15, 0.35, 0.60, 0.85])
    # Include correlated bottlenecks, rather than only independent light loads.
    hotspot = rng.choice([None, "cpu", "cache", "bandwidth"])
    hot_nodes = set(rng.sample(list(net.graph), rng.randint(1, 6)))
    for _, data in net.graph.nodes(data=True):
        data["cpu_used"] = rng.uniform(0, maximum) * data["cpu_capacity"]
        data["cache_used"] = rng.uniform(0, maximum) * data["cache_capacity"]
    for node in hot_nodes:
        if hotspot in ("cpu", "cache"):
            data = net.graph.nodes[node]
            data[f"{hotspot}_used"] = rng.uniform(0.8, 1.0) * data[f"{hotspot}_capacity"]
    for u, v, data in net.graph.edges(data=True):
        utilization = (
            rng.uniform(0.8, 1.0)
            if (hotspot == "bandwidth" and (u in hot_nodes or v in hot_nodes))
            else rng.uniform(0, maximum)
        )
        data["bandwidth_used"] = utilization * data["bandwidth_capacity"]
    if rng.random() < 0.4:
        host = rng.choice(list(net.graph))
        for vnf in sfc.vnfs.values():
            if is_shareable(vnf):
                # A hotspot may leave insufficient space for the warm cache.
                with suppress(ValueError):
                    reserve_node(net.graph, sfc, vnf, host, True)
    return MUVRMAPPOEnv(net.graph, sfc)


def rollout(env, trainer, buffer=None, deterministic=False):
    obs, state = env.reset_sfc(env.sfc)
    done = False
    while not done:
        masks = env.get_action_masks()
        actions, log_probs = trainer.select_actions(obs, masks, deterministic=deterministic)
        value = trainer.get_value(state) if buffer is not None else 0
        next_obs, next_state, reward, done, _ = env.step(actions)
        if buffer is not None:
            buffer.insert(obs, state, actions, log_probs, masks, reward, value, done)
        obs, state = next_obs, next_state
    return env.success


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--demonstrations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--output", type=Path, default=Path("rl_saved_models/MAPPO_muvr_model.pt"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=5e-4)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--ppo-epoch", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--cloning-epochs", type=int, default=15)
    parser.add_argument("--cloning-batch-size", type=int, default=128)
    parser.add_argument("--update-every", type=int, default=5)
    args = parser.parse_args(argv)
    if args.episodes < 5 or args.demonstrations < 1:
        parser.error("At least five MAPPO episodes and one demonstration are required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    sample = sampled_env(200000 + args.seed)
    trainer = MAPPOTrainer(
        **dimensions(sample),
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        device="cpu",
    )
    # Selection seeds and final test seeds are disjoint from training and benchmark.
    validation = [sampled_env(800000 + index) for index in range(30)]
    held_out = [sampled_env(900000 + index) for index in range(30)]
    before = sum(rollout(env, trainer, deterministic=True) for env in held_out)
    demonstration_data = {region: [] for region in range(sample.n_regions)}
    for episode in range(args.demonstrations):
        env = sampled_env(200000 + args.seed * 1000 + episode)
        obs, _ = env.reset_sfc(env.sfc)
        done = False
        while not done:
            actions, masks = env.expert_actions()
            for region in range(env.n_regions):
                if masks[region].any():
                    demonstration_data[region].append(
                        (obs[region], masks[region], actions[region])
                    )
            obs, _, _, done, _ = env.step(actions)
        if (episode + 1) % 25 == 0:
            print(json.dumps({"stage": "demonstrations", "episodes": episode + 1}), flush=True)
    # Behavior-cloning warm start, followed by actual on-policy MAPPO/GAE updates.
    cloning_updates = 0
    for region, samples in demonstration_data.items():
        actor = trainer.actors[region]
        optimizer = trainer.actor_optimizers[region]
        observations = torch.tensor(np.asarray([r[0] for r in samples]), dtype=torch.float32)
        masks = torch.tensor(np.asarray([r[1] for r in samples]), dtype=torch.float32)
        targets = torch.tensor([r[2] for r in samples], dtype=torch.long)
        for _ in range(args.cloning_epochs):
            order = torch.randperm(len(samples))
            for ids in order.split(args.cloning_batch_size):
                distribution = actor(observations[ids], action_mask=masks[ids])
                loss = F.nll_loss(distribution.logits, targets[ids])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                optimizer.step()
                cloning_updates += 1
    after_cloning = sum(rollout(env, trainer, deterministic=True) for env in held_out)
    print(
        json.dumps(
            {
                "stage": "cloning_complete",
                "held_out_success": after_cloning,
                "held_out_count": len(held_out),
            }
        ),
        flush=True,
    )
    buffer = MAPPOBuffer(sample.n_regions)
    history = []
    training_steps = 0
    updates = 0
    best_validation = -1
    best_weights = None
    selected_episode = None
    validation_history = []
    for episode in range(args.episodes):
        env = sampled_env(500000 + args.seed * 1000 + episode)
        success = rollout(env, trainer, buffer)
        training_steps += len(env.actor_decisions)
        history.append({"episode": episode + 1, "success": success})
        if (episode + 1) % args.update_every == 0 or episode + 1 == args.episodes:
            history[-1].update(trainer.train(buffer))
            buffer.clear()
            updates += 1
            if (episode + 1) % 50 == 0 or episode + 1 == args.episodes:
                score = sum(rollout(env, trainer, deterministic=True) for env in validation)
                validation_history.append({"episode": episode + 1, "success": score})
                if score > best_validation:
                    best_validation = score
                    selected_episode = episode + 1
                    best_weights = copy.deepcopy(
                        {
                            "actors": [actor.state_dict() for actor in trainer.actors],
                            "critic": trainer.critic.state_dict(),
                            "training_steps": training_steps,
                            "MAPPO_updates": updates,
                        }
                    )
        if (episode + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "stage": "MAPPO",
                        "episodes": episode + 1,
                        "recent_success_percent": 100
                        * sum(r["success"] for r in history[-25:])
                        / len(history[-25:]),
                    }
                ),
                flush=True,
            )
    for actor, weights in zip(trainer.actors, best_weights["actors"], strict=True):
        actor.load_state_dict(weights)
    trainer.critic.load_state_dict(best_weights["critic"])
    after = sum(rollout(env, trainer, deterministic=True) for env in held_out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    trainer.save(temporary)
    checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
    checkpoint.update(
        {
            "training_steps": best_weights["training_steps"],
            "MAPPO_updates": best_weights["MAPPO_updates"],
            "profile": "MUVR_25_unscaled_SLA6",
            "training_seed": args.seed,
            "behavior_cloning_updates": cloning_updates,
            "policy_candidates_only": True,
            "selected_episode": selected_episode,
            "training_distribution": "mixed_load_correlated_bottlenecks_v2",
        }
    )
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    metadata = {
        "dimensions": dimensions(sample),
        "seed": args.seed,
        "episodes": args.episodes,
        "demonstration_episodes": args.demonstrations,
        "cloning_updates": cloning_updates,
        "training_steps": training_steps,
        "MAPPO_updates": updates,
        "training_seeds_exclude_benchmark": [42, 43, 44],
        "held_out_single_player_count": len(held_out),
        "held_out_before_percent": 100 * before / len(held_out),
        "held_out_after_cloning_percent": 100 * after_cloning / len(held_out),
        "held_out_after_MAPPO_percent": 100 * after / len(held_out),
        "validation_count": len(validation),
        "validation_history": validation_history,
        "selected_episode": selected_episode,
        "selected_checkpoint_training_steps": best_weights["training_steps"],
        "selected_checkpoint_MAPPO_updates": best_weights["MAPPO_updates"],
        "validation_best_percent": 100 * best_validation / len(validation),
        "training_distribution": "mixed_load_correlated_bottlenecks_v2",
        "checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "history": history,
        "device": "cpu",
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "output"},
        "unscaled": True,
        "sla_ms": 6,
    }
    args.output.with_suffix(".training.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()

"""Train mobile MAPPO on complete four-player sessions from validated SUMO replay."""

import argparse
import copy
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from muar_sfc.algorithms.mappo.mappo_core import MAPPOBuffer, MAPPOTrainer
from muar_sfc.algorithms.mappo.mobile import (
    MOBILE_DIMENSIONS,
    OBSERVATION_SCHEMA,
    OBSERVATION_SCHEMA_SHA256,
    POLICY_SCHEMA,
    REWARD_SPEC,
    rollout_mobile_session,
)
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile
from muar_sfc.core.scenarios.sumo_replay import TrajectoryReplay, load_dataset


def evaluate(trainer, traces, load_balance_scale):
    summaries = []
    for trace in traces:
        row = rollout_mobile_session(
            MUVRProfile(**trace["profile"]),
            trace["seed"],
            TrajectoryReplay(trace),
            trainer,
            deterministic=True,
            load_balance_scale=load_balance_scale,
        )
        summaries.append(
            {
                "seed": trace["seed"],
                "accepted": row["accepted"],
                "interruptions": row["mobility"]["sessions_interrupted"],
                "unserved_session_seconds": row["mobility"]["unserved_session_seconds"],
                "return": row["mobile_training"]["episode_return"],
            }
        )
    return {
        "mean_return": sum(row["return"] for row in summaries) / len(summaries),
        "sessions": summaries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--ppo-epoch", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr-actor", type=float, default=0.0001)
    parser.add_argument("--lr-critic", type=float, default=0.0005)
    parser.add_argument("--load-balance-scale", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, default=Path("rl_saved_models/mappo_sumo"))
    parser.add_argument(
        "--check", action="store_true", help="Validate dataset only; no optimizer."
    )
    args = parser.parse_args(argv)
    if min(args.epochs, args.threads, args.update_every, args.ppo_epoch) <= 0:
        parser.error("Epochs, threads and update intervals must be positive")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("Gamma and GAE lambda must be in [0,1]")
    manifest, splits = load_dataset(args.dataset)
    if args.check:
        print(
            json.dumps(
                {
                    "check": "passed",
                    "policy_schema": POLICY_SCHEMA,
                    "dimensions": MOBILE_DIMENSIONS,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "seed_splits": manifest["seed_splits"],
                }
            )
        )
        return
    directory = args.output_root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    trainer = MAPPOTrainer(
        **MOBILE_DIMENSIONS,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        ppo_epoch=args.ppo_epoch,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    buffer = MAPPOBuffer(4)
    history, steps, updates, episodes = [], 0, 0, 0
    best, best_score = None, -float("inf")
    shuffle_rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        traces = list(splits["train"])
        shuffle_rng.shuffle(traces)
        for trace in traces:
            row = rollout_mobile_session(
                MUVRProfile(**trace["profile"]),
                trace["seed"],
                TrajectoryReplay(trace),
                trainer,
                buffer,
                load_balance_scale=args.load_balance_scale,
            )
            episodes += 1
            steps += row["mobile_training"]["decisions"]
            record = {
                "epoch": epoch + 1,
                "episode": episodes,
                "seed": trace["seed"],
                "accepted": row["accepted"],
                "training": row["mobile_training"],
                "interruptions": row["mobility"]["sessions_interrupted"],
            }
            # Every rollout is terminal; no truncated bootstrap or cross-session GAE.
            if episodes % args.update_every == 0 or trace is traces[-1]:
                record["loss"] = trainer.train(buffer)
                buffer.clear()
                updates += 1
            history.append(record)
        validation = evaluate(trainer, splits["validation"], args.load_balance_scale)
        if validation["mean_return"] > best_score:
            best_score = validation["mean_return"]
            best = {
                "actors": copy.deepcopy(trainer.actors.state_dict()),
                "critic": copy.deepcopy(trainer.critic.state_dict()),
                "epoch": epoch + 1,
                "steps": steps,
                "updates": updates,
                "validation": validation,
            }
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "training_steps": steps,
                    "MAPPO_updates": updates,
                    "validation": validation,
                }
            ),
            flush=True,
        )
    trainer.actors.load_state_dict(best["actors"])
    trainer.critic.load_state_dict(best["critic"])
    # Test is consulted once, after selection using validation alone.
    held_out = evaluate(trainer, splits["test"], args.load_balance_scale)
    checkpoint_path = directory / "model.pt"
    trainer.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.update(
        {
            "policy_schema": POLICY_SCHEMA,
            "observation_schema_sha256": OBSERVATION_SCHEMA_SHA256,
            "observation_schema": OBSERVATION_SCHEMA,
            "training_steps": best["steps"],
            "MAPPO_updates": best["updates"],
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "training_distribution": "one_session_four_players_real_SUMO_replay",
            "reward_spec": REWARD_SPEC,
            "profile": manifest["profile"],
            "seed_splits": manifest["seed_splits"],
            "selected_epoch": best["epoch"],
            "policy_candidates_only": True,
            "search_attempts": 1,
            "load_balance_scale": args.load_balance_scale,
        }
    )
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        "policy_schema": POLICY_SCHEMA,
        "dimensions": MOBILE_DIMENSIONS,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_schema_sha256": OBSERVATION_SCHEMA_SHA256,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest": manifest,
        "hyperparameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "training_steps": steps,
        "MAPPO_updates": updates,
        "selected_training_steps": best["steps"],
        "selected_MAPPO_updates": best["updates"],
        "selected_epoch": best["epoch"],
        "validation": best["validation"],
        "held_out_test": held_out,
        "reward_spec": REWARD_SPEC,
        "history": history,
        "LLM_calls": 0,
        "SUMO_calls_during_training": 0,
    }
    with checkpoint_path.with_suffix(".training.json").open("x") as output:
        json.dump(metadata, output, indent=2, allow_nan=False)
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

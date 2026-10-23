import random

import numpy as np
import pytest
import torch

from muar_sfc.algorithms.mappo.mappo_core import MAPPOBuffer, MAPPOTrainer
from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv, dimensions
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, build_network, generate_session


def test_static_policy_keeps_checkpoint_dimensions_when_mobility_support_is_added():
    profile = MUVRProfile()
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(42))
    env = MUVRMAPPOEnv(net.graph, batch[0])
    assert dimensions(env) == dict(n_agents=4, obs_dim=48, action_dim=7, state_dim=196)
    assert all(obs.shape == (48,) for obs in env.get_observations().values())
    assert env.get_global_state().shape == (196,)


def test_terminal_reward_reaches_previous_decisions_without_crossing_sessions(monkeypatch):
    torch.manual_seed(123)
    trainer = MAPPOTrainer(n_agents=2, obs_dim=3, action_dim=2, state_dim=6,
                           gamma=0.9, gae_lambda=1.0, ppo_epoch=1)
    buffer = MAPPOBuffer(n_agents=2)
    # First session: admission, handover, final continuity reward.
    # Next session must not contribute its reward to the preceding trajectory.
    for index, (reward, done) in enumerate([(0, False), (0, False), (10, True), (100, True)]):
        obs = {agent: np.array([index, agent, 1], dtype=np.float32) for agent in range(2)}
        masks = {agent: np.ones(2, dtype=np.float32) for agent in range(2)}
        actions, log_probs = trainer.select_actions(obs, masks)
        state = np.concatenate(list(obs.values()))
        buffer.insert(obs, state, actions, log_probs, masks, reward, 0.0, done)
    original_loss = torch.nn.functional.mse_loss
    observed_returns = []

    def capture_loss(predictions, targets, *args, **kwargs):
        observed_returns.append(targets.detach().cpu().numpy().copy())
        return original_loss(predictions, targets, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "mse_loss", capture_loss)
    before = [parameter.detach().clone() for parameter in trainer.critic.parameters()]
    metrics = trainer.train(buffer)
    assert observed_returns[0] == pytest.approx([8.1, 9, 10, 100])
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(old, new)
               for old, new in zip(before, trainer.critic.parameters(), strict=True))

import random

import numpy as np
import pytest

from muar_sfc.algorithms.mappo.mappo_core import MAPPOBuffer
from muar_sfc.algorithms.mappo.mobile import (
    MOBILE_DIMENSIONS,
    MobileMUVRMAPPOEnv,
    causal_context,
    rollout_mobile_session,
    session_terminal_reward,
)
from muar_sfc.algorithms.mappo.muvr import dimensions
from muar_sfc.core.scenarios import muvr_reference as scenario


class Trajectory:
    step_s = 1.0

    def __init__(self):
        self.time = 0
        self.features = np.zeros((4, 12), dtype=np.float32)
        self.retired = []

    def advance(self, at):
        self.time = at
        self.features[:, 0] = at / 120

    def destinations(self, index):
        return [1 if self.time < 1 else 2] * 4

    def observation(self, index):
        return dict(time_s=self.time, features=self.features)

    def retire(self, index):
        self.retired.append(index)

    def evidence(self):
        return dict(mode="deterministic_test_trajectory")


class GlobalHostPolicy:
    def select_actions(self, obs, masks, deterministic=False):
        actions = {agent: 6 if mask[6] else int(np.flatnonzero(mask)[0])
                   for agent, mask in masks.items()}
        return actions, dict.fromkeys(actions, 0.0)

    def get_value(self, state):
        return 0.0


def test_mobile_schema_observations_keep_all_players_causal_context_isolated():
    profile = scenario.MUVRProfile()
    net = scenario.build_network(profile, 42)
    batch, _ = scenario.generate_session(profile, 0, random.Random(42))
    raw = np.arange(48, dtype=np.float32).reshape(4, 12)
    context = causal_context(dict(features=raw, time_s=0), "handover")
    expected = context["features"].copy()
    raw[:] = -99
    env = MobileMUVRMAPPOEnv(net.graph, batch[0], context, 2)
    context["features"][:] = -88
    obs = env.get_observations()
    assert dimensions(env) == MOBILE_DIMENSIONS
    assert all(value.shape == (60,) for value in obs.values())
    for value in obs.values():
        np.testing.assert_array_equal(value[-12:], expected[2])
    state = env.get_global_state(obs)
    assert state.shape == (244,)
    np.testing.assert_array_equal(state[-48:], expected.ravel())
    assert expected[:, 10].tolist() == [1] * 4


def test_complete_session_and_handover_share_one_temporal_episode(monkeypatch):
    original = scenario.generate_session

    def fixed_lifetime(*args):
        batch, metadata = original(*args)
        for sfc in batch:
            sfc.duration = 3.5
        metadata["lifetime_s"] = 3.5
        return batch, metadata

    monkeypatch.setattr(scenario, "generate_session", fixed_lifetime)
    mobility = Trajectory()
    buffer = MAPPOBuffer(4)
    row = rollout_mobile_session(scenario.MUVRProfile(requests=1, sla=12), 42,
                                 mobility, GlobalHostPolicy(), buffer)
    assert row["accepted"] == 1
    assert row["mobility"]["handovers_committed"] == 1
    assert mobility.retired == [0]
    events = row["mobile_training"]["decision_events"]
    assert [event["phase"] for event in events] == ["admission", "handover"]
    assert len(buffer) >= 72  # Nine SF decisions for each of four players, twice.
    assert sum(buffer.dones) == 1 and buffer.dones[-1]
    assert all(not done for done in buffer.dones[:-1])
    assert buffer.rewards[-1] == pytest.approx(98.999)
    assert row["mobile_training"]["terminal_reward"] == 100
    # Source feature mutations on later ticks cannot overwrite admission samples.
    assert all(obs[-12] == 0 for obs in buffer.obs[0].values())
    assert all(obs[-2] == 0 for obs in buffer.obs[0].values())
    assert all(obs[-2] == 1 for obs in buffer.obs[events[1]["first_buffer_index"]].values())


def test_interruption_reward_penalizes_early_lost_service_and_rejects_multi_session():
    def row(unserved, interrupted):
        return dict(requests=1, accepted=1, records=[dict(lifetime_s=100)],
                    mobility=dict(unserved_session_seconds=unserved,
                                  sessions_interrupted=interrupted))
    assert session_terminal_reward(row(0, 0)) == 100
    assert session_terminal_reward(row(75, 1)) == -75
    assert session_terminal_reward(row(25, 1)) == -25
    assert session_terminal_reward(dict(requests=1, accepted=0)) == -100
    with pytest.raises(ValueError, match="exactly one"):
        session_terminal_reward(dict(requests=2))

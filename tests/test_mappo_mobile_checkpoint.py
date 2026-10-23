import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from muar_sfc.algorithms.mappo.mobile import (
    MOBILE_DIMENSIONS,
    OBSERVATION_SCHEMA_SHA256,
    MobileMUVRMAPPOEnv,
    MobileSessionAdmitter,
)
from muar_sfc.algorithms.mappo.muvr import TrainedMUVRMAPPO
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, build_network, generate_session


@pytest.mark.parametrize("overrides, message", [
    ({"observation_schema_sha256": "wrong"}, "checksum"),
    ({"n_agents": 3}, "four actors"),
    ({"action_dim": 8}, "seven actions"),
    ({"obs_dim": 48}, "dimensions"),
    ({"MAPPO_updates": 0}, "evidence"),
    ({"training_steps": 0}, "evidence"),
    ({"policy_schema": "muvr_static_v1"}, "schema"),
])
def test_incompatible_mobile_checkpoint_rejected_before_weights(tmp_path, overrides, message):
    checkpoint = dict(**MOBILE_DIMENSIONS, policy_schema="muvr_sumo_v1",
                      observation_schema_sha256=OBSERVATION_SCHEMA_SHA256,
                      training_steps=1, MAPPO_updates=1)
    checkpoint.update(overrides)
    model = tmp_path / "incompatible.pt"
    torch.save(checkpoint, model)
    settings = SimpleNamespace(mappo_model_path=model, mappo_search_attempts=1)
    with pytest.raises(ValueError, match=message):
        TrainedMUVRMAPPO(settings, policy_schema="muvr_sumo_v1")


def test_training_and_evaluation_build_identical_mobile_features():
    profile = MUVRProfile(requests=1)
    graph = build_network(profile, 42).graph
    batch, _ = generate_session(profile, 0, random.Random(42))
    context = dict(time_s=5, features=np.arange(48, dtype=np.float32).reshape(4, 12))
    # Evaluation uses the same causal context setter as the live simulator.
    evaluation = TrainedMUVRMAPPO.__new__(TrainedMUVRMAPPO)
    evaluation.policy_schema = "muvr_sumo_v1"
    evaluation.set_mobility_context(context, "handover")
    training = MobileSessionAdmitter(trainer=None)
    training.set_mobility_context(context, "handover")
    context["features"][:] = -999  # Both paths must own an independent current snapshot.
    for player in range(4):
        train_env = MobileMUVRMAPPOEnv(graph, batch[player], training.context, player)
        eval_env = MobileMUVRMAPPOEnv(graph, batch[player], evaluation.mobility_context, player)
        for region, observation in train_env.get_observations().items():
            np.testing.assert_array_equal(observation, eval_env.get_observations()[region])
        np.testing.assert_array_equal(train_env.get_global_state(), eval_env.get_global_state())

"""Versioned causal mobile MAPPO and complete four-player session rollouts."""

import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np

from muar_sfc.algorithms.mappo.mappo_core import MAPPOBuffer
from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv, dimensions
from muar_sfc.core.scenarios.muvr_reference import simulate

POLICY_SCHEMA = "muvr_sumo_v1"
FEATURE_NAMES = [
    "x_normalized",
    "y_normalized",
    "speed_over_60_mps",
    "heading_sin",
    "heading_cos",
    "last_dx_normalized",
    "last_dy_normalized",
    "access_over_25",
    "previous_access_over_25",
    "time_since_access_change_over_120",
    "handover_phase",
    "elapsed_over_120",
]
MOBILE_DIMENSIONS = {"n_agents": 4, "obs_dim": 60, "action_dim": 7, "state_dim": 244}
REWARD_SPEC = {
    "served_fraction": 100.0,
    "rejection": -100.0,
    "interruption": -100.0,
    "handover_attempt": -1.0,
    "placement_step": -0.001,
    "discount_clock": "actor decision, not virtual seconds",
}

OBSERVATION_SCHEMA = {
    "id": POLICY_SCHEMA,
    "features": FEATURE_NAMES,
    "dimensions": MOBILE_DIMENSIONS,
    "actor": "static48 plus current-player causal12",
    "critic": "static196 plus four players causal12 in player order",
    "normalization": "anchor extrema x/y, speed60m/s, access25, time120s",
}
OBSERVATION_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(OBSERVATION_SCHEMA, sort_keys=True).encode()
).hexdigest()


def causal_context(context, phase):
    features = np.asarray(context["features"], dtype=np.float32).copy()
    if features.shape != (4, 12) or not np.isfinite(features).all():
        raise ValueError("Mobile observation must contain four finite 12-feature vectors")
    if phase not in ("admission", "handover"):
        raise ValueError("Unknown mobile decision phase")
    features[:, 10] = float(phase == "handover")
    return {"features": features, "time_s": float(context["time_s"]), "phase": phase}


class MobileMUVRMAPPOEnv(MUVRMAPPOEnv):
    def __init__(self, graph, sfc, context, player_index, n_regions=4, guidance=None):
        self.mobile_features = np.asarray(context["features"], dtype=np.float32).copy()
        if self.mobile_features.shape != (4, 12) or not np.isfinite(self.mobile_features).all():
            raise ValueError("Invalid causal mobile features")
        self.player_index = player_index
        super().__init__(graph, sfc, n_regions, guidance)
        self.obs_dim += 12
        self.state_dim += 48
        if dimensions(self) != MOBILE_DIMENSIONS:
            raise ValueError("Mobile schema requires the finite 25-host four-region profile")

    def get_observations(self):
        original = super().get_observations()
        return {
            region: np.concatenate([obs, self.mobile_features[self.player_index]]).copy()
            for region, obs in original.items()
        }

    def get_global_state(self, obs=None):
        if obs is None:
            obs = self.get_observations()
        physical = {region: observation[:-12] for region, observation in obs.items()}
        return np.concatenate([super().get_global_state(physical), self.mobile_features.ravel()])


class MobileSessionAdmitter:
    """One on-policy proposal per SF; no retry/off-policy samples in the rollout."""

    requires_mobility_context = True

    def __init__(self, trainer, buffer=None, deterministic=False, load_balance_scale=0.0):
        self.trainer = trainer
        self.buffer = buffer
        self.deterministic = deterministic
        self.load_balance_scale = load_balance_scale
        self.context = None
        self.decisions = 0
        self.decision_events = []

    def set_mobility_context(self, context, phase):
        self.context = causal_context(context, phase)

    def __call__(self, net, batch, strategy):
        if self.context is None or len(batch) != 4:
            raise ValueError("Temporal mobile rollouts require a complete four-player session")
        trial = copy.deepcopy(net)
        latencies = []
        first = len(self.buffer) if self.buffer is not None else 0
        for player, original in enumerate(batch):
            sfc = copy.deepcopy(original)
            env = MobileMUVRMAPPOEnv(trial.graph, sfc, self.context, player)
            env.load_balance_scale = self.load_balance_scale
            obs, state = env.get_observations(), env.get_global_state()
            done = False
            while not done:
                masks = env.get_action_masks()
                actions, log_probs = self.trainer.select_actions(obs, masks, self.deterministic)
                value = self.trainer.get_value(state) if self.buffer is not None else 0.0
                next_obs, next_state, _, done, _ = env.step(actions)
                if self.buffer is not None:
                    # Physical chain completion is not a temporal episode boundary.
                    self.buffer.insert(
                        copy.deepcopy(obs),
                        state.copy(),
                        dict(actions),
                        dict(log_probs),
                        copy.deepcopy(masks),
                        REWARD_SPEC["placement_step"],
                        value,
                        False,
                    )
                self.decisions += 1
                obs, state = next_obs, next_state
            if not env.success:
                self._record_event(first, False)
                return None, [], "mobile_mappo_no_feasible_proposal"
            sfc.edge_routes, sfc.edge_latencies = env.edge_routes, env.edge_latencies
            sfc.comp_latencies = env.comp_latencies
            trial.deploy_sfc(sfc, env.route_info)
            latencies.append(env.total_latency)
        self._record_event(first, True)
        return trial, latencies, "mobile_mappo_on_policy"

    def _record_event(self, first, success):
        if self.buffer is not None and self.context["phase"] == "handover":
            self.buffer.rewards[-1] += REWARD_SPEC["handover_attempt"]
        self.decision_events.append(
            {
                "time_s": self.context["time_s"],
                "phase": self.context["phase"],
                "success": success,
                "first_buffer_index": first,
                "last_buffer_index": len(self.buffer) - 1 if self.buffer is not None else None,
            }
        )


def session_terminal_reward(row):
    if row["requests"] != 1:
        raise ValueError("A temporal training episode must contain exactly one session")
    if not row["accepted"]:
        return REWARD_SPEC["rejection"]
    lifetime = row["records"][0]["lifetime_s"]
    unserved = row["mobility"]["unserved_session_seconds"]
    fraction = max(0.0, 1.0 - unserved / max(lifetime, 1e-9))
    return (
        REWARD_SPEC["served_fraction"] * fraction
        + REWARD_SPEC["interruption"] * row["mobility"]["sessions_interrupted"]
    )


def rollout_mobile_session(
    profile, seed, mobility, trainer, buffer=None, deterministic=False, load_balance_scale=0.0
):
    if profile.requests != 1 or profile.players != 4:
        raise ValueError("Train one complete four-player session per temporal episode")
    local_buffer = MAPPOBuffer(4) if buffer is None else buffer
    start = len(local_buffer)
    adapter = MobileSessionAdmitter(trainer, local_buffer, deterministic, load_balance_scale)
    row = simulate(profile, seed, "mappo_mobile", admitter=adapter, mobility=mobility)
    if len(local_buffer) == start:
        raise ValueError("Mobile episode contains no actor decisions")
    terminal_reward = session_terminal_reward(row)
    local_buffer.rewards[-1] += terminal_reward
    local_buffer.dones[-1] = True
    row["mobile_training"] = {
        "policy_schema": POLICY_SCHEMA,
        "profile": asdict(profile),
        "terminal_reward": terminal_reward,
        "episode_return": sum(local_buffer.rewards[start:]),
        "decisions": adapter.decisions,
        "decision_events": adapter.decision_events,
    }
    return row

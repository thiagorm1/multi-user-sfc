from types import SimpleNamespace

import networkx as nx
import numpy as np

from muar_sfc.algorithms.mappo import muvr
from scripts.train_mappo_muvr import sampled_env


def test_search_changes_failed_branch_and_preserves_prefix(monkeypatch):
    class BranchEnv:
        def __init__(self, graph, sfc, *args):
            self.graph = graph
            self.allocated_nodes = {}
            self.actor_decisions = []
            self.success = False
            self.fail_reason = None
            self.excluded_nodes = {}

        def reset_sfc(self, sfc):
            return {}, None

        def get_observations(self):
            return {}

        def get_action_masks(self):
            name = "abc"[len(self.allocated_nodes)]
            feasible = [node not in self.excluded_nodes.get(name, set()) for node in self.graph]
            if name == "c" and (
                self.allocated_nodes["b"] == 0 or self.excluded_nodes.get("a") != {1, 2}
            ):
                feasible = [False] * len(self.graph)
            return {0: np.array(feasible, dtype=np.float32)}

        def step(self, actions):
            name = "abc"[len(self.allocated_nodes)]
            selected = actions[0] if self.get_action_masks()[0][actions[0]] else None
            self.actor_decisions.append({"vnf": name, "selected_host": selected})
            if selected is None:
                self.fail_reason = "blocked continuation"
                return {}, None, -30, True, {}
            self.allocated_nodes[name] = selected
            self.success = name == "c"
            return {}, None, 0, self.success, {}

    class MaskedActor:
        def select_actions(self, obs, masks, deterministic):
            valid = np.flatnonzero(masks[0])
            return {0: int(valid[0]) if len(valid) else 0}, {0: 0.0}

    monkeypatch.setattr(muvr, "MUVRMAPPOEnv", BranchEnv)
    monkeypatch.setattr(muvr, "dimensions", lambda env: {})
    adapter = muvr.TrainedMUVRMAPPO.__new__(muvr.TrainedMUVRMAPPO)
    adapter.settings = SimpleNamespace(mappo_search_attempts=16, n_mappo_regions=1)
    adapter.guidance = None
    adapter.checkpoint_dimensions = {}
    adapter.trainer = MaskedActor()
    graph = nx.empty_graph(3)
    env, decisions = adapter._search_policy(graph, object())
    assert env.success
    assert env.allocated_nodes == {"a": 0, "b": 1, "c": 0}
    assert len(env.search_summary) <= 16
    assert env.search_summary[-1]["strategy"] == "fixed_prefix"
    assert env.search_summary[0]["no_feasible_masked_action"]
    assert env.search_summary[0]["failed_vnf"] == "c"
    prefix_decisions = [r for r in decisions if r["search_attempt"] > 8]
    assert [r["selected_host"] for r in prefix_decisions if r["vnf"] == "a"] == [0]
    adapter.settings.mappo_search_attempts = 1
    limited, _ = adapter._search_policy(graph, object())
    assert not limited.success
    assert len(limited.search_summary) == 1


def test_training_load_is_reproducible_and_within_capacity():
    for seed in (900000, 900001, 900002, 900003):
        first, second = sampled_env(seed), sampled_env(seed)
        assert muvr.dimensions(first) == muvr.dimensions(second)
        for node, data in first.graph.nodes(data=True):
            for resource in ("cpu", "cache"):
                assert 0 <= data[f"{resource}_used"] <= data[f"{resource}_capacity"]
                assert data[f"{resource}_used"] == second.graph.nodes[node][f"{resource}_used"]
        for _, _, data in first.graph.edges(data=True):
            assert 0 <= data["bandwidth_used"] <= data["bandwidth_capacity"]

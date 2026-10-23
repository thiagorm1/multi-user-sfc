"""Validator regressions for the TopoRAG adaptation.

Local fixtures only: no trained checkpoint, no live LLM, no evaluation campaign.
These tests complement tests/test_toporag.py with checks that were missing there:
canonical signature separation on arbitrary DAGs, the real ``_search_policy``
wiring under the inert control, propagation of a non-zero lagged kappa into the
environment, and the replay contract with a TopoRAG-augmented source.
"""

import copy
import random

import networkx as nx

import muar_sfc.algorithms.mappo.muvr as muvr_module
from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv as RealMUVRMAPPOEnv
from muar_sfc.algorithms.mappo.muvr import TrainedMUVRMAPPO, dimensions
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.cloud_llm_planner import CognitiveGuidance
from muar_sfc.controllers.toporag import TopoRAGMemory, canonical_dag, structural_signature
from muar_sfc.core.orchestration.handover import fingerprint
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, build_network, generate_session
from scripts import benchmark_load_tradeoff as tradeoff


def random_dag(rng, size):
    dag = nx.DiGraph()
    for index in range(size):
        dag.add_node(index, sf=rng.choice(["IA", "BE", "FI"]))
    order = list(range(size))
    rng.shuffle(order)
    for i in range(size):
        for j in range(i + 1, size):
            if rng.random() < 0.4:
                dag.add_edge(order[i], order[j])
    return dag


def test_canonical_dag_is_isomorphism_invariant_and_separating():
    rng = random.Random(20261021)
    pool = [random_dag(rng, rng.randint(3, 5)) for _ in range(40)]
    signatures = [canonical_dag(dag) for dag in pool]
    for dag, signature in zip(pool, signatures, strict=True):
        renamed = nx.relabel_nodes(dag, {n: f"node_{i}" for i, n in enumerate(dag)})
        assert canonical_dag(renamed) == signature
        shuffled_nodes = list(renamed)
        rng.shuffle(shuffled_nodes)
        rebuilt = nx.DiGraph()
        for node in shuffled_nodes:
            rebuilt.add_node(node, **renamed.nodes[node])
        shuffled_edges = list(renamed.edges)
        rng.shuffle(shuffled_edges)
        rebuilt.add_edges_from(shuffled_edges)
        assert canonical_dag(rebuilt) == signature
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            isomorphic = nx.is_isomorphic(
                pool[i],
                pool[j],
                node_match=nx.algorithms.isomorphism.categorical_node_match("sf", None),
            )
            assert (signatures[i] == signatures[j]) is isomorphic


def test_canonical_dag_returns_a_valid_permutation_encoding():
    rng = random.Random(7)
    for _ in range(20):
        dag = random_dag(rng, rng.randint(3, 5))
        signature = canonical_dag(dag)
        assert signature == canonical_dag(dag)
        labels, bits = signature
        assert sorted(labels) == sorted(dag.nodes[node]["sf"] for node in dag)
        assert len(bits) == dag.number_of_nodes() ** 2
        assert sum(bits) == dag.number_of_edges()


def test_signature_ignores_player_identity_and_does_not_touch_the_network():
    profile = MUVRProfile(players=2)
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    before = fingerprint(net)
    signatures = [structural_signature(sfc, net.graph) for sfc in batch]
    assert signatures[0] == signatures[1]
    assert fingerprint(net) == before
    assert signatures[0] == structural_signature(batch[0], net.graph)


class FirstFeasible:
    """Deterministic stand-in for trained actors; this is not a policy claim."""

    n_agents = 4

    def select_actions(self, obs, masks, deterministic=False):
        return {region: int(mask.argmax()) for region, mask in masks.items()}, {
            region: 0.0 for region in masks
        }


def synthetic_adapter(settings, checkpoint_dimensions):
    adapter = TrainedMUVRMAPPO.__new__(TrainedMUVRMAPPO)
    adapter.settings = settings
    adapter.qwen = False
    adapter.guidance = None
    adapter.pending = adapter.guidance_ready_at = None
    adapter.planning_records, adapter.request_evidence, adapter.call_results = [], [], []
    adapter.guidance_applied_requests = 0
    adapter.guidance_misses = 0
    adapter.background_skipped_busy = 0
    adapter.toporag_memory = TopoRAGMemory()
    adapter.toporag_decisions = []
    adapter._toporag_prior = {}
    adapter._toporag_kappa = 0.0
    adapter._previous_cycle_success = 0.0
    adapter.trainer = FirstFeasible()
    adapter.checkpoint_dimensions = checkpoint_dimensions
    return adapter


def session_fixture():
    profile = MUVRProfile()
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    for sfc in batch:
        sfc.arrival_time = 0
    probe, _ = generate_session(profile, 0, random.Random(100042))
    checkpoint_dimensions = dimensions(
        RealMUVRMAPPOEnv(build_network(profile, 42).graph, probe[0], 4)
    )
    return net, batch, checkpoint_dimensions


def test_real_search_policy_inert_control_is_identical_and_never_leaves_proposals():
    net, batch, checkpoint_dimensions = session_fixture()
    control = synthetic_adapter(
        SimulationSettings(_env_file=None, toporag_enabled=False), checkpoint_dimensions
    )
    inert = synthetic_adapter(
        SimulationSettings(_env_file=None, toporag_enabled=True, toporag_kappa=0.0),
        checkpoint_dimensions,
    )
    control_net, inert_net = copy.deepcopy(net), copy.deepcopy(net)
    plain, plain_latencies, plain_reason = control(control_net, batch, "audit")
    trial, latencies, reason = inert(inert_net, batch, "audit")
    assert reason == plain_reason and latencies == plain_latencies
    assert (plain is None) == (trial is None)
    if trial is not None:
        assert fingerprint(trial) == fingerprint(plain)
    assert fingerprint(control_net) == fingerprint(inert_net) == fingerprint(net)
    evidence = copy.deepcopy(inert.request_evidence[-1])
    evidence.pop("toporag")
    assert evidence == control.request_evidence[-1]
    assert all(
        decision["selected_host"] is None
        or decision["selected_host"] in decision["proposals"].values()
        for request in inert.request_evidence
        for decision in request["actor_decisions"]
    )
    assert len(inert.toporag_memory) == len(inert.request_evidence[-1]["toporag"])


def test_lagged_kappa_reaches_the_environment_only_after_a_successful_cycle(monkeypatch):
    net, batch, checkpoint_dimensions = session_fixture()
    seen = []

    class SpyEnv(RealMUVRMAPPOEnv):
        def step(self, actions):
            seen.append(
                (
                    getattr(self, "toporag_kappa", None),
                    bool(getattr(self, "toporag_prior", {})),
                )
            )
            return super().step(actions)

    monkeypatch.setattr(muvr_module, "MUVRMAPPOEnv", SpyEnv)
    adapter = synthetic_adapter(
        SimulationSettings(
            _env_file=None,
            toporag_enabled=True,
            toporag_confidence_mode="lagged_expost",
            toporag_kappa_max=0.7,
        ),
        checkpoint_dimensions,
    )
    adapter.guidance = CognitiveGuidance()
    adapter_net = copy.deepcopy(net)
    adapter(adapter_net, batch, "audit")
    assert seen and {kappa for kappa, _ in seen} == {0.0}
    assert not any(has_prior for _, has_prior in seen)
    seen.clear()
    adapter(adapter_net, batch, "audit")
    assert seen and {kappa for kappa, _ in seen} == {0.7}
    assert all(has_prior for _, has_prior in seen)


def test_replay_contract_accepts_toporag_augmented_source():
    source = {
        "requests": 2,
        "topology_sha256": "topology",
        "workload_sha256": "workload",
        "records": [
            {
                "arrival_s": float(10 * index),
                "h": 0.7,
                "f": 0.2,
                "source": 0,
                "destinations": [1],
                "lifetime_s": 10.0,
                "distinct_sfs": 5,
                "accepted_immediate": index == 0,
            }
            for index in range(2)
        ],
        "accepted": 1,
        "inference_evidence": {
            "model_sha256": "model",
            "planning_records": [
                {
                    "qwen_success": True,
                    "guidance": {
                        "reasoning": f"cycle {index}",
                        "latency_weight": 0.4,
                        "load_balance_weight": 0.6,
                    },
                    "toporag": {"memory_size": index},
                    "toporag_admissions": [{"admission_cycle": index + 1, "players": []}],
                }
                for index in range(2)
            ],
            "request_evidence": [
                {
                    "guidance_used": True,
                    "guidance_cycle": index + 1,
                    "accepted": index == 0,
                    "toporag": [{"kappa": 0.0, "retrieved_k": index}],
                }
                for index in range(2)
            ],
            "toporag": {"parameters": {"enabled": True}, "state_hash": "hash", "memory_size": 2},
        },
    }
    limited = tradeoff.limit_source_requests(source, 1)
    assert limited["requests"] == 1
    assert len(limited["inference_evidence"]["request_evidence"]) == 1
    assert limited["workload_sha256"] != source["workload_sha256"]
    assert limited["inference_evidence"]["planning_records"] == source["inference_evidence"][
        "planning_records"
    ]
    replay = tradeoff.ReplayedGuidance.__new__(tradeoff.ReplayedGuidance)
    replay.source, replay.index = source, 0
    replay.prepare(None, None)
    assert replay.guidance.reasoning == "cycle 0" and replay.index == 1
    replay.prepare(None, None)
    assert replay.guidance.reasoning == "cycle 1" and replay.index == 2

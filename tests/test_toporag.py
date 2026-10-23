"""Local fixtures only: no trained checkpoint, live LLM, or evaluation campaign."""

import copy
import json
import random
import sys

import networkx as nx
import pytest

from muar_sfc.algorithms.mappo.muvr import MUVRMAPPOEnv, MUVRQwenPlanner, TrainedMUVRMAPPO
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.cloud_llm_planner import CognitiveGuidance
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.controllers.toporag import (
    TopoRAGMemory,
    canonical_dag,
    node_prior,
    structural_signature,
)
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.orchestration.handover import fingerprint
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, build_network, generate_session
from muar_sfc.experiment_config import ExperimentConfig, command_plan, load_config
from scripts import benchmark_load_tradeoff as tradeoff


def fixture():
    net = Net2()
    for node in (1, 2):
        net.add_node(node, "server", cpu_capacity=100, cache_capacity=100, ips=0.1)
    net.add_edge(1, 2, bandwidth_capacity=1000, latency=1)
    sfc = SFCGenerator({
        "name": "test_p1_1",
        "vnf_list": [{"type": 2, "name": "vnf", "CPU": 10, "cache": 10,
                      "in_bw": 1, "out_bw": 1}],
        "bandwidth": 1, "src_node": 1, "dst_node": 2, "closer_router": 2,
        "duration": 120, "latency": 6,
    }).generate()
    return net, sfc


@pytest.mark.parametrize("location", ["evidence", "manifest", "execution"])
@pytest.mark.parametrize("mode,kappa", [("fixed", 0.2), ("lagged_expost", 0.0)])
def test_tradeoff_rejects_active_toporag_before_any_run(
    tmp_path, monkeypatch, capsys, location, mode, kappa
):
    parameters = dict(enabled=True, kappa=kappa, confidence_mode=mode)
    document = {"rows": [{"seed": 42}]}
    if location == "evidence":
        document["rows"][0]["inference_evidence"] = {"toporag": {"parameters": parameters}}
    elif location == "manifest":
        document["toporag_manifest"] = {"42": {"parameters": parameters}}
    else:
        document["execution_parameters"] = {"toporag_" + k: v for k, v in parameters.items()}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", [
        "tradeoff", "--guidance-source", str(source), "--output", str(tmp_path / "output"),
        "--model", str(tmp_path / "nonexistent.pt"),
    ])
    monkeypatch.setattr(tradeoff, "create_run_directory", lambda *a: pytest.fail("Created output"))
    monkeypatch.setattr(tradeoff, "simulate", lambda *a, **kw: pytest.fail("Started simulation"))
    with pytest.raises(SystemExit) as error:
        tradeoff.main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "TopoRAG enabled in source seed 42" in message
    assert "Refusing a silent change" in message
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("recorded", [False, True])
def test_tradeoff_legacy_or_disabled_source_keeps_inert_parameters(recorded):
    document = {"rows": [{"seed": 42}]}
    if recorded:
        document["rows"][0]["inference_evidence"] = {
            "toporag": {"parameters": {"enabled": False, "kappa": 0.2}}
        }
    before = copy.deepcopy(document)
    entry = tradeoff.toporag_replay_manifest(document)["42"]
    assert entry["replay"] == ExperimentConfig().toporag.model_dump()
    assert entry["source"]["enabled"] is False
    assert entry["source"]["kappa"] == (0.2 if recorded else 0)
    assert (entry["source_parameters_origin"] == "legacy_defaults_no_toporag_record") != recorded
    assert document == before


def test_memory_determinism_copy_reset_and_network_isolation():
    net, sfc = fixture()
    before = fingerprint(net)
    signature = structural_signature(sfc, net.graph)
    trace = dict(signature=signature, hosts_by_type={"2": [2]}, accepted=True, sla_met=True)
    memory, other = TopoRAGMemory(), TopoRAGMemory()
    empty = memory.state_hash()
    memory.record(trace)
    trace["hosts_by_type"]["2"].append(1)
    assert memory.state_hash() != empty
    first = memory.retrieve(signature, 4)
    assert first == memory.retrieve(signature, 4)
    assert node_prior(first) == {"2": {2: 1.0}}
    first[0]["trace"]["hosts_by_type"]["2"].append(1)
    assert node_prior(memory.retrieve(signature, 4)) == {"2": {2: 1.0}}
    assert len(other) == 0
    assert fingerprint(net) == before
    memory.reset()
    assert memory.state_hash() == empty


def test_exact_match_precedes_approximate_and_failed_traces_have_no_prior():
    memory = TopoRAGMemory()
    query = {"dag": "a", "sf_types": ["2"], "free_band": 3}
    for signature, accepted in [(dict(query, dag="b"), True), (query, False)]:
        memory.record(dict(signature=signature, accepted=accepted, sla_met=accepted,
                           hosts_by_type={"2": [1]}))
    result = memory.retrieve(query, 2)
    assert result[0]["trace"]["signature"] == query
    assert node_prior(result) == {"2": {1: 2 / 3}}


def test_canonical_dag_ignores_ids_including_symmetric_forks():
    dag = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    nx.set_node_attributes(dag, "2", "sf")
    renamed = nx.relabel_nodes(dag, {0: "z", 1: "b", 2: "a", 3: "y"})
    assert canonical_dag(dag) == canonical_dag(renamed)
    renamed.add_edge("b", "a")
    assert canonical_dag(dag) != canonical_dag(renamed)


def test_prior_changes_only_viable_actor_ranking():
    net, sfc = fixture()
    before = fingerprint(net)
    plain = MUVRMAPPOEnv(net.graph, sfc, n_regions=2)
    guided = MUVRMAPPOEnv(net.graph, sfc, n_regions=2)
    guided.toporag_prior = {"2": {2: 1, 999: 9999}}
    guided.toporag_kappa, guided.toporag_prior_scale = 0.5, 10
    masks = copy.deepcopy(plain.get_action_masks())
    for region, mask in guided.get_action_masks().items():
        assert (mask == masks[region]).all()
    actions = {0: 0, 1: plain.max_nodes_per_region}
    plain.step(actions)
    guided.step(actions)
    assert plain.allocated_nodes["vnf"] == 1
    assert guided.allocated_nodes["vnf"] == 2
    assert guided.total_latency <= 6
    assert set(guided.actor_decisions[0]["proposals"].values()) == {1, 2}
    assert fingerprint(net) == before
    impossible = copy.deepcopy(sfc)
    impossible.latency_request = 0.00001
    env = MUVRMAPPOEnv(net.graph, impossible, n_regions=2)
    env.toporag_prior, env.toporag_kappa, env.toporag_prior_scale = {"2": {2: 1}}, 0.9, 1e9
    env.step(actions)
    assert not env.success and not env.allocated_nodes


@pytest.mark.parametrize("seed", [42, 43])
def test_zero_kappa_is_identical_and_prompt_context_is_opt_in(seed):
    net = build_network(MUVRProfile(), seed)
    batch, _ = generate_session(MUVRProfile(), 0, random.Random(seed + 100000))
    plain, inert = (MUVRMAPPOEnv(net.graph, batch[0]) for _ in range(2))
    inert.toporag_prior = {str(v.type): {n: 1 for n in net.graph} for v in inert.vnf_sequence}
    inert.toporag_kappa, inert.toporag_prior_scale = 0, 999
    done = False
    while not done:
        actions, _ = plain.expert_actions()
        p = plain.step(actions)
        q = inert.step(actions)
        done = p[3]
        assert p[2:] == q[2:]
        assert plain.allocated_nodes == inert.allocated_nodes
        assert plain.total_latency == inert.total_latency
    planner = MUVRQwenPlanner(SimulationSettings(_env_file=None), net)
    planner.current_batch = batch
    net._toporag_context = [{"example": "detached"}]
    snapshot = planner.extract_telemetry_snapshot()
    assert "toporag_context" not in planner.compact_telemetry(snapshot)
    planner.settings = planner.settings.model_copy(update={"toporag_prompt_context": True})
    assert planner.compact_telemetry(planner.extract_telemetry_snapshot())["toporag_context"]


def test_evidence_and_lagged_confidence_without_checkpoint():
    net, sfc = fixture()
    sfc.arrival_time, sfc.session_id = 0, 0
    adapter = TrainedMUVRMAPPO.__new__(TrainedMUVRMAPPO)
    adapter.settings = SimulationSettings(_env_file=None, toporag_enabled=True,
                                          toporag_confidence_mode="lagged_expost")
    adapter.toporag_memory = TopoRAGMemory()
    adapter.reset_toporag()
    adapter.request_evidence = [{"guidance_used": True}]
    adapter.planning_records = [{"guidance": {"reasoning": "unchanged"}}]
    env = MUVRMAPPOEnv(net.graph, sfc, n_regions=2)
    env.step({0: 0, 1: env.max_nodes_per_region})
    metadata = adapter._toporag_begin(net.graph, sfc)
    adapter._toporag_finish(adapter.request_evidence[-1], [(sfc, env, metadata)], True)
    assert adapter.request_evidence[-1]["toporag"][0]["retrieved_k"] == 0
    assert adapter._previous_cycle_success == 1
    assert adapter._effective_toporag_kappa() == 0.9
    assert adapter.planning_records[0]["guidance"] == {"reasoning": "unchanged"}
    assert adapter.planning_records[0]["toporag_admissions"][0]["players"][0]["kappa"] == 0
    assert len(adapter.toporag_memory) == 1
    assert adapter.toporag_decisions[0]["state_hash_after"] == adapter.toporag_memory.state_hash()
    adapter.reset_toporag()
    assert len(adapter.toporag_memory) == 0
    assert adapter._previous_cycle_success == 0


def test_config_defaults_validation_and_forwarding(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[execution]\nstage="evaluate"\n')
    assert load_config(path).toporag == ExperimentConfig().toporag
    path.write_text('[toporag]\nenabled=true\nkappa=0.2\nprompt_context=true\n')
    config = load_config(path)
    command = command_plan(config, tmp_path, "python")[-1]
    assert "--toporag-enabled" in command and "--toporag-prompt-context" in command
    assert command[command.index("--toporag-kappa") + 1] == "0.2"
    for fields in [{"kappa_max": 1}, {"top_k": 0}, {"confidence_mode": "learned"}]:
        with pytest.raises(ValueError):
            ExperimentConfig(toporag=fields)


def synthetic_adapter(enabled, **settings):
    """Exercise admission integration, without claiming synthetic actions were trained."""
    adapter = TrainedMUVRMAPPO.__new__(TrainedMUVRMAPPO)
    adapter.settings = SimulationSettings(_env_file=None, toporag_enabled=enabled, **settings)
    adapter.qwen = False
    adapter.pending = adapter.guidance_ready_at = adapter.guidance = None
    adapter.planning_records, adapter.request_evidence = [], []
    adapter.toporag_memory = TopoRAGMemory()
    adapter.reset_toporag()

    def search(graph, sfc):
        env = MUVRMAPPOEnv(graph, sfc)
        env.toporag_prior = adapter._toporag_prior
        env.toporag_kappa = adapter._toporag_kappa
        env.toporag_prior_scale = adapter.settings.toporag_prior_scale
        done = False
        while not done:
            actions, _ = env.expert_actions()
            _, _, _, done, _ = env.step(actions)
        env.search_summary = []
        return env, env.actor_decisions

    adapter._search_policy = search
    return adapter


def test_admission_zero_kappa_matches_control_and_records_per_player():
    profile = MUVRProfile(players=2)
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    for sfc in batch:
        sfc.arrival_time = 0
    control, inert = synthetic_adapter(False), synthetic_adapter(True)
    before = fingerprint(net)
    plain, latencies, reason = control(net, batch, "test")
    trial, actual, actual_reason = inert(net, batch, "test")
    assert actual == latencies and actual_reason == reason
    assert (plain is None) == (trial is None)
    if trial is not None:
        assert fingerprint(plain) == fingerprint(trial)
    assert fingerprint(net) == before
    evidence = copy.deepcopy(inert.request_evidence[-1])
    metadata = evidence.pop("toporag")
    assert evidence == control.request_evidence[-1]
    assert len(metadata) == len(inert.toporag_memory)
    assert len(metadata) >= 1
    assert all(row["kappa"] == 0 for row in metadata)


def test_confidence_is_lagged_and_atomic_failure_does_not_supply_positive_prior():
    profile = MUVRProfile(players=1)
    net = build_network(profile, 42)
    batch, _ = generate_session(profile, 0, random.Random(100042))
    batch[0].arrival_time = 0
    adapter = synthetic_adapter(
        True, toporag_confidence_mode="lagged_expost", toporag_kappa_max=0.7
    )
    adapter.guidance = CognitiveGuidance()
    adapter.planning_records = [{"guidance": adapter.guidance.model_dump()}]
    trial, _, _ = adapter(net, batch, "test")
    assert trial is not None
    assert adapter.request_evidence[-1]["toporag"][0]["kappa"] == 0
    adapter(net, batch, "test")
    assert adapter.request_evidence[-1]["toporag"][0]["kappa"] == 0.7
    assert len(adapter.planning_records) == 1  # replay cycle numbering preserved
    assert len(adapter.planning_records[0]["toporag_admissions"]) == 2
    adapter.reset_toporag()
    failed = copy.deepcopy(batch[0])
    failed.latency_request = 0.00001
    adapter(net, [batch[0], failed], "test")
    assert not adapter.request_evidence[-1]["accepted"]
    assert len(adapter.toporag_memory) == 2
    assert not node_prior(adapter.toporag_memory.retrieve(
        structural_signature(batch[0], net.graph), 4
    ))
    assert adapter._effective_toporag_kappa() == 0

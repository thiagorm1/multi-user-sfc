import copy
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from muar_sfc.algorithms.mappo.muvr import TrainedMUVRMAPPO
from muar_sfc.controllers.cloud_llm_planner import CognitiveGuidance
from muar_sfc.core.ar_metrics import ar_frame_summary, distribution
from muar_sfc.core.orchestration.handover import fingerprint, prepare_handover
from muar_sfc.core.scenarios.muvr_reference import (
    MUVRProfile,
    build_network,
    generate_session,
    plan_colocated,
)


@pytest.fixture
def adapter():
    a = TrainedMUVRMAPPO.__new__(TrainedMUVRMAPPO)
    a.qwen = True
    a.settings = SimpleNamespace(
        cognitive_planner_interval=60, cognitive_guidance_max_age=120, seed=42
    )
    a.executor = ThreadPoolExecutor(max_workers=1)
    a.pending = None
    a.last_submission = None
    a.guidance_ready_at = None
    a.guidance = None
    a.guidance_applied_requests = a.guidance_misses = a.background_skipped_busy = 0
    a.closed = False
    a.call_results, a.planning_records, a.request_evidence = [], [], []
    a._search_policy = lambda graph, sfc: (SimpleNamespace(success=False, search_summary=[]), [])
    yield a
    a.close()


def batch():
    p = MUVRProfile(players=1)
    result, _ = generate_session(p, 0, random.Random(100042))
    result[0].arrival_time = 0
    return result


def test_slow_qwen_does_not_block_admission_and_snapshot_is_detached(adapter):
    started, release = threading.Event(), threading.Event()
    net = build_network(MUVRProfile(), 42)
    original_cpu = net.graph.nodes[1]["cpu_used"]

    def slow(snapshot, requests, now):
        snapshot.graph.nodes[1]["cpu_used"] = 999
        started.set()
        assert release.wait(5)
        return (
            CognitiveGuidance(),
            time.perf_counter(),
            {
                "qwen_success": True,
                "planning_seconds": 1,
            },
        )

    adapter._background_plan = slow
    requests = batch()
    try:
        adapter.prepare(net, requests)
        assert started.wait(2)
        result = adapter(net, requests, "mappo_qwen")
        assert result[0] is None
        assert not adapter.pending.done()
        assert net.graph.nodes[1]["cpu_used"] == original_cpu
        assert adapter.request_evidence[-1]["qwen_seconds"] == 0
        assert not adapter.request_evidence[-1]["guidance_used"]
        requests[0].arrival_time = 120
        adapter.prepare(net, requests)
        assert adapter.background_skipped_busy == 1
    finally:
        release.set()
    adapter.close()
    assert adapter.guidance is not None
    assert adapter.call_results == [True]


def test_failed_qwen_is_recorded_without_fake_guidance(adapter):
    adapter._background_plan = lambda *args: (
        None,
        time.perf_counter(),
        {
            "qwen_success": False,
            "planning_seconds": 1,
            "error": "offline",
        },
    )
    net = build_network(MUVRProfile(), 42)
    adapter.prepare(net, batch())
    adapter.close()
    adapter(net, batch(), "mappo_qwen")
    assert adapter.guidance is None
    assert adapter.call_results == [False]
    assert adapter.guidance_misses == 1


def test_expired_guidance_is_not_used(adapter):
    adapter.guidance = CognitiveGuidance()
    adapter.guidance_ready_at = time.perf_counter() - 121
    adapter(build_network(MUVRProfile(), 42), batch(), "mappo_qwen")
    assert adapter.guidance is None
    assert adapter.guidance_misses == 1


def test_ar_telemetry_is_not_inferred_from_sfc_latency():
    assert ar_frame_summary()["status"] == "not_measured"
    measured = ar_frame_summary(
        [
            {"motion_s": 1, "display_s": 1.005},
            {"motion_s": 2, "display_s": 2.025},
            {"dropped": True},
        ],
        budget_ms=10,
    )
    assert measured["motion_to_photon_ms"]["p99"] == pytest.approx(24.8)
    assert measured["budget_exceeded_frames"] == 1
    assert measured["dropped_frames"] == 1
    assert measured["latency_jitter_ms"]["mean"] == pytest.approx(20)
    assert measured["display_interval_ms"]["mean"] == pytest.approx(1020)
    assert not measured["comfort_validated"]
    assert distribution([])["p95"] is None
    with pytest.raises(ValueError):
        ar_frame_summary([{"motion_s": 2, "display_s": 1}])


def test_unexpected_worker_error_keeps_cached_guidance(adapter):
    def broken(*args):
        raise RuntimeError("worker failure")

    adapter._background_plan = broken
    adapter.guidance = CognitiveGuidance()
    adapter.guidance_ready_at = time.perf_counter()
    net = build_network(MUVRProfile(), 42)
    adapter.prepare(net, batch())
    adapter.close()
    adapter(net, batch(), "mappo_qwen")
    assert adapter.call_results == [False]
    assert adapter.guidance_applied_requests == 1
    assert adapter.planning_records[0]["error"] == "RuntimeError: worker failure"


def handover_setup():
    profile = MUVRProfile(players=1, sla=100)
    net = build_network(profile, 42)
    old, _ = generate_session(profile, 0, random.Random(100042))
    new, _ = generate_session(profile, 1, random.Random(100043))
    net, _ = plan_colocated(net, old, 1)
    return net, old, new


def colocated(net, requests, strategy):
    try:
        result, latencies = plan_colocated(net, requests, 2)
        return result, latencies, "colocated"
    except ValueError:
        return None, [], "no_capacity"


def test_handover_reserves_both_versions_before_atomic_commit():
    net, old, new = handover_setup()
    before = fingerprint(net)
    prepared = prepare_handover(net, [old[0].id], new, colocated)
    assert fingerprint(net) == before
    assert old[0].id in prepared.candidate.sfc_dict
    assert new[0].id in prepared.candidate.sfc_dict
    replacement = prepared.commit(net)
    assert old[0].id in net.sfc_dict
    assert old[0].id not in replacement.sfc_dict
    assert new[0].id in replacement.sfc_dict
    with pytest.raises(ValueError, match="already"):
        prepared.commit(net)


def test_failed_or_stale_handover_preserves_original_session():
    net, old, new = handover_setup()
    before = fingerprint(net)
    saturated = copy.deepcopy(net)
    saturated.graph.nodes[2]["cpu_capacity"] = 0
    assert prepare_handover(saturated, [old[0].id], new, colocated) is None
    prepared = prepare_handover(net, [old[0].id], new, colocated)
    net.graph.nodes[2]["cpu_capacity"] -= 1
    with pytest.raises(ValueError, match="changed"):
        prepared.commit(net)
    assert old[0].id in net.sfc_dict
    assert new[0].id not in net.sfc_dict
    assert fingerprint(prepared.candidate) != before

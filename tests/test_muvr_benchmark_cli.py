import json

from scripts import benchmark_mappo_qwen_muvr as benchmark
from muar_sfc.controllers.toporag import TopoRAGMemory


def test_sla_override_reaches_workload_and_adapter_without_real_inference(tmp_path, monkeypatch):
    observed = []

    class Adapter:
        def __init__(self, settings, qwen, policy_schema):
            observed.append((settings.latency, settings.seed, qwen, policy_schema))
            self.closed = False
            self.toporag_memory = TopoRAGMemory()

        def toporag_parameters(self):
            return {"enabled": False}

        def close(self):
            self.closed = True

        def evidence(self):
            assert self.closed
            return dict(checkpoint_loaded=True, qwen_successful_calls=0,
                        qwen_fallback_calls=0, qwen_planning_seconds=0)

    def simulate(profile, seed, strategy, admitter, **kwargs):
        assert profile.sla == 12
        assert profile.requests == 50
        return dict(seed=seed, strategy=strategy, accepted=0, acceptance_percent=0,
                    mean_latency_ms=None, max_latency_ms=None, mean_decision_s=0)

    monkeypatch.setattr(benchmark, "TrainedMUVRMAPPO", Adapter)
    monkeypatch.setattr(benchmark, "simulate", simulate)
    benchmark.main(["--sla-ms", "12", "--requests", "50", "--seeds", "42", "43", "44",
                    "--variants", "mappo", "--output", str(tmp_path / "run")])
    document = json.loads((tmp_path / "run/mappo.json").read_text())
    assert document["profile"]["sla"] == 12
    assert document["execution_parameters"]["sla_ms"] == 12
    assert observed == [(12, seed, False, "muvr_static_v1") for seed in (42, 43, 44)]
    assert len(document["rows"]) == 3

import hashlib
import json
import sys

import pytest

from scripts import benchmark_load_tradeoff as benchmark


def test_colliding_runs_preserve_existing_files_and_allocate_new_directories(tmp_path):
    prefix = tmp_path / "experiment"
    prefix.mkdir()
    original = prefix / "summary.json"
    original.write_text("existing result")
    first = benchmark.create_run_directory(prefix, stamp="same_time")
    benchmark.write_json_new(first / "summary.json", {"run": 1})
    second = benchmark.create_run_directory(prefix, stamp="same_time")
    assert second != first and second.is_dir()
    assert original.read_text() == "existing result"
    assert json.loads((first / "summary.json").read_text()) == {"run": 1}
    assert not (second / "summary.json").exists()


def test_exclusive_write_refuses_to_replace_an_existing_result(tmp_path):
    target = tmp_path / "result.json"
    benchmark.write_json_new(target, {"accepted": 126})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.write_json_new(target, {"accepted": 130})
    assert target.read_bytes() == before


def test_interrupted_run_keeps_completed_seed_and_restart_uses_new_directory(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint fixture")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "seed": seed,
                        "requests": 1,
                        "topology_sha256": "topology",
                        "workload_sha256": "workload",
                        "inference_evidence": {
                            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()
                        },
                    }
                    for seed in (42, 43)
                ]
            }
        )
    )
    original_source, original_model = source.read_bytes(), model.read_bytes()
    prefix = tmp_path / "results"
    create = benchmark.create_run_directory
    monkeypatch.setattr(benchmark, "create_run_directory", lambda base: create(base, "fixed"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--guidance-source",
            str(source),
            "--model",
            str(model),
            "--scales",
            "1",
            "0",
            "--output",
            str(prefix),
        ],
    )

    class Adapter:
        def __init__(self, *_args):
            self.node_samples = [20]
            self.peak_samples = [20]
            self.imbalance_samples = [0]

        def close(self):
            pass

        def evidence(self):
            return {}

    monkeypatch.setattr(benchmark, "ReplayedGuidance", Adapter)

    def interrupted(_profile, seed, *_args, **_kwargs):
        if seed == 43:
            raise KeyboardInterrupt
        return {
            "seed": seed,
            "accepted": 1,
            "max_latency_ms": 5,
            "topology_sha256": "topology",
            "workload_sha256": "workload",
        }

    monkeypatch.setattr(benchmark, "simulate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        benchmark.main()
    first = tmp_path / "results_fixed"
    completed = first / "seeds/scale_1/seed_42.json"
    before = completed.read_bytes()
    assert not (first / "summary.json").exists()
    assert not (first / "scale_1.json").exists()

    def successful(_profile, seed, *_args, **_kwargs):
        return {
            "seed": seed,
            "accepted": 1,
            "max_latency_ms": 5,
            "topology_sha256": "topology",
            "workload_sha256": "workload",
            "utilization_samples": [{"cpu_percent": 20}],
            "records": [{"latencies_ms": [5], "decision_s": 0.01}],
        }

    monkeypatch.setattr(benchmark, "simulate", successful)
    benchmark.main()
    second = tmp_path / "results_fixed_01"
    assert completed.read_bytes() == before
    assert source.read_bytes() == original_source
    assert model.read_bytes() == original_model
    assert len(json.loads((second / "summary.json").read_text())) == 2
    assert len(list((second / "seeds").glob("*/seed_*.json"))) == 4
    assert (second / "scale_1.json").exists()
    assert (second / "scale_0.json").exists()


def prefix_source():
    records = [
        {
            "arrival_s": index * 10,
            "h": 0.8,
            "f": 0.2,
            "source": 1,
            "destinations": [2, 2, 2, 2],
            "lifetime_s": 60,
            "distinct_sfs": 18,
            "accepted_immediate": index != 1,
            "decision_s": 0.1,
        }
        for index in range(3)
    ]
    return {
        "requests": 3,
        "records": records,
        "workload_sha256": "original-full-hash",
        "inference_evidence": {
            "request_evidence": [{"guidance_cycle": i} for i in range(3)],
            "planning_records": [{"guidance": "unchanged"}],
        },
    }


def test_shorter_workload_preserves_source_and_guidance_prefix():
    source = prefix_source()
    before = json.dumps(source, sort_keys=True)
    selected = benchmark.limit_source_requests(source, 2)
    assert selected["requests"] == 2
    assert selected["accepted"] == 1
    assert selected["records"] == source["records"][:2]
    assert (
        selected["inference_evidence"]["request_evidence"]
        == (source["inference_evidence"]["request_evidence"][:2])
    )
    assert selected["workload_sha256"] != source["workload_sha256"]
    assert json.dumps(source, sort_keys=True) == before
    # Placement outcomes and controller speed must not change workload identity.
    other = prefix_source()
    other["records"][0]["decision_s"] = 9
    other["records"][0]["accepted_immediate"] = False
    assert (
        benchmark.limit_source_requests(other, 2)["workload_sha256"]
        == (selected["workload_sha256"])
    )


@pytest.mark.parametrize("requests", [0, -1, 4])
def test_request_limit_cannot_exceed_recorded_workload(requests):
    with pytest.raises(ValueError):
        benchmark.limit_source_requests(prefix_source(), requests)

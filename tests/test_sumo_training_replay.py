import copy
import json
from dataclasses import asdict

import pytest

from muar_sfc.algorithms.mappo.mobile import OBSERVATION_SCHEMA_SHA256
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile
from muar_sfc.core.scenarios.sumo_replay import (
    DATASET_SCHEMA,
    TRACE_SCHEMA,
    TrajectoryReplay,
    document_sha256,
    load_dataset,
    validate_seed_splits,
    validate_trace,
)


def trace(seed=1):
    profile = asdict(MUVRProfile(requests=1, lifetime=2))
    frames = []
    for at in range(3):
        features = [[float(at)] + [0.0] * 6 + [1 / 25] + [0.0] * 4 for _ in range(4)]
        frames.append(dict(time_s=at, destinations=[1] * 4,
                           observation=dict(time_s=at, features=features)))
    document = dict(schema=TRACE_SCHEMA, seed=seed, profile=profile, step_s=1.0,
                    observation_schema_sha256=OBSERVATION_SCHEMA_SHA256,
                    frames=frames, source_evidence=dict(mode="sumo_traci", sumo_version="fixture",
                    seed=seed, step_s=1.0,
                    net_sha256="net", routes_sha256="routes", anchors_sha256="anchors"))
    document["trace_sha256"] = document_sha256(document)
    return document


def rehash(document, key):
    document.pop(key, None)
    document[key] = document_sha256(document)
    return document


def test_integer_anchor_keys_keep_checksum_after_json_roundtrip():
    anchors = {node: [float(node), 0.0] for node in range(1, 26)}
    evidence = dict(anchors=anchors)
    assert document_sha256(evidence) == document_sha256(json.loads(json.dumps(evidence)))
    document = trace()
    document["source_evidence"]["anchors"] = anchors
    rehash(document, "trace_sha256")
    restored = json.loads(json.dumps(document))
    assert validate_trace(restored) == restored
    replay = TrajectoryReplay(restored)
    assert replay.destinations(0) == [1] * 4


def test_replay_exposes_only_past_frames_and_returns_independent_observations():
    original = trace()
    altered = copy.deepcopy(original)
    altered["frames"][2]["observation"]["features"][0][0] = 999
    rehash(altered, "trace_sha256")
    first, second = TrajectoryReplay(original), TrajectoryReplay(altered)
    for at in (0, 0.5, 1, 1.9):
        first.advance(at)
        second.advance(at)
        assert first.observation(0) == second.observation(0)
        assert first.observation(0)["time_s"] <= at
    mutation = first.observation(0)
    mutation["features"][0][0] = -999
    assert first.observation(0)["features"][0][0] == 1
    with pytest.raises(ValueError, match="horizon"):
        first.advance(1)
    first.retire(0)
    with pytest.raises(ValueError, match="inactive"):
        first.observation(0)


@pytest.mark.parametrize("splits", [
    dict(train=[1], validation=[1], test=[3]),
    dict(train=[1, 1], validation=[2], test=[3]),
    dict(train=[], validation=[2], test=[3]),
])
def test_seed_splits_reject_overlap_duplicates_and_empty_groups(splits):
    with pytest.raises(ValueError, match="disjoint"):
        validate_seed_splits(splits)


def test_truncated_trace_is_rejected_even_if_rehashed():
    document = trace()
    document["frames"].pop()
    rehash(document, "trace_sha256")
    with pytest.raises(ValueError, match="complete lifetime"):
        validate_trace(document)


def test_incomplete_dataset_manifest_cannot_be_used_for_training(tmp_path):
    traces = [trace(seed) for seed in (1, 2, 3)]
    for item in traces:
        (tmp_path / f"seed_{item['seed']}.json").write_text(json.dumps(item))
    manifest = dict(schema=DATASET_SCHEMA, complete=False,
                    profile=traces[0]["profile"], step_s=1.0,
                    seed_splits=dict(train=[1], validation=[2], test=[3]),
                    trace_hashes={str(item["seed"]): item["trace_sha256"] for item in traces})
    rehash(manifest, "manifest_sha256")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete"):
        load_dataset(tmp_path)

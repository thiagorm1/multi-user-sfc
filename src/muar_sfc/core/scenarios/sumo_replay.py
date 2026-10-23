"""Validated real-SUMO traces and causal replay for single-session training."""

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

TRACE_SCHEMA = "sumo_muvr_trace_v1"
DATASET_SCHEMA = "sumo_muvr_dataset_v1"


def document_sha256(document):
    # JSON object keys become strings on disk. Normalize before sorting so integer
    # anchor IDs have exactly the same canonical ordering before and after loading.
    normalized = json.loads(json.dumps(document, allow_nan=False))
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_trace(document):
    if document.get("schema") != TRACE_SCHEMA:
        raise ValueError("Unknown SUMO trace schema")
    payload = {key: value for key, value in document.items() if key != "trace_sha256"}
    if document_sha256(payload) != document.get("trace_sha256"):
        raise ValueError("SUMO trace checksum mismatch")
    from muar_sfc.algorithms.mappo.mobile import OBSERVATION_SCHEMA_SHA256
    from muar_sfc.core.scenarios.muvr_reference import MUVRProfile

    if document.get("observation_schema_sha256") != OBSERVATION_SCHEMA_SHA256:
        raise ValueError("SUMO trace observation schema mismatch")
    profile = document["profile"]
    MUVRProfile(**profile)
    if profile["requests"] != 1 or profile["players"] != 4 or profile["servers"] != 25:
        raise ValueError("Training traces require one four-player session and 25 servers")
    step = document["step_s"]
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Invalid SUMO trace step")
    frames = document["frames"]
    expected_count = math.ceil(profile["lifetime"] / step) + 1
    if len(frames) != expected_count:
        raise ValueError("SUMO trace does not cover the complete lifetime horizon")
    for number, frame in enumerate(frames):
        if not math.isclose(frame["time_s"], number * step):
            raise ValueError("SUMO trace timestamps must cover every step from zero")
        if frame["observation"]["time_s"] != frame["time_s"]:
            raise ValueError("SUMO observation timestamp mismatch")
        destinations = frame["destinations"]
        features = np.asarray(frame["observation"]["features"], dtype=np.float32)
        if (
            len(destinations) != 4
            or any(not 1 <= node <= 25 for node in destinations)
            or features.shape != (4, 12)
            or not np.isfinite(features).all()
        ):
            raise ValueError("Invalid SUMO trace observation")
        if not np.allclose(features[:, 7], np.asarray(destinations) / 25):
            raise ValueError("SUMO access features disagree with destinations")
    evidence = document["source_evidence"]
    if evidence.get("seed") != document["seed"] or evidence.get("step_s") != step:
        raise ValueError("SUMO source seed/step disagree with the trace")
    if evidence.get("mode") != "sumo_traci" or any(
        not evidence.get(key)
        for key in ("sumo_version", "net_sha256", "routes_sha256", "anchors_sha256")
    ):
        raise ValueError("Trace lacks real SUMO provenance")
    return document


def collect_trace(profile, seed, net_file, binary="sumo", step_s=1.0):
    """Runs SUMO only, independent of actor admission decisions; never calls an LLM."""
    from muar_sfc.algorithms.mappo.mobile import OBSERVATION_SCHEMA_SHA256
    from muar_sfc.core.scenarios.sumo_mobility import SumoMobility

    if profile.requests != 1 or profile.players != 4:
        raise ValueError("Collect one complete four-player session per seed")
    frames = []
    with SumoMobility(
        net_file,
        seed,
        profile.servers,
        1,
        4,
        profile.arrival_interval,
        profile.lifetime,
        binary=binary,
        step_s=step_s,
    ) as source:
        for index in range(math.ceil(profile.lifetime / step_s) + 1):
            at = index * step_s
            if index:
                source.advance(at)
            destinations = source.destinations(0)
            frames.append(
                {"time_s": at, "destinations": destinations, "observation": source.observation(0)}
            )
        evidence = source.evidence()
    document = {
        "schema": TRACE_SCHEMA,
        "observation_schema_sha256": OBSERVATION_SCHEMA_SHA256,
        "seed": seed,
        "profile": asdict(profile),
        "step_s": step_s,
        "frames": frames,
        "source_evidence": evidence,
        "collection_policy": "all four vehicles retained for the full horizon",
    }
    document["trace_sha256"] = document_sha256(document)
    return validate_trace(document)


class TrajectoryReplay:
    """Only the latest sample at or before the requested time is exposed."""

    def __init__(self, document):
        self._document = copy.deepcopy(validate_trace(document))
        self.step_s = document["step_s"]
        self.time_s = 0.0
        self._frame = self._document["frames"][0]
        self._retired = False

    def advance(self, time_s):
        if time_s < self.time_s or time_s > self._document["frames"][-1]["time_s"]:
            raise ValueError("Replay clock outside its causal validated horizon")
        self.time_s = time_s
        index = min(
            int(math.floor((time_s + 1e-9) / self.step_s)), len(self._document["frames"]) - 1
        )
        self._frame = self._document["frames"][index]
        if self._frame["time_s"] > time_s + 1e-9:
            raise ValueError("Replay would expose a future frame")

    def destinations(self, index):
        if index != 0 or self._retired:
            raise ValueError("Training replay contains only the active session zero")
        return list(self._frame["destinations"])

    def observation(self, index):
        if index != 0 or self._retired:
            raise ValueError("No observation for inactive replay session")
        return copy.deepcopy(self._frame["observation"])

    def retire(self, index):
        if index != 0:
            raise ValueError("Unknown replay session")
        self._retired = True

    def evidence(self):
        source = self._document["source_evidence"]
        return {
            "mode": "sumo_real_trace_replay",
            "trace_schema": TRACE_SCHEMA,
            "trace_sha256": self._document["trace_sha256"],
            "seed": self._document["seed"],
            "step_s": self.step_s,
            "net_sha256": source["net_sha256"],
            "routes_sha256": source["routes_sha256"],
            "anchors_sha256": source["anchors_sha256"],
            "future_frames_exposed_to_policy": False,
            "radio_or_packet_continuity_simulated": False,
        }


def validate_seed_splits(splits):
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Declare train, validation and test seed splits")
    seen = set()
    for name, seeds in splits.items():
        if not seeds or len(seeds) != len(set(seeds)) or seen.intersection(seeds):
            raise ValueError(f"Seed splits must be nonempty, unique and disjoint: {name}")
        seen.update(seeds)
    return splits


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("Unknown SUMO dataset schema")
    if manifest.get("complete") is not True:
        raise ValueError("SUMO dataset collection is incomplete")
    validate_seed_splits(manifest["seed_splits"])
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if document_sha256(payload) != manifest.get("manifest_sha256"):
        raise ValueError("SUMO dataset manifest checksum mismatch")
    result = {name: [] for name in manifest["seed_splits"]}
    for name, seeds in manifest["seed_splits"].items():
        for seed in seeds:
            trace = validate_trace(json.loads((directory / f"seed_{seed}.json").read_text()))
            if (
                trace["seed"] != seed
                or trace["trace_sha256"] != manifest["trace_hashes"][str(seed)]
            ):
                raise ValueError("Dataset seed or trace hash mismatch")
            if trace["profile"] != manifest["profile"]:
                raise ValueError("Dataset profiles differ")
            if trace["step_s"] != manifest["step_s"]:
                raise ValueError("Dataset trajectory steps differ")
            result[name].append(trace)
    return manifest, result

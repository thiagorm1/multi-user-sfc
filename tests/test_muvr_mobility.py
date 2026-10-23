import pytest

from muar_sfc.core.scenarios import muvr_reference as scenario


class MovingPlayers:
    step_s = 1.0

    def __init__(self, switch_at=1):
        self.time = 0
        self.switch_at = switch_at
        self.retired = []
        self.advances = []

    def advance(self, time):
        assert time >= self.time
        self.time = time
        self.advances.append(time)

    def destinations(self, index):
        assert index not in self.retired
        return [1 if self.time < self.switch_at else 2] * 4

    def retire(self, index):
        assert index not in self.retired
        self.retired.append(index)

    def evidence(self):
        return {"mode": "deterministic_test_source"}


def fixed_duration(monkeypatch, duration):
    original = scenario.generate_session

    def generate(*args):
        batch, metadata = original(*args)
        for sfc in batch:
            sfc.duration = duration
        metadata["lifetime_s"] = duration
        return batch, metadata

    monkeypatch.setattr(scenario, "generate_session", generate)


@pytest.mark.parametrize("fail_handover", [False, True])
def test_moving_session_reserves_before_release_and_cleans_up(monkeypatch, fail_handover):
    fixed_duration(monkeypatch, 3.5)
    mobility = MovingPlayers()
    snapshots = []
    phases = []

    def placer(net, batch, strategy):
        is_handover = bool(net.sfc_dict)
        phases.append(is_handover)
        if is_handover:
            assert len(net.sfc_dict) == 4  # Old service still occupies resources.
            assert net.metrics.total_cpu_used == pytest.approx(20)
            assert all(s.dst_node == 2 for s in batch)
            if fail_handover:
                return None, [], "forced_failure"
        candidate, latencies = scenario.plan_colocated(net, batch, 1)
        snapshots.append(candidate)
        return candidate, latencies, "test_colocated"

    row = scenario.simulate(scenario.MUVRProfile(requests=1, sla=100), 42,
                            admitter=placer, mobility=mobility)
    assert phases == [False, True]
    assert row["accepted"] == 1
    assert mobility.retired == [0]
    assert row["mobility"]["access_change_events"] == 1
    assert row["mobility"]["sessions_interrupted"] == int(fail_handover)
    assert row["mobility"]["handovers_committed"] == int(not fail_handover)
    assert row["mobility"]["unserved_session_seconds"] == (2.5 if fail_handover else 0)
    assert row["records"][0]["destinations"] == [1] * 4
    if fail_handover:
        assert not snapshots[0].sfc_dict
        assert snapshots[0].metrics.total_cpu_used == pytest.approx(0)
    else:
        # Prepared snapshot includes both IDs; shared instances are reference counted.
        assert len(snapshots[1].sfc_dict) == 8
        scenario.audit(snapshots[1])
        assert row["mobility"]["events"][0]["latencies_ms"]


def test_expiry_before_tick_prevents_handover_after_session_end(monkeypatch):
    fixed_duration(monkeypatch, 0.5)
    mobility = MovingPlayers()
    row = scenario.simulate(scenario.MUVRProfile(requests=1, sla=100), 42,
                            mobility=mobility)
    assert row["accepted"] == 1
    assert row["observation_seconds"] == 0.5
    assert row["mobility"]["events"] == []
    assert mobility.retired == [0]


def test_mobile_destinations_change_workload_identity_but_not_topology(monkeypatch):
    fixed_duration(monkeypatch, 0.5)
    profile = scenario.MUVRProfile(requests=1, sla=100)
    first = scenario.simulate(profile, 42, mobility=MovingPlayers(switch_at=1))
    second = scenario.simulate(profile, 42, mobility=MovingPlayers(switch_at=0))
    assert first["topology_sha256"] == second["topology_sha256"]
    assert first["workload_sha256"] != second["workload_sha256"]
    assert "mobility" not in scenario.simulate(profile, 42)

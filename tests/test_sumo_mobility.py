from types import SimpleNamespace

import numpy as np
import pytest

from muar_sfc.core.scenarios.sumo_mobility import SumoMobility


def test_missing_sumo_fails_instead_of_using_static_positions(monkeypatch, tmp_path):
    monkeypatch.setattr("muar_sfc.core.scenarios.sumo_mobility.shutil.which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="Eclipse SUMO"):
        SumoMobility(tmp_path / "unused.net.xml", 42, 25, 1, 1, 10, 120)


def fake_source(tmp_path):
    source = SumoMobility.__new__(SumoMobility)
    source.players, source.step_s, source.seed = 2, 1.0, 42
    source.time_s, source.positions = 0.0, []
    source.anchors = {1: (0, 0), 2: (100, 0)}
    source.routes = {"muvr_0_0": ["a", "b"], "muvr_0_1": ["b", "a"]}
    source.net_file = tmp_path / "roads.net.xml"
    source.net_file.write_text("test network")
    positions = {"muvr_0_0": (5, 0), "muvr_0_1": (95, 0)}
    steps, removed, teleports, pending = [], [], [], set()
    def remove(vehicle):
        removed.append(vehicle)
        positions.pop(vehicle, None)

    source.connection = SimpleNamespace(
        simulationStep=steps.append,
        vehicle=SimpleNamespace(getIDList=lambda: [v for v in positions if v not in pending],
                                getPosition=positions.__getitem__, getAngle=lambda _: 0,
                                getSpeed=lambda _: 0, remove=remove),
        simulation=SimpleNamespace(getStartingTeleportIDList=lambda: teleports,
                                   getPendingVehicles=lambda: sorted(pending)),
        getVersion=lambda: (22, "fake TraCI"),
    )
    source.road = SimpleNamespace(
        getEdge=lambda edge: SimpleNamespace(
            getShape=lambda: [(0, 0), (0, 100)] if edge == "a" else [(100, 0), (100, 100)]
        )
    )
    source._pending_departures = set()
    source._test_pending = pending
    return source, positions, steps, removed, teleports


def test_traci_clock_association_and_retirement_are_explicit(tmp_path):
    source, positions, steps, removed, _ = fake_source(tmp_path)
    source.advance(0)
    assert steps == [1]
    assert source.destinations(0) == [1, 2]
    positions["muvr_0_0"] = (90, 0)
    source.advance(10)
    assert steps == [1, 11]
    assert source.destinations(0) == [2, 2]
    evidence = source.evidence()
    assert evidence["mode"] == "sumo_traci"
    assert evidence["radio_or_packet_continuity_simulated"] is False
    old_hash = evidence["positions_sha256"]
    source.destinations(0)
    assert source.evidence()["positions_sha256"] != old_hash
    with pytest.raises(ValueError, match="backwards"):
        source.advance(9)
    source.retire(0)
    assert set(removed) == {"muvr_0_0", "muvr_0_1"}
    with pytest.raises(RuntimeError, match="unavailable"):
        source.destinations(0)


def test_teleport_aborts_instead_of_fabricating_vehicle_continuity(tmp_path):
    source, _, _, _, teleports = fake_source(tmp_path)
    teleports.append("muvr_0_0")
    with pytest.raises(RuntimeError, match="teleport"):
        source.advance(5)


def test_pending_departure_uses_route_start_and_zero_speed_until_inserted(tmp_path):
    source, positions, _, _, _ = fake_source(tmp_path)
    source.arrival_interval = 10
    source.connection.vehicle.getAngle = lambda _: 90
    source.connection.vehicle.getSpeed = lambda _: 10
    pending = source._test_pending
    pending.add("muvr_0_0")
    positions.pop("muvr_0_0")

    source.advance(0)
    assert source.destinations(0) == [1, 2]
    queued = source.observation(0)["features"][0]
    assert queued[2] == 0
    assert queued[3] == pytest.approx(0)
    assert queued[4] == pytest.approx(1)
    assert source.positions[-2]["pending_departure"] is True

    pending.remove("muvr_0_0")
    positions["muvr_0_0"] = (20, 0)
    source.advance(1)
    source.destinations(0)
    inserted = source.observation(0)["features"][0]
    assert inserted[2] == pytest.approx(10 / 60)
    assert source.positions[-2]["pending_departure"] is False


def test_retired_pending_vehicle_is_removed_after_later_insertion(tmp_path):
    source, positions, _, removed, _ = fake_source(tmp_path)
    pending = source._test_pending
    pending.add("muvr_0_0")
    positions.pop("muvr_0_0")
    source.advance(0)
    source.retire(0)
    assert "muvr_0_0" in source._pending_departures

    pending.remove("muvr_0_0")
    positions["muvr_0_0"] = (20, 0)
    source.advance(1)
    assert "muvr_0_0" not in source.present
    assert "muvr_0_0" in removed
    assert not source._pending_departures


def test_observation_uses_previous_tick_even_when_getter_is_skipped_and_is_idempotent(tmp_path):
    source, positions, _, _, _ = fake_source(tmp_path)
    source.arrival_interval = 10
    source.connection.vehicle.getAngle = lambda _: 90
    source.connection.vehicle.getSpeed = lambda _: 10
    source.advance(0)
    source.destinations(0)
    source.observation(0)
    positions["muvr_0_0"] = (20, 0)
    source.advance(1)
    source.destinations(0)  # No orchestration decision on this tick.
    positions["muvr_0_0"] = (30, 0)
    source.advance(2)
    source.destinations(0)
    first = source.observation(0)
    second = source.observation(0)
    np.testing.assert_array_equal(first["features"], second["features"])
    assert first["features"][0][5] == pytest.approx(0.1)

"""Opt-in SUMO mobility; spatial association is a simulator assumption, not radio."""

import hashlib
import json
import math
import random
import shutil
import uuid
from pathlib import Path


class SumoMobility:
    def __init__(
        self,
        net_file,
        seed,
        servers,
        requests,
        players,
        arrival_interval,
        lifetime,
        binary="sumo",
        step_s=1.0,
    ):
        if step_s <= 0 or not math.isfinite(step_s):
            raise ValueError("Mobility step must be finite and positive")
        if not math.isclose(arrival_interval / step_s, round(arrival_interval / step_s)):
            raise ValueError("Mobility step must divide the arrival interval")
        if not shutil.which(binary):
            raise FileNotFoundError(f"Eclipse SUMO executable not found: {binary}")
        import sumolib
        import traci

        self.binary = binary
        self.traci = traci
        self.net_file = Path(net_file).resolve()
        self.seed, self.step_s, self.players = seed, step_s, players
        self.connection = None
        self.label = f"muvr-{uuid.uuid4().hex}"
        road = sumolib.net.readNet(str(self.net_file))
        self.road = road
        xmin, ymin, xmax, ymax = road.getBoundary()
        width = math.ceil(math.sqrt(servers))
        height = math.ceil(servers / width)
        self.anchors = {
            node + 1: (
                xmin + (node % width + 0.5) * (xmax - xmin) / width,
                ymin + (node // width + 0.5) * (ymax - ymin) / height,
            )
            for node in range(servers)
        }
        import networkx as nx

        transitions = nx.DiGraph()
        for edge in road.getEdges():
            if edge.allows("passenger"):
                transitions.add_node(edge.getID())
                for target in edge.getAllowedOutgoing("passenger"):
                    if target.allows("passenger"):
                        transitions.add_edge(edge.getID(), target.getID())
        component = max(nx.strongly_connected_components(transitions), key=len)
        rng = random.Random(seed)
        eligible = sorted(
            (
                e
                for e in road.getEdges()
                if e.allows("passenger")
                and e.getID() in component
                and e.getAllowedOutgoing("passenger")
            ),
            key=lambda e: e.getID(),
        )
        if not eligible:
            raise ValueError("SUMO network has no drivable edges")
        self.routes = {}
        # Long random walks, enough even at the SUMO vehicle's maximum speed.
        distance = (requests * arrival_interval + lifetime + 2 * step_s) * 60
        for index in range(requests):
            for player in range(players):
                edge = rng.choice(eligible)
                edges, length = [edge.getID()], edge.getLength()
                while length < distance:
                    options = sorted(
                        (
                            e
                            for e in edge.getAllowedOutgoing("passenger")
                            if e.allows("passenger") and e.getID() in component
                        ),
                        key=lambda e: e.getID(),
                    )
                    if not options:
                        raise ValueError("Generated SUMO route reached a dead end")
                    edge = rng.choice(options)
                    edges.append(edge.getID())
                    length += max(edge.getLength(), 0.1)
                self.routes[f"muvr_{index}_{player}"] = edges
        self.arrival_interval = arrival_interval
        self.positions = []
        self.time_s = 0.0
        self._pending_departures = set()

    def __enter__(self):
        self.traci.start(
            [
                self.binary,
                "--net-file",
                str(self.net_file),
                "--seed",
                str(self.seed),
                "--step-length",
                str(self.step_s),
                "--no-step-log",
                "true",
                "--time-to-teleport",
                "-1",
            ],
            label=self.label,
            doSwitch=False,
        )
        self.connection = self.traci.getConnection(self.label)
        try:
            for vehicle, edges in self.routes.items():
                index = int(vehicle.split("_")[1])
                self.connection.route.add(vehicle, edges)
                self.connection.vehicle.add(
                    vehicle,
                    vehicle,
                    depart=str(index * self.arrival_interval),
                    departLane="best",
                    departSpeed="max",
                )
            self.advance(0.0)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def advance(self, time_s):
        if time_s < self.time_s:
            raise ValueError("SUMO clock cannot move backwards")
        # One step offset materializes departures at virtual time zero.
        self.connection.simulationStep(time_s + self.step_s)
        self.time_s = time_s
        self.present = set(self.connection.vehicle.getIDList())
        inserted_retired = self._pending_departures & self.present
        for vehicle in inserted_retired:
            self.connection.vehicle.remove(vehicle)
            self.present.remove(vehicle)
        self._pending_departures.difference_update(inserted_retired)
        teleports = self.connection.simulation.getStartingTeleportIDList()
        if teleports:
            raise RuntimeError(f"Unexpected SUMO teleport: {teleports}")

    def destinations(self, index):
        result = []
        pending = set(self.connection.simulation.getPendingVehicles())
        for player in range(self.players):
            vehicle = f"muvr_{index}_{player}"
            queued = vehicle in pending
            if vehicle not in self.present and not queued:
                raise RuntimeError(f"SUMO vehicle unavailable at {self.time_s}: {vehicle}")
            if queued:
                shape = self.road.getEdge(self.routes[vehicle][0]).getShape()
                x, y = shape[0]
                dx, dy = shape[1][0] - x, shape[1][1] - y
                angle, speed = math.degrees(math.atan2(dx, dy)) % 360, 0.0
            else:
                x, y = self.connection.vehicle.getPosition(vehicle)
                angle = self.connection.vehicle.getAngle(vehicle)
                speed = self.connection.vehicle.getSpeed(vehicle)
            node = min(
                self.anchors,
                key=lambda n: ((self.anchors[n][0] - x) ** 2 + (self.anchors[n][1] - y) ** 2, n),
            )
            sample = {
                "time_s": self.time_s,
                "vehicle": vehicle,
                "x_m": x,
                "y_m": y,
                "angle_deg": angle,
                "speed_mps": speed,
                "pending_departure": queued,
                "access_node": node,
            }
            kinematics = getattr(self, "_kinematics", {})
            old = kinematics.get(vehicle)
            if old is None:
                kinematics[vehicle] = {
                    "current": sample,
                    "previous": sample,
                    "changed_at": self.time_s,
                }
            elif old["current"]["time_s"] != self.time_s:
                kinematics[vehicle] = {
                    "current": sample,
                    "previous": old["current"],
                    "changed_at": self.time_s
                    if old["current"]["access_node"] != node
                    else old["changed_at"],
                }
            self._kinematics = kinematics
            self.positions.append(sample)
            result.append(node)
        return result

    def observation(self, index):
        """Current/past kinematics only; route and future samples are never features."""
        result = []
        xs, ys = zip(*self.anchors.values(), strict=True)
        width, height = max(max(xs) - min(xs), 1), max(max(ys) - min(ys), 1)
        for player in range(self.players):
            vehicle = f"muvr_{index}_{player}"
            state = getattr(self, "_kinematics", {}).get(vehicle)
            if state is None or state["current"]["time_s"] != self.time_s:
                raise ValueError("Read destinations before requesting mobility observations")
            current, previous = state["current"], state["previous"]
            changed_at = state["changed_at"]
            result.append(
                [
                    (current["x_m"] - min(xs)) / width,
                    (current["y_m"] - min(ys)) / height,
                    current["speed_mps"] / 60,
                    math.sin(math.radians(current["angle_deg"])),
                    math.cos(math.radians(current["angle_deg"])),
                    (current["x_m"] - previous["x_m"]) / width,
                    (current["y_m"] - previous["y_m"]) / height,
                    current["access_node"] / len(self.anchors),
                    previous["access_node"] / len(self.anchors),
                    min((self.time_s - changed_at) / 120, 1),
                    0.0,
                    min((self.time_s - index * self.arrival_interval) / 120, 1),
                ]
            )
        return {"features": result, "time_s": self.time_s}

    def retire(self, index):
        pending = set(self.connection.simulation.getPendingVehicles())
        for player in range(self.players):
            vehicle = f"muvr_{index}_{player}"
            if vehicle in self.present:
                self.connection.vehicle.remove(vehicle)
                self.present.remove(vehicle)
            elif vehicle in pending:
                self._pending_departures.add(vehicle)

    def evidence(self):
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

        return {
            "mode": "sumo_traci",
            "seed": self.seed,
            "step_s": self.step_s,
            "sumo_time_offset_s": self.step_s,
            "sumo_version": self.connection.getVersion(),
            "net_file": str(self.net_file),
            "net_sha256": hashlib.sha256(self.net_file.read_bytes()).hexdigest(),
            "routes_sha256": digest(self.routes),
            "anchors": self.anchors,
            "anchors_sha256": digest(self.anchors),
            "positions": self.positions,
            "positions_sha256": digest(self.positions),
            "source_policy": "original fixed source; mobile per-player destination",
            "handover_policy": "make-before-break; failed handover terminates whole session",
            "pending_departure_model": (
                "waiting vehicles stay at their first route-edge start with zero speed "
                "until SUMO inserts them"
            ),
            "radio_or_packet_continuity_simulated": False,
        }

    def __exit__(self, *_):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

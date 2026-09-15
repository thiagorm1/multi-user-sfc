import time
from typing import Any, TypedDict

from loguru import logger

from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC


class SessionInfo(TypedDict):
    sfc_list: list[SFC]
    solution: dict
    duration: float
    timer: float


class SFCStateTracker:
    def __init__(self):
        self.sfcs_tracker: dict[str, SessionInfo] = {}
        self.sfcs_routing_info: dict[str, dict] = {}
        self.sfc_id_duration: dict[str, dict] = {}
        self.crashed_servers: list[str] = []
        self.risk_servers: list[str] = []
        self.counter = 0

    def get_expired_sessions(self, current_time: float) -> list[str]:
        expired_sessions = []
        for sfc_list_id, info in self.sfcs_tracker.items():
            start_time = info.get("timer", 0)
            duration = info.get("duration", 0)
            elapsed_time = current_time - start_time

            if elapsed_time >= duration:
                expired_sessions.append(sfc_list_id)
                logger.debug(
                    f"[EXPIRAÇÃO] Sessão {sfc_list_id} estourou o tempo! "
                    f"(Durou: {elapsed_time:.2f}s / Limite: {duration}s)"
                )

        return expired_sessions

    def cleanup_session_state(self, sfc_list_id: str) -> list[str]:
        sfc_group = self.sfcs_tracker.pop(sfc_list_id, None)
        if not sfc_group:
            return []

        individual_sfc_ids = [sfc.id for sfc in sfc_group["sfc_list"]]

        for sfc_id in individual_sfc_ids:
            self.sfcs_routing_info.pop(sfc_id, None)

        return individual_sfc_ids

    def rebuild_sfcs_for_requeue(
        self, sfc_list: list[SFC], changed_location: bool, new_location: Any
    ) -> tuple[list[SFC], float]:
        if not sfc_list:
            return [], 0.0

        session_id = sfc_list[0].dst_node
        sfcs_tracker_info = self.sfcs_tracker.get(session_id)

        if not sfcs_tracker_info:
            return [], 0.0

        session_start_time = sfcs_tracker_info["timer"]
        original_duration = sfcs_tracker_info["duration"]

        time_elapsed_absolute = time.time() - session_start_time
        remaining_duration = original_duration - time_elapsed_absolute

        if remaining_duration <= 1.0:
            return [], remaining_duration

        new_sfc_list_dicts = []

        for sfc in sfc_list:
            sfc_id = sfc.id
            location = new_location if changed_location else getattr(sfc, "closer_router", None)
            
            # OTIMIZAÇÃO: Cópia rasa super rápida nativa em C, evitando o deepcopy custoso
            new_vnfs_list_dict = [vnf.copy() for vnf in sfc.vnfs_dict]

            if changed_location and "cache" in sfc_id:
                old_loc = str(getattr(sfc, "closer_router", ""))
                new_loc = str(new_location)

                ma_old_key = f"MA_region_{old_loc}"
                re_old_key = f"RE_region_{old_loc}"

                if sfc_id in self.sfcs_routing_info:
                    current_sfc_routes = self.sfcs_routing_info[sfc_id]

                    if ma_old_key in current_sfc_routes:
                        ma_new_key = ma_old_key.replace(old_loc, new_loc)
                        new_vnfs_list_dict[1]["name"] = ma_new_key

                    if re_old_key in current_sfc_routes:
                        re_new_key = re_old_key.replace(old_loc, new_loc)
                        new_vnfs_list_dict[2]["name"] = re_new_key

            new_sfc_dict = {
                "name": sfc_id,
                "vnf_list": new_vnfs_list_dict,
                "bandwidth": sfc.input_throughput,
                "src_node": sfc.src.substrate_node,
                "dst_node": getattr(sfc, "dst_node", None),
                "duration": remaining_duration,
                "closer_router": location,
                "latency": getattr(sfc, "latency_request", 10),
            }
            new_sfc_list_dicts.append(new_sfc_dict)

        new_sfcs_objects = [SFCGenerator(d).generate() for d in new_sfc_list_dicts]

        current_enqueue_time = time.time()
        for sfc in new_sfcs_objects:
            sfc.enqueue_time = current_enqueue_time

        return new_sfcs_objects, remaining_duration

    def get_sfc_list(self, sfc_id: str, sb_net: Net2) -> list[SFC]:
        try:
            sfc = sb_net.get_sfc_by_id(sfc_id)
        except ValueError:
            return []

        group_id = getattr(sfc, "dst_node", None)
        if group_id and group_id in self.sfcs_tracker:
            return self.sfcs_tracker[group_id]["sfc_list"]
        return []

    def get_running_players_sessions(self) -> tuple[int, int, int]:
        running_sfcs = self.sfcs_tracker
        players = set()
        sessions = set(running_sfcs.keys())

        for group_id in running_sfcs.keys():
            parts = str(group_id).split("_")
            for p in parts:
                if p.startswith("p") and p[1:].isdigit():
                    players.add(p)
                    break

        return len(running_sfcs), len(players), len(sessions)
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from muar_sfc.controllers.modules.crasher import Crasher
from muar_sfc.controllers.modules.mobility_manager import MobilityManager
from muar_sfc.controllers.modules.sfcs_manager import SFCManager
from muar_sfc.core.net_v2 import Net2
from muar_sfc.utils.manager_results import OutputWritter


class FailureOrchestrator:
    """
    Orquestrador especialista responsável pelo ciclo de vida de falhas e recuperações (SRP).
    """
    def __init__(
        self,
        substrate_network: Net2,
        sfc_manager: SFCManager,
        fail_manager: Crasher,
        mobility_manager: MobilityManager | None,
        output_writter: OutputWritter,
        alg: Any,
        verbose: bool,
        requeue_callback: Callable,
    ):
        self.network = substrate_network
        self.sfc_manager = sfc_manager
        self.fail_manager = fail_manager
        self.mobility_manager = mobility_manager
        self.output_writter = output_writter
        self.alg = alg
        self.verbose = verbose

        self.requeue_callback = requeue_callback

        self.crashs_trials = 0
        self.sfcs_crash_affected: dict[str, Any] = {}

    def server_fail_operation(self) -> tuple[list[str], list]:
        servers_failed = self._trigger_crash()
        if not servers_failed:
            return [], []

        affected_sfc_ids, sfc_failed_nodes_map = self._map_affected_sfcs(servers_failed)
        high_risk, med_risk, low_risk = self._classify_crash_risk(servers_failed, affected_sfc_ids)
        pre_crash_latencies, sfc_owners_map = self._audit_pre_crash(affected_sfc_ids)

        self._crash_servers(servers_failed)

        fallen_sfcs_list, post_crash_latencies = self._recover_sfcs(
            affected_sfc_ids, sfc_failed_nodes_map, pre_crash_latencies, sfc_owners_map
        )

        self._compute_crash_metrics(
            servers_failed, affected_sfc_ids, high_risk, med_risk, low_risk,
            pre_crash_latencies, post_crash_latencies,
        )

        self.sfc_manager.crashed_servers = list(self.fail_manager.nodes_crashed)
        self.crashs_trials += 1

        return servers_failed, fallen_sfcs_list

    def server_recovery_operation(self, nodes_to_recover: list[str]) -> None:
        if not nodes_to_recover:
            return
        recovered_count = 0
        for node in nodes_to_recover:
            if self.fail_manager.recover_specific_node(self.network, node):
                logger.info(f">>> [RECOVERY] Servidor recuperado: {node}")
                recovered_count += 1
        if recovered_count > 0:
            self.sfc_manager.crashed_servers = list(self.fail_manager.nodes_crashed)

    def link_fail_operation(self) -> tuple[Any, list]:
        link_failed = self.fail_manager.activate_link_crasher(self.network)
        if not link_failed:
            logger.info(">>> [CRASHER] Nenhum link disponível para falhar.")
            return None, []

        u, v = link_failed
        logger.warning(f">>> [CRASHER] Link Sorteado na Roleta: {u} <-> {v}")

        affected_sfcs = []
        for sfc_id, routing_info in self.sfc_manager.get_routing_info().items():
            path_broken = False
            for vnf, path in routing_info.items():
                if vnf in ["src", "dst"] or not path:
                    continue
                for i in range(len(path) - 1):
                    if {path[i], path[i + 1]} == {u, v}:
                        path_broken = True
                        break
                if path_broken:
                    break
            if path_broken:
                affected_sfcs.append(sfc_id)

        self.network.set_link_down(u, v)

        fallen_sfcs_objects = []
        affected_groups = set()

        for sfc_id in affected_sfcs:
            try:
                sfc_obj = self.network.get_sfc_by_id(sfc_id)
                affected_groups.add(sfc_obj.dst_node)
            except KeyError as e:
                logger.debug(f"[LINK FAIL] Falha ao obter SFC {sfc_id} para afetar grupo: {e}")

        for group_id in affected_groups:
            session_info = self.sfc_manager.get_session_info(group_id)
            if session_info:
                if self.mobility_manager:
                    self.mobility_manager.mark_vehicle_as_redeploying(group_id)
                sfc_list = session_info.get("sfc_list", [])
                self.requeue_callback(sfc_list, changed_location=False)
                fallen_sfcs_objects.extend(sfc_list)

        return link_failed, fallen_sfcs_objects

    def link_recovery_operation(self, link_tuple) -> None:
        u, v = link_tuple
        self.network.restore_link(u, v)
        logger.info(f">>> [RECOVERY] Link Restaurado: {u} <-> {v}")

    def _trigger_crash(self) -> list[str]:
        servers_failed = self.fail_manager.activate_crasher(self.network, self.sfc_manager, self.alg)
        if servers_failed:
            logger.warning(f">>> [CRASH] Servidores a serem derrubados: {servers_failed}")
        return servers_failed

    def _map_affected_sfcs(self, servers_failed) -> tuple[set, dict]:
        affected_sfc_ids = set()
        sfc_failed_nodes_map = {}
        for server in servers_failed:
            sfcs_in_node = self.network.get_node_sfcs(server)
            for sfc_id in sfcs_in_node:
                if "backup" in sfc_id:
                    continue
                affected_sfc_ids.add(sfc_id)
                sfc_failed_nodes_map.setdefault(sfc_id, []).append(server)
        return affected_sfc_ids, sfc_failed_nodes_map

    def _classify_crash_risk(self, servers_failed, affected_sfc_ids) -> tuple[int, int, int]:
        if not servers_failed:
            return 0, 0, 0
        target_node = servers_failed[0]
        base_id = str(target_node).split(".")[0]
        server_reliability = self.network.get_node_reliability(base_id)
        total_victims = len(affected_sfc_ids)
        high_risk = med_risk = low_risk = 0

        if server_reliability < 0.8666:
            high_risk = total_victims
        elif 0.8666 <= server_reliability <= 0.9333:
            med_risk = total_victims
        else:
            low_risk = total_victims

        bucket_name = 'High' if high_risk else 'Med' if med_risk else 'Low'
        logger.info(f" -> Crash Source Reliability: {server_reliability:.4f} | Impact: {bucket_name} Risk")
        return high_risk, med_risk, low_risk

    def _audit_pre_crash(self, affected_sfc_ids) -> tuple[dict, dict]:
        pre_crash_latencies = {}
        sfc_owners_map = {}
        for sfc_id in affected_sfc_ids:
            try:
                sfc_obj_temp = self.network.get_sfc_by_id(sfc_id)
                sfc_owners_map[sfc_id] = sfc_obj_temp.dst_node
            except KeyError as e:
                logger.debug(f"SFC {sfc_id} não encontrada durante auditoria pré-crash: {e}")

            pre_crash_latencies[sfc_id] = self.network.calculate_sfc_total_latency(sfc_id)
        return pre_crash_latencies, sfc_owners_map

    def _crash_servers(self, servers_failed):
        for server in servers_failed:
            self.network.set_node_down(server)

    def _get_real_risk_with_backups(self, sfc_id: str) -> str:
        # EAFP: Resgate de backups
        try:
            backups_dict = self.sfc_manager.backup_manager.sfcs_backups_instatiated
        except AttributeError:
            backups_dict = {}

        if sfc_id not in self.network.sfc_route_info:
            return "Medium"

        route_info = self.network.sfc_route_info[sfc_id]
        backup_reliability_map = {}
        
        if sfc_id in backups_dict:
            for b in backups_dict[sfc_id]:
                vnf_id = b.get("vnf_id")
                bk_route = b.get("route_info", {})
                bk_node_list = next((v for k, v in bk_route.items() if k.endswith("_b") and v), None)
                if vnf_id and bk_node_list:
                    backup_reliability_map[vnf_id] = self.network.get_node_reliability(bk_node_list[0])

        node_groups = defaultdict(list)
        for vnf_id, path in route_info.items():
            if vnf_id in ["src", "dst"] or not path:
                continue
            node_groups[path[0]].append(vnf_id)

        total_reliability = 1.0
        for node_id, vnfs_list in node_groups.items():
            try:
                rel_primary = self.network.get_node_reliability(node_id)
            except KeyError:
                rel_primary = 1.0

            all_vnfs_protected = True
            prod_fail_backups = 1.0
            for vnf_id in vnfs_list:
                rel_bk = backup_reliability_map.get(vnf_id, 0.0)
                if rel_bk > 0.0:
                    prod_fail_backups *= 1.0 - rel_bk
                else:
                    all_vnfs_protected = False

            if all_vnfs_protected:
                group_rel = 1.0 - ((1.0 - rel_primary) * prod_fail_backups)
            else:
                group_rel = rel_primary
            total_reliability *= group_rel

        if total_reliability < 0.8666: return "High"
        elif 0.8666 <= total_reliability <= 0.9333: return "Medium"
        else: return "Low"

    def _recover_sfcs(
            self, affected_sfc_ids: set, sfc_failed_nodes_map: dict,
            pre_crash_latencies: dict, sfc_owners_map: dict,
        ) -> tuple[list, dict]:
            fallen_sfcs_list = []
            post_crash_latencies = {}
            
            if affected_sfc_ids:
                logger.warning(f">>> [IMPACTO] {len(affected_sfc_ids)} serviços (SFCs) foram atingidos pela queda!")

            for sfc_id in affected_sfc_ids:
                risk_level = self._get_real_risk_with_backups(sfc_id)
                failed_nodes = sfc_failed_nodes_map.get(sfc_id, [])
                relevant_server_down = failed_nodes[0] if failed_nodes else None

                try:
                    sfc_obj = self.network.get_sfc_by_id(sfc_id)
                    base_route_info = self.sfc_manager.get_routing_info(sfc_id)
                    old_route_info = {k: list(v) for k, v in base_route_info.items()}
                except (KeyError, ValueError) as e:
                    logger.debug(f"Ignorando recuperação de SFC {sfc_id}: {e}")
                    continue

                affected_vnf_id = None
                if old_route_info:
                    for vnf, path in old_route_info.items():
                        if (vnf not in ["src", "dst"] and "virt" not in vnf) and path and path[0] == relevant_server_down:
                            affected_vnf_id = vnf
                            break

                has_viable_backup = False
                try:
                    aux = self.sfc_manager.backup_manager.sfcs_backups_instatiated
                except AttributeError:
                    aux = {}
                    
                if affected_vnf_id and sfc_id in aux:
                    for backup_entry in aux[sfc_id]:
                        if backup_entry["vnf_id"].replace("_b", "") == affected_vnf_id:
                            has_viable_backup = True
                            break

                if not has_viable_backup:
                    if self.verbose and "backup" not in sfc_id:
                        logger.warning(f" ⚠️ [FALHA/FILA] SFC {sfc_id} perdeu a VNF '{affected_vnf_id}' e NÃO possuía backup. Enviando para re-deploy...")
                    
                    self.sfcs_crash_affected[sfc_id] = {
                        "fall_time": time.time(), "old_latency": pre_crash_latencies.get(sfc_id, 0),
                        "resource_info": 0, "backup_success": False, "crash_trial": self.crashs_trials,
                        "recover_success": False, "risk_level": risk_level, "final_status": "Processing",
                    }
                    tracker_id = sfc_owners_map.get(sfc_id)
                    if tracker_id:
                        session_info = self.sfc_manager.get_session_info(tracker_id)
                        if session_info:
                            if self.mobility_manager:
                                self.mobility_manager.mark_vehicle_as_redeploying(tracker_id)
                            sfc_list_tracker = session_info.get("sfc_list", [])
                            if sfc_list_tracker:
                                self.requeue_callback(sfc_list_tracker, changed_location=False)
                                fallen_sfcs_list.extend(sfc_list_tracker)
                    continue

                try:
                    self.network.undeploy_specific_vnf_context(sfc_id, affected_vnf_id)
                except (KeyError, RuntimeError) as e:
                    logger.error(f"Erro CRÍTICO no undeploy pré-recuperação da SFC {sfc_id}: {e}")
                    raise RuntimeError(f"Corrupção de grafo ao remover VNF {affected_vnf_id} da SFC {sfc_id}") from e

                recovery_start_time = time.time()
                
                try:
                    recovered = self.sfc_manager.reconstruct_and_redeploy(
                        sfc_obj, relevant_server_down, old_route_info, self.network
                    )
                except (KeyError, ValueError, RuntimeError) as e:
                    logger.error(f"Falha na tentativa de stitching da SFC {sfc_id}: {e}")
                    recovered = False

                if recovered:
                    logger.success(f" ✔️ [RECUPERADO] SFC {sfc_id} salva quase instantaneamente via backup da VNF '{affected_vnf_id}'!")
                    new_lat = self.network.calculate_sfc_total_latency(sfc_id)
                    post_crash_latencies[sfc_id] = new_lat
                    old_lat = pre_crash_latencies.get(sfc_id, 0)
                    info_log = {
                        "crash_trial": self.crashs_trials, "recover_success": True, "backup_success": True,
                        "backup_efficient": "Yes", "latency_before": old_lat, "latency_after": new_lat,
                        "latency_diff": new_lat - old_lat, "time_to_recover": time.time() - recovery_start_time,
                        "vnf_id": affected_vnf_id, "latency_degrad": new_lat - old_lat, "resource_degrad": 0,
                        "risk_level": risk_level, "final_status": "Fast Recover",
                    }
                    self.output_writter.resilient_output(sfc_id, info_log, self.crashs_trials)
                else:
                    logger.warning(f" ⚠️ [FALHA/FILA] SFC {sfc_id} possuía backup, mas a costura de rede falhou. Re-enfileirando...")
                    self.sfcs_crash_affected[sfc_id] = {
                        "fall_time": time.time(), "old_latency": pre_crash_latencies.get(sfc_id, 0),
                        "resource_info": 0, "backup_success": False, "crash_trial": self.crashs_trials,
                        "recover_success": False, "risk_level": risk_level, "final_status": "Processing",
                    }
                    tracker_id = sfc_owners_map.get(sfc_id)
                    if tracker_id:
                        self._requeue_session(tracker_id, fallen_sfcs_list)

            return fallen_sfcs_list, post_crash_latencies

    def _requeue_session(self, tracker_id: str, fallen_list: list) -> None:
        session_info = self.sfc_manager.get_session_info(tracker_id)
        if session_info:
            if self.mobility_manager:
                self.mobility_manager.mark_vehicle_as_redeploying(tracker_id)
            sfc_list_tracker = session_info.get("sfc_list", [])
            self.requeue_callback(sfc_list_tracker, changed_location=False)
            fallen_list.extend(sfc_list_tracker)

    def _compute_crash_metrics(self, servers_failed, affected_sfc_ids, high_risk, med_risk, low_risk, pre_crash_latencies, post_crash_latencies):
        total_affected = len(affected_sfc_ids)
        total_active_sfcs = self.network.get_number_active_primary_sfcs()
        avg_lat_before = np.mean(list(pre_crash_latencies.values())) if pre_crash_latencies else 0.0
        avg_lat_after = np.mean(list(post_crash_latencies.values())) if post_crash_latencies else 0.0
        lat_diffs = [
            post_crash_latencies[sfc_id] - pre_crash_latencies[sfc_id]
            for sfc_id in post_crash_latencies if sfc_id in pre_crash_latencies
        ]
        avg_lat_diff = np.mean(lat_diffs) if lat_diffs else 0.0
        affected_pct = (total_affected / total_active_sfcs * 100) if total_active_sfcs > 0 else 0.0

        self.output_writter.output_crash_impact(
            self.crashs_trials, len(servers_failed), total_affected,
            high_risk, med_risk, low_risk, avg_lat_before, avg_lat_after, affected_pct, avg_lat_diff,
        )

    def update_crash_recovery_status(self, sfc_id: str, results_dict: dict | None, is_success: bool, current_time: float) -> None:
        if sfc_id in self.sfcs_crash_affected:
            stored_data = self.sfcs_crash_affected[sfc_id]

            if results_dict:
                stored_data["recover_success"] = is_success

                if is_success:
                    logger.info(f" 🔄 [SLOW RECOVER] SFC {sfc_id} re-implantada fisicamente com sucesso após fila.")
                    latency_diff = results_dict["latency"] - stored_data["old_latency"]

                    stored_data.update({
                        "latency_before": stored_data["old_latency"],
                        "latency_after": results_dict["latency"],
                        "latency_diff": latency_diff,
                        "latency_degrad": latency_diff,
                        "resource_degrad": stored_data["resource_info"] - results_dict.get("resource_info", 0),
                        "time_to_recover": current_time - stored_data["fall_time"],
                        "final_status": "Slow Recover",
                    })
                else:
                    logger.error(f" ❌ [FATALIDADE] SFC {sfc_id} falhou ao ser re-implantada. Recursos esgotados.")
                    stored_data["final_status"] = "Failed"
            else:
                logger.error(f" ❌ [FATALIDADE] SFC {sfc_id} falhou de forma crítica (sem dados de rota).")
                stored_data.update({
                    "recover_success": False,
                    "final_status": "Failed"
                })

            stored_data.setdefault("risk_level", "Medium")
            trial_id = stored_data.get("crash_trial", self.crashs_trials)

            self.output_writter.resilient_output(sfc_id, stored_data, trial_id)

            del self.sfcs_crash_affected[sfc_id]
import time
from typing import Any

from loguru import logger

from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.sfc_state_tracker import SFCStateTracker
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC


class SFCDeployer:
    def __init__(self, tracker: SFCStateTracker, verbose: bool = True):
        self.tracker = tracker
        self.verbose = verbose

    def submit_solution(
        self, sfc_list: list[SFC], solution: dict, substrate_network: Net2, is_backup: bool = False
    ) -> dict[str, Any]:
        deployment_success = True
        route_info_export = {}
        deployed_in_this_batch = []

        for sfc in sfc_list:
            if sfc.id not in solution or solution[sfc.id].get("route_info") is None:
                if self.verbose:
                    logger.warning(f"[Deployer] Solução inválida ou sem rotas para a SFC {sfc.id}.")
                return {"is_success": False, "route_info": None}

        for sfc in sfc_list:
            rf = solution[sfc.id]["route_info"]

            try:
                substrate_network.deploy_sfc(sfc, rf)
                # OTIMIZAÇÃO: Cópia rasa de dicionário de listas substitui o deepcopy custoso
                self.tracker.sfcs_routing_info[sfc.id] = {k: list(v) for k, v in rf.items()}
                route_info_export[sfc.id] = rf
                deployed_in_this_batch.append(sfc.id)

            except Exception as e:
                logger.error(f"[Deployer] Deploy falhou na infraestrutura para {sfc.id}: {e}")
                deployment_success = False
                break

        if not deployment_success:
            for sfc_id in deployed_in_this_batch:
                logger.warning(f"[Rollback] Desfazendo alocação parcial da SFC {sfc_id}")
                try:
                    substrate_network.undeploy_sfc(sfc_id)
                    self.tracker.sfcs_routing_info.pop(sfc_id, None)
                except RuntimeError as rollback_e:
                    logger.exception(f"Falha CRÍTICA ao aplicar rollback na SFC {sfc_id}: {rollback_e}")

            return {"is_success": False, "route_info": None}

        if not is_backup and deployment_success:
            group_id = getattr(sfc_list[0], "dst_node", None)
            duration = getattr(sfc_list[0], "duration", 100)

            if group_id in self.tracker.sfcs_tracker:
                logger.warning(f"[Deployer] Aviso: SFC Session {group_id} já instanciada. Sobrescrevendo rastreamento.")

            self.tracker.sfcs_tracker[group_id] = {
                "sfc_list": sfc_list,
                "solution": solution,
                "duration": duration,
                "timer": time.time()
            }
            self.tracker.counter += 1

        return {"is_success": deployment_success, "route_info": route_info_export}

    def run_garbage_collection(self, substrate_network: Net2, backup_manager: BackupManager | None) -> None:
        infra_sfc_ids = set(substrate_network.sfc_dict.keys())
        valid_logical_ids = set()

        for session_info in self.tracker.sfcs_tracker.values():
            valid_logical_ids.update(sfc.id for sfc in session_info["sfc_list"])

        if backup_manager:
            valid_logical_ids.update(backup_manager.backups_sfc_instantiated.keys())

        zombie_ids = infra_sfc_ids - valid_logical_ids

        if zombie_ids:
            if self.verbose:
                logger.warning(
                    f"[GC] Inconsistência detectada. Removendo {len(zombie_ids)} "
                    f"SFCs órfãs da rede: {zombie_ids}"
                )

            for z_id in zombie_ids:
                try:
                    substrate_network.undeploy_sfc(z_id)
                except RuntimeError as e:
                    # Fail-Fast: Se o zumbi corromper o grafo, a simulação cai.
                    logger.exception(f"[GC] Erro crítico ao limpar zumbi {z_id} da infraestrutura.")
                    raise

    def safe_network_removal(self, sfc_id: str, substrate_network: Net2) -> None:
        try:
            substrate_network.undeploy_sfc(sfc_id)
            if self.verbose:
                logger.debug(f"[RECURSOS LIBERADOS] Instância {sfc_id} purgada da rede física.")
        except RuntimeError as e:
            logger.exception(f"[FALHA DE DESALOCAÇÃO] Erro CRÍTICO ao tentar remover {sfc_id} da infraestrutura.")
            raise
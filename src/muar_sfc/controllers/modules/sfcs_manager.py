from typing import Any

from loguru import logger

from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.sfc_deployer import SFCDeployer
from muar_sfc.controllers.modules.sfc_recovery_service import SFCRecoveryService

# Importando nossos novos sub-módulos refatorados
from muar_sfc.controllers.modules.sfc_state_tracker import SFCStateTracker
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC


class SFCManager:
    """
    Facade (Fachada) que gerencia o ciclo de vida das SFCs.
    Delega as operações complexas para serviços especializados (SRP).
    """

    def __init__(self, args: Any, backup_manager: BackupManager | None, alg: Any):
        self.args = args
        self.backup_manager = backup_manager
        self.alg = alg
        self.alg_name = getattr(alg, "name", str(alg))
        self.verbose = getattr(args, "verbose", True)

        # 1. Instancia o guardião do estado
        self.tracker = SFCStateTracker()

        # 2. Instancia o serviço de deploy (Injetando o estado nele)
        self.deployer = SFCDeployer(self.tracker, self.verbose)

        # 3. Instancia o serviço de recuperação de falhas
        self.recovery_service = SFCRecoveryService(
            self.tracker, self.backup_manager, self.verbose
        )

    # ==========================================
    # Delegação: Core Lifecycle Methods (Deploy)
    # ==========================================

    def submit_solution(self, sfc_list: list[SFC], solution: dict, substrate_network: Net2, is_backup: bool = False) -> dict[str, Any]:
        return self.deployer.submit_solution(sfc_list, solution, substrate_network, is_backup)

    # ==========================================
    # Delegação: Lifecycle Management (Expiration/Cleanup)
    # ==========================================

    def get_expired_sessions(self, current_time: float) -> list[str]:
        return self.tracker.get_expired_sessions(current_time)

    def cleanup_session_state(self, sfc_list_id: str) -> list[str]:
        return self.tracker.cleanup_session_state(sfc_list_id)

    # ==========================================
    # Delegação: Backup & Recovery
    # ==========================================

    def create_backups(self, network: Net2, agent_ref: Any = None) -> tuple[list[Any], str]:
        if self.backup_manager:
            return self.backup_manager.create_backups(network, agent_ref)
        return [], "none"

    def reconstruct_and_redeploy(self, sfc_obj: SFC, crashed_node_id: str, old_route_info: dict, substrate_network: Net2) -> bool:
        return self.recovery_service.reconstruct_and_redeploy(
            sfc_obj, crashed_node_id, old_route_info, substrate_network
        )

    # ==========================================
    # Delegação: Helpers & Requeue
    # ==========================================

    def rebuild_sfcs_for_requeue(self, sfc_list: list[SFC], changed_location: bool, new_location: Any) -> tuple[list[SFC], float]:
        # Como mexe ativamente em manipulação de SFC e Estado, delegamos ao Tracker ou Deployer.
        # Por coerência, colocamos no Tracker (já que lê o tempo/estado) ou em um helper.
        return self.tracker.rebuild_sfcs_for_requeue(sfc_list, changed_location, new_location)

    def run_garbage_collection(self, substrate_network: Net2) -> None:
        self.deployer.run_garbage_collection(substrate_network, self.backup_manager)

    def get_sfc_list(self, sfc_id: str, sb_net: Net2) -> list[SFC]:
        return self.tracker.get_sfc_list(sfc_id, sb_net)

    def get_running_players_sessions(self) -> tuple[int, int, int]:
        return self.tracker.get_running_players_sessions()

    def deploy_success(self, sfc: SFC) -> None:
        if self.verbose: logger.info(f"Deploy succeed: {sfc.id}")

    def deploy_failed(self, sfc: SFC) -> None:
        if self.verbose: logger.warning(f"Deploy FAILED: {sfc.id}")

    def cleanup_network_resources(self, substrate_network: Net2, current_time: float) -> None:
        """Remove instâncias expiradas e lixo da rede física orquestrando os serviços."""

        # 1. Limpa sessões expiradas
        expired_sessions = self.get_expired_sessions(current_time)
        for session_id in expired_sessions:
            logger.debug(f"[DESALOCAÇÃO] Iniciando limpeza da sessão expirada: {session_id}")
            sfc_ids = self.cleanup_session_state(session_id)

            for sfc_id in sfc_ids:
                # ---> CORREÇÃO VITAL: Exterminar os backups Zumbis antes da SFC primária sumir <---
                if self.backup_manager and sfc_id in self.backup_manager.sfcs_backups_instatiated:
                    backups_list = list(self.backup_manager.sfcs_backups_instatiated[sfc_id])
                    for backup_entry in backups_list:
                        bkp_id = backup_entry["sfc_backup_id"]
                        
                        # Removemos o backup fisicamente da rede
                        self.deployer.safe_network_removal(bkp_id, substrate_network)
                        # Limpamos a existência dele do registro lógico
                        self.backup_manager.cleanup_internal_state(bkp_id)

                # DELEGAÇÃO PURA: O Deployer remove a SFC primária
                self.deployer.safe_network_removal(sfc_id, substrate_network)

        # 2. Limpa Backups probabilísticos obsoletos (se a estratégia GA/Vegeta estiver ativada)
        if self.backup_manager:
            obsolete_backups = self.backup_manager.identify_obsolete_backups()
            for backup_id in obsolete_backups:
                self.deployer.safe_network_removal(backup_id, substrate_network)
                # Garante que o estado interno do manager também esqueça o backup
                self.backup_manager.cleanup_internal_state(backup_id)

    def is_session_active(self, session_id: str) -> bool:
        """Verifica de forma segura se uma sessão (SFC) já está ativa na rede."""
        return session_id in self.tracker.sfcs_tracker


    # ==========================================
    # Delegação: Contratos de Estado (Leitura Segura)
    # ==========================================

    def get_routing_info(self, sfc_id: str | None = None) -> dict | Any:
        """Fornece leitura segura para o dicionário de rotas sem expor a raiz mutável."""
        if sfc_id:
            return self.tracker.sfcs_routing_info.get(sfc_id)
        return self.tracker.sfcs_routing_info

    def get_session_info(self, session_id: str | None = None) -> dict | Any:
        """Fornece leitura segura para as informações de uma sessão específica."""
        if session_id:
            return self.tracker.sfcs_tracker.get(session_id)
        return self.tracker.sfcs_tracker

import copy

from loguru import logger

from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.sfc_state_tracker import SFCStateTracker
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC


class SFCRecoveryService:
    """
    Serviço especializado na Recuperação de Desastres (Disaster Recovery).
    
    Responsabilidade Única (SRP): Executar o 'Stitching' (costura de rotas) e
    o Atomic Swap (troca atômica de recursos) entre as instâncias físicas caídas
    e os backups instanciados na rede, atualizando o estado global em caso de sucesso.
    """

    def __init__(
        self,
        tracker: SFCStateTracker,
        backup_manager: BackupManager | None,
        verbose: bool = True
    ):
        # Inversão de Controle: Recebemos o estado e o gerenciador de backup prontos
        self.tracker = tracker
        self.backup_manager = backup_manager
        self.verbose = verbose

    def reconstruct_and_redeploy(
        self, sfc_obj: SFC, crashed_node_id: str, old_route_info: dict, substrate_network: Net2
    ) -> bool:
        """
        Executa a recuperação de falha (Stitching) ativando um backup existente.
        Substitui instâncias caídas de forma atômica e limpa os rastros físicos.
        """
        # ==========================================
        # 1. VERIFICAÇÃO DE DISPONIBILIDADE
        # ==========================================
        if not self.backup_manager or sfc_obj.id not in self.backup_manager.sfcs_backups_instatiated:
            if self.verbose:
                logger.warning(f"[STITCH-FAIL] {sfc_obj.id}: Nenhum backup instanciado no BackupManager.")
            return False

        # ==========================================
        # 2. IDENTIFICAÇÃO DO RECURSO CORROMPIDO
        # ==========================================
        affected_vnf_id = None
        for vnf_id, path in old_route_info.items():
            if vnf_id in ["src", "dst"] or not path:
                continue
            if path[0] == crashed_node_id:
                affected_vnf_id = vnf_id
                break

        if not affected_vnf_id:
            return False

        # ==========================================
        # 3. BUSCA DA RÉPLICA ESPECÍFICA DE BACKUP
        # ==========================================
        backups_list = self.backup_manager.sfcs_backups_instatiated[sfc_obj.id]
        target_backup = None

        for backup_entry in backups_list:
            b_vnf_clean = backup_entry["vnf_id"].replace("_b", "")
            if b_vnf_clean == affected_vnf_id:
                target_backup = backup_entry
                break

        if not target_backup:
            return False

        # ==========================================
        # 4. PREPARAÇÃO DA NOVA ROTA E STITCHING
        # ==========================================
        new_route_info = copy.deepcopy(old_route_info)
        backup_route = target_backup["route_info"]
        backup_sfc_id = target_backup["sfc_backup_id"]

        key_ingress = "src_virt"
        key_egress = None
        for k in backup_route:
            if k.endswith("_b") and k not in ("src_virt", "dst_virt"):
                key_egress = k
                break

        if not key_egress:
            return False

        path_ingress = backup_route.get(key_ingress)
        path_egress = backup_route.get(key_egress)

        if not path_ingress or not path_egress:
            return False

        # Verifica a saúde do nó de backup antes de transferir a carga
        backup_node = path_egress[0]
        backup_node_obj = substrate_network.graph.nodes[backup_node]
        if not backup_node_obj.get("is_active", True):
            if self.verbose:
                logger.warning(f"[STITCH-FAIL] Nó de backup {backup_node} também falhou no ecossistema.")
            return False

        # ==========================================
        # 5. EXECUÇÃO CRÍTICA: ATOMIC SWAP
        # ==========================================
        affected_vnf_obj = sfc_obj.get_vnf_by_id(affected_vnf_id)
        bw_in = affected_vnf_obj.get_income_interface_bandwidth()
        bw_out = affected_vnf_obj.get_outcome_interface_bandwidth()

        # Ativação agressiva da banda nos caminhos de backup
        if not substrate_network.activate_backup_path_bandwidth(path_ingress, bw_in, sfc_obj.id):
            return False
        if not substrate_network.activate_backup_path_bandwidth(path_egress, bw_out, sfc_obj.id):
            return False

        try:
            backup_sfc_obj = substrate_network.get_sfc_by_id(backup_sfc_id)
            backup_vnf_obj = backup_sfc_obj.get_vnf_by_id(key_egress)

            # A) Desaloca fisicamente o backup isolado
            substrate_network.deallocate_microservice(backup_node, backup_sfc_id, backup_vnf_obj)
            substrate_network.detach_vnf_from_route_record(backup_sfc_id, key_egress)

            # B) Aloca a VNF original no novo local limpo
            substrate_network.allocate_microservice(sfc_obj, affected_vnf_obj, backup_node)

        except ValueError as e:
            if self.verbose:
                logger.error(f"[STITCH-FAIL] Falha crítica na troca de recursos no nível da rede: {e}")
            return False

        # ==========================================
        # 6. ATUALIZAÇÃO DO ESTADO GLOBAL
        # ==========================================
        prev_vnf = sfc_obj.get_previous_vnf(affected_vnf_obj)
        if prev_vnf.id == "src":
            new_route_info["src"] = path_ingress
        else:
            new_route_info[prev_vnf.id] = path_ingress

        new_route_info[affected_vnf_id] = path_egress

        # Atualizando no guardião de estado rastreador (Tracker)
        self.tracker.sfcs_routing_info[sfc_obj.id] = new_route_info

        if hasattr(substrate_network, "sfc_route_info"):
            substrate_network.sfc_route_info[sfc_obj.id] = copy.deepcopy(new_route_info)

        if sfc_obj.id not in substrate_network.sfc_dict:
            substrate_network.sfc_dict[sfc_obj.id] = sfc_obj

        # ==========================================
        # 7. LIMPEZA PÓS-RECUPERAÇÃO E SINCRONIA
        # ==========================================
        try:
            # Abordagem de mitigação e limpeza do lixo residual (Garbage Collection ativo)
            substrate_network.undeploy_sfc(backup_sfc_id)
            self.backup_manager.cleanup_internal_state(backup_sfc_id)

            if backup_sfc_id not in self.backup_manager.backups_activated:
                self.backup_manager.backups_activated.append(backup_sfc_id)

        except Exception as e:
            logger.warning(f"[STITCH-CLEANUP] Aviso na limpeza residual do backup {backup_sfc_id}: {e}")

        if self.verbose:
            logger.success(
                f"[STITCH-SUCCESS] SFC {sfc_obj.id}: VNF '{affected_vnf_id}' recuperada "
                f"com sucesso no nó de backup {backup_node}."
            )

        return True

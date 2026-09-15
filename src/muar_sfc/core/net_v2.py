import re
from collections import defaultdict
from contextlib import suppress
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger

from muar_sfc.core.infrastructure.enums import SHAREABLE_PREFIXES, NodeLevel
from muar_sfc.core.infrastructure.physics import TelecomPhysics
from muar_sfc.core.infrastructure.resource_allocator import ResourceAllocator
from muar_sfc.core.infrastructure.topology import TopologyManager
from muar_sfc.core.network_metrics import NetworkMetrics
from muar_sfc.core.vnf import VNF

# =========================================================================
# FUNÇÕES E CONSTANTES GLOBAIS
# =========================================================================

def extrair_sessao(s: str) -> str | None:
    """Extrai o ID da sessão de uma string de SFC de forma segura usando regex."""
    match = re.search(r"p\d+_(\d+)(?:_|$)", s)
    return match.group(1) if match else None

# =========================================================================
# CLASSE PRINCIPAL NET2 (Facade / Fachada)
# =========================================================================

class Net2:
    def __init__(self) -> None:
        # --- 1. Inicializando a Infraestrutura Moderna ---
        self.metrics = NetworkMetrics()
        self.topology = TopologyManager()
        self.allocator = ResourceAllocator(self.topology, self.metrics)

        # --- 2. Ponte de Retrocompatibilidade (Magia do Strangler Fig) ---
        self.graph = self.topology.graph
        self.md_graph = self.topology.md_graph

        # --- Estruturas de Dados de Controle e Roteamento ---
        self.sfc_dict: dict[str, Any] = {}
        self.sfc_route_info: dict[str, dict] = {}
        self.nodes_reliability: dict[str, float] = {}

        self.alpha_stress: dict[str, float] = {NodeLevel.DEFAULT.value: 0.0, NodeLevel.A.value: 0.0, NodeLevel.B.value: 0.0, NodeLevel.C.value: 0.0}
        self.tier_reliability: dict[str, float] = {NodeLevel.DEFAULT.value: 1.0, NodeLevel.A.value: 1.0, NodeLevel.B.value: 1.0, NodeLevel.C.value: 1.0}

        self.processing_delay_info: list[Any] = []
        self.shareable_sf_sfc: dict[str, Any] = {}
        self.sf_route_info: dict[str, Any] = {}
        self.sfs_flux_info: dict[str, Any] = {}
        self.shared_sfs: dict[str, Any] = {}
        self.single_source_minimum_latency_path: dict | None = None

        # --- Flags de Configuração ---
        self.shareable_band: bool = False
        self.shareable_node: bool = True
        self.verbose: bool = False

        # --- Capacidades Totais da Infraestrutura ---
        self.total_cpu_capacity: float = 0.00
        self.total_gpu_capacity: float = 0.00
        self.total_cache_capacity: float = 0.00
        self.total_bandwidth_capacity: float = 0.00

    # =========================================================================
    # 1. DELEGAÇÃO DE INFRAESTRUTURA (FACADE)
    # =========================================================================

    def add_node(self, *args, **kwargs) -> None:
        self.topology.add_node(*args, **kwargs)

    def add_edge(self, *args, **kwargs) -> None:
        self.topology.add_edge(*args, **kwargs)

    def remove_node(self, *args, **kwargs) -> None:
        self.topology.remove_node(*args, **kwargs)

    def connect_mobile_user(self, *args, **kwargs) -> None:
        self.topology.connect_mobile_user(*args, **kwargs)

    def allocate_microservice(self, sfc, vnf, node_id):
        self.allocator.allocate_microservice(sfc, vnf, node_id)

    def deallocate_microservice(self, node_id: str, sfc_id: str, vnf: VNF) -> None:
        self.allocator.deallocate_microservice(node_id, sfc_id, vnf)

    def allocate_bandwidth(self, node1, node2, bw_required, ms_name, is_backup=False):
        self.allocator.allocate_bandwidth(node1, node2, bw_required, ms_name, is_backup)
        return self.get_link_latency(node1, node2)

    def release_bandwidth(self, node1, node2, ms_name):
        self.allocator.release_bandwidth(node1, node2, ms_name)

    def allocate_wireless_bandwidth(self, node1, node2, bw_required, ms_name):
        self.allocator.allocate_wireless_bandwidth(node1, node2, bw_required, ms_name)
        return 0

    def release_wireless_bandwidth(self, node1, node2, ms_name):
        self.allocator.release_wireless_bandwidth(node1, node2, ms_name)

    def set_sharing_params(self, share_arg: str | bool) -> None:
        self.shareable_node = str(share_arg).lower() == "y"
        self.allocator.shareable_node = self.shareable_node

    # =========================================================================
    # 2. CICLO DE VIDA DE SFC (DEPLOY E UNDEPLOY LÓGICO)
    # =========================================================================

    def detach_vnf_from_route_record(self, sfc_id: str, vnf_name: str) -> None:
        if sfc_id in self.sfc_route_info and vnf_name in self.sfc_route_info[sfc_id]:
            del self.sfc_route_info[sfc_id][vnf_name]

    def deploy_sfc(self, sfc, route_info):
        sfc_id = sfc.id
        is_backup_sfc = getattr(sfc, "is_backup", False)

        if sfc_id not in self.sfc_dict:
            self.sfc_dict[sfc_id] = sfc
        if sfc_id not in self.sfc_route_info:
            self.sfc_route_info[sfc_id] = route_info

        for ms_name, path in route_info.items():
            if ms_name in ["src", "dst"] or "virt" in ms_name:
                continue
            vnf = sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]

            self.allocate_microservice(sfc, vnf, node_allocated)
            bw_req = vnf.get_outcome_interface_bandwidth()

            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:], strict=True):
                    if isinstance(v, str):
                        self.allocate_wireless_bandwidth(u, v, bw_req, ms_name)
                    elif isinstance(u, str):
                        self.allocate_wireless_bandwidth(v, u, bw_req, ms_name)
                    else:
                        self.allocate_bandwidth(u, v, bw_req, ms_name, is_backup=is_backup_sfc)
        return True

    def undeploy_sfc(self, sfc_id):
            if sfc_id not in self.sfc_dict:
                return

            try:
                sfc = self.sfc_dict[sfc_id]
                route_info = self.sfc_route_info.get(sfc_id, {})

                for ms_name, path in list(route_info.items()):
                    if ms_name in ["src", "dst"] or "virt" in ms_name or not path:
                        continue

                    node_allocated = path[0]
                    node_exists = (node_allocated in self.graph) or (node_allocated in self.md_graph)

                    if node_exists:
                        # FAIL-FAST: Erros de desalocação não são ignorados.
                        vnf = sfc.get_vnf_by_id(ms_name)
                        self.deallocate_microservice(node_allocated, sfc_id, vnf)

                    if len(path) > 1:
                        self._safe_release_bandwidth(path, ms_name)

            except (KeyError, ValueError) as critical_error:
                if self.verbose: 
                    logger.error(f"Erro crítico (corrupção de grafo) durante undeploy de {sfc_id}: {critical_error}")
                raise RuntimeError(f"O grafo físico dessincronizou ao tentar remover {sfc_id}.") from critical_error
            finally:
                self._remove_sfc_from_registry(sfc_id)

    def _safe_release_bandwidth(self, path, ms_name):
        try:
            for u, v in zip(path[:-1], path[1:], strict=True):
                u_exists = u in self.graph or u in self.md_graph
                v_exists = v in self.graph or v in self.md_graph

                if u_exists and v_exists:
                    if isinstance(v, str): self.release_wireless_bandwidth(u, v, ms_name)
                    elif isinstance(u, str): self.release_wireless_bandwidth(v, u, ms_name)
                    else: self.release_bandwidth(u, v, ms_name)
        except ValueError:
            pass

    def _remove_sfc_from_registry(self, sfc_id):
        if sfc_id in self.sfc_dict: del self.sfc_dict[sfc_id]
        if sfc_id in self.sfc_route_info: del self.sfc_route_info[sfc_id]

        self.metrics.total_cpu_used = max(0.0, self.metrics.total_cpu_used)
        self.metrics.total_cache_used = max(0.0, self.metrics.total_cache_used)
        self.metrics.total_bandwidth_used = max(0.0, self.metrics.total_bandwidth_used)

    def undeploy_specific_vnf_context(self, sfc_id, vnf_id_to_remove):
            if sfc_id not in self.sfc_dict or not self.sfc_route_info.get(sfc_id):
                return False

            sfc = self.sfc_dict[sfc_id]
            route_info = self.sfc_route_info[sfc_id]
            vnf_target = sfc.get_vnf_by_id(vnf_id_to_remove)
            if not vnf_target: return False

            vnf_prev = sfc.get_previous_vnf(vnf_target)

            if vnf_id_to_remove in route_info:
                path = route_info[vnf_id_to_remove]
                if path:
                    # FAIL-FAST: Sem a supressão cega. Se der erro no recurso físico, alertamos.
                    self.deallocate_microservice(path[0], sfc_id, vnf_target)
                    
                    if len(path) > 1:
                        self._safe_release_bandwidth(path, vnf_id_to_remove)

                del route_info[vnf_id_to_remove]

                if vnf_prev and vnf_prev.id != "src" and vnf_prev.id in route_info:
                    path_prev = route_info[vnf_prev.id]
                    if len(path_prev) > 1:
                        self._safe_release_bandwidth(path_prev, vnf_prev.id)
                    if path_prev:
                        route_info[vnf_prev.id] = [path_prev[0]]

            return True

    # =========================================================================
    # 3. FÍSICA E CÁLCULO DE LATÊNCIA
    # =========================================================================

    def calculate_computational_latency(self, graph: nx.Graph, node: str, vnf: VNF) -> float:
        try:
            ips = self.md_graph.nodes[node]["ips"] if self.is_mobile_node(node) else self.graph.nodes[node]["ips"]
        except KeyError:
            return float('inf')

        packet = (vnf.get_income_interface_bandwidth() / 60) * 1e6
        return (packet * 10 * 1000) / ips if ips > 0 else float('inf')

    def calculate_latency_betwen_nodes(self, graph: nx.Graph, node1: str, node2: str, vnf: VNF) -> float:
        data_packet = (vnf.get_outcome_interface_bandwidth() / 60) * 1e6
        if self.is_mobile_node(node1):
            return TelecomPhysics.calculate_5g_latency(data_packet, self.md_graph.nodes[node1]["position"])
        elif self.is_mobile_node(node2):
            return TelecomPhysics.calculate_5g_latency(data_packet, self.md_graph.nodes[node2]["position"])
        return self.get_link_latency(node1, node2)

    def calculate_average_sfc_latency(self) -> float:
        if not self.sfc_dict: return 0.0
        return sum(self.calculate_sfc_total_latency(s_id) for s_id in self.sfc_dict) / len(self.sfc_dict)

    def calculate_sfc_total_latency(self, sfc_id: str) -> float:
        if sfc_id not in self.sfc_dict or sfc_id not in self.sfc_route_info: return 0.0

        sfc = self.sfc_dict[sfc_id]
        route_info = self.sfc_route_info[sfc_id]
        total_latency = 0.0

        current_vnf = sfc.get_vnf_by_id("src")
        while current_vnf and current_vnf.id != "dst":
            next_vnf = sfc.get_next_vnf(current_vnf)
            if not next_vnf or next_vnf.id == "dst": break

            allocation_path = route_info.get(next_vnf.id)
            if not allocation_path:
                current_vnf = next_vnf
                continue

            allocated_node = allocation_path[0]
            ips = self.md_graph.nodes[allocated_node]["ips"] if self.is_mobile_node(allocated_node) else self.graph.nodes[allocated_node]["ips"]

            packet = (next_vnf.get_income_interface_bandwidth() / 60) * 1e6
            total_latency += (packet * 10 * 1000) / ips if ips > 0 else float('inf')

            if len(allocation_path) > 1:
                for u, v in zip(allocation_path[:-1], allocation_path[1:]):
                    total_latency += self.calculate_latency_betwen_nodes(self.graph, u, v, next_vnf)

            current_vnf = next_vnf
        return total_latency

    # =========================================================================
    # 4. SIMULAÇÃO DE FALHAS (A SER EXTRAÍDO NO FUTURO)
    # =========================================================================

    def calculate_average_system_reliability(self, backups_dict: dict[str, list[dict]] | None = None) -> float:
        if not self.sfc_dict: return 0.0

        total_reliability_sum = 0.0
        active_primary_sfcs_count = 0
        backups_dict = backups_dict or {}

        for sfc_id, sfc in self.sfc_dict.items():
            if "backup" in sfc_id or getattr(sfc, "is_backup", False) or sfc_id not in self.sfc_route_info: continue

            route_info = self.sfc_route_info[sfc_id]
            vnf_backup_reliability_map = {}

            if sfc_id in backups_dict:
                for backup_entry in backups_dict[sfc_id]:
                    vnf_target_id = backup_entry.get("vnf_id")
                    bk_route = backup_entry.get("route_info", {})
                    bk_node_list = next((v for k, v in bk_route.items() if k.endswith("_b") and v), None)
                    if vnf_target_id and bk_node_list:
                        vnf_backup_reliability_map[vnf_target_id] = self.get_node_reliability(bk_node_list[0])

            node_groups = defaultdict(list)
            for vnf_id, path in route_info.items():
                if vnf_id in ["src", "dst"] or not path: continue
                node_groups[path[0]].append(vnf_id)

            sfc_reliability = 1.0
            for primary_node, vnfs_list in node_groups.items():
                try: r_primary = self.get_node_reliability(primary_node)
                except (KeyError, AttributeError): r_primary = 1.0

                all_vnfs_protected = True
                prob_backups_fail_combined = 1.0

                for vnf_id in vnfs_list:
                    r_backup = vnf_backup_reliability_map.get(vnf_id, 0.0)
                    if r_backup > 0.0: prob_backups_fail_combined *= (1.0 - r_backup)
                    else: all_vnfs_protected = False

                stage_reliability = 1.0 - ((1.0 - r_primary) * prob_backups_fail_combined) if all_vnfs_protected else r_primary
                sfc_reliability *= stage_reliability

            total_reliability_sum += sfc_reliability
            active_primary_sfcs_count += 1

        return (total_reliability_sum / active_primary_sfcs_count) if active_primary_sfcs_count > 0 else 0.0

    def set_node_down(self, node_id: str) -> None:
        try:
            node = self.graph.nodes[node_id]
        except KeyError:
            raise ValueError(f"Nó {node_id} inexistente.")

        node["is_active"] = False
        node.setdefault("original_cpu_capacity", node.get("cpu_capacity", 0.0))
        node.setdefault("original_cache_capacity", node.get("cache_capacity", 0.0))
        node["cpu_capacity"] = 0.0
        node["cache_capacity"] = 0.0

    def restore_node(self, node_id: str, cpu_capacity: float | None = None, cache_capacity: float | None = None) -> None:
        try:
            node = self.graph.nodes[node_id]
        except KeyError:
            raise ValueError(f"Nó {node_id} inexistente.")

        node["is_active"] = True
        node["cpu_capacity"] = cpu_capacity if cpu_capacity is not None else node.get("original_cpu_capacity", 100.0)
        node["cache_capacity"] = cache_capacity if cache_capacity is not None else node.get("original_cache_capacity", 100.0)
        node.pop("original_cpu_capacity", None)
        node.pop("original_cache_capacity", None)

    def set_link_down(self, u: str | int, v: str | int) -> None:
        try:
            edge = self.graph.edges[u, v]
        except KeyError:
            raise ValueError(f"Link {u}-{v} inexistente.")

        edge.setdefault("original_bw", edge.get("bandwidth_capacity", 1000.0))
        edge.setdefault("original_lat", edge.get("latency", 1.0))
        edge["bandwidth_capacity"] = 0.0
        edge["latency"] = float("inf")

    def restore_link(self, u: str | int, v: str | int) -> None:
        try:
            edge = self.graph.edges[u, v]
            if "original_bw" in edge:
                edge["bandwidth_capacity"] = edge["original_bw"]
                edge["latency"] = edge["original_lat"]
                del edge["original_bw"]
                del edge["original_lat"]
        except KeyError:
            pass

    def activate_backup_path_bandwidth(self, path: list[str | int], bw_required: float, vnf_id_backup: str) -> bool:
        for u, v in zip(path[:-1], path[1:], strict=True):
            if not self.graph.has_edge(u, v): continue
            edge = self.graph.edges[u, v]

            if edge["bandwidth_used"] + bw_required > edge["bandwidth_capacity"]:
                return False

            edge["bandwidth_used"] += bw_required
            self.metrics.total_bandwidth_used += bw_required
            edge.setdefault("services_in_transit", {}).setdefault(vnf_id_backup, {"copys": 0, "bw_used": 0.0})
            edge["services_in_transit"][vnf_id_backup]["copys"] += 1
            edge["services_in_transit"][vnf_id_backup]["bw_used"] += bw_required
        return True

    def set_reliability_params(self, args: Any) -> None:
        self.alpha_stress = {
            NodeLevel.C.value: max(0.0, getattr(args, "stress_high", 0.0009)),
            NodeLevel.B.value: max(0.0, getattr(args, "stress_normal", 0.04)),
            NodeLevel.A.value: max(0.0, getattr(args, "stress_low", 0.15)),
            NodeLevel.DEFAULT.value: max(0.0, getattr(args, "stress_normal", 0.04)),
        }
        def validate(val: float) -> float: return val if 0.0 <= val <= 1.0 else 0.99
        self.tier_reliability = {
            NodeLevel.C.value: validate(getattr(args, "rel_high", 0.9999)),
            NodeLevel.B.value: validate(getattr(args, "rel_normal", 0.99)),
            NodeLevel.A.value: validate(getattr(args, "rel_low", 0.95)),
            NodeLevel.DEFAULT.value: validate(getattr(args, "rel_normal", 0.99)),
        }

    def get_node_reliability(self, node_id: str | int) -> float:
        node_id_str = str(node_id)
        base_id, gpu_id = (node_id_str.replace(".1", ""), node_id_str) if node_id_str.endswith(".1") else (node_id_str, node_id_str + ".1")

        def get_part_data(nid):
            if nid in self.graph and self.graph.nodes[nid].get("is_active", True):
                node = self.graph.nodes[nid]
                cap = node.get("cpu_capacity", 0.0) or node.get("original_cpu_capacity", 1.0) or 1.0
                return float(node.get("cpu_used", 0.0)), float(cap), node.get("level_server", NodeLevel.DEFAULT.value), True
            return 0.0, 0.0, None, False

        try: used_cpu, cap_cpu, lvl_cpu, active_cpu = get_part_data(int(base_id))
        except ValueError: used_cpu, cap_cpu, lvl_cpu, active_cpu = get_part_data(base_id)

        try: used_gpu, cap_gpu, lvl_gpu, active_gpu = get_part_data(float(gpu_id))
        except ValueError: used_gpu, cap_gpu, lvl_gpu, active_gpu = get_part_data(gpu_id)

        if not active_cpu and not active_gpu: return 0.0

        server_level = lvl_cpu or lvl_gpu or NodeLevel.DEFAULT.value
        total_capacity = cap_cpu + cap_gpu
        if total_capacity <= 0: return 0.0

        level_key = str(server_level).lower()
        base_r = self.tier_reliability.get(level_key, self.tier_reliability[NodeLevel.DEFAULT.value])
        alpha = self.alpha_stress.get(level_key, self.alpha_stress[NodeLevel.DEFAULT.value])

        return max(0.0, base_r - ((used_cpu + used_gpu) / total_capacity) * alpha)


    # =========================================================================
    # ALGORITMOS DE CAMINHO MÍNIMO E ROTEAMENTO
    # =========================================================================

    def get_shortest_path_length(self, source, target):
        try:
            return nx.dijkstra_path_length(self.graph, source, target, weight="latency")
        except nx.NetworkXNoPath:
            return float("inf")

    def get_shortest_path(self, source, target):
        try:
            return nx.dijkstra_path(self.graph, source, target, weight="latency")
        except nx.NetworkXNoPath:
            return []

    def get_single_source_minimum_latency_path(self, src):
        return nx.single_source_dijkstra_path(self.graph, src, weight="latency")

    def pre_get_single_source_minimum_latency_path(self):
        single_source_minimum_latency_path = {}
        for node in self.graph.nodes():
            single_source_minimum_latency_path[node] = nx.single_source_dijkstra(
                self.graph, source=node, cutoff=None, weight="latency"
            )
        self.single_source_minimum_latency_path = single_source_minimum_latency_path
        return single_source_minimum_latency_path

    # =========================================================================
    # 5. GETTERS (EAFP Aplicado) E MÉTODOS AUXILIARES
    # =========================================================================

    def get_link_bandwidth_used(self, node1, node2):
        try:
            return self.graph.edges[node1, node2]["bandwidth_used"]
        except KeyError:
            raise ValueError("Aresta inexistente.")

    def get_link_bandwidth_free(self, node1, node2):
        try:
            edge = self.graph.edges[node1, node2]
            return edge["bandwidth_capacity"] - edge["bandwidth_used"]
        except KeyError:
            raise ValueError("Aresta inexistente.")

    def get_link_bandwidth_capacity(self, node1, node2):
        try:
            return self.graph.edges[node1, node2]["bandwidth_capacity"]
        except KeyError:
            raise ValueError("Aresta inexistente.")

    def get_link_info(self, node1, node2):
        try:
            return self.graph.edges[node1, node2]
        except KeyError:
            raise ValueError("Aresta inexistente.")

    def get_node_sfcs(self, node_id):
        try:
            return self.graph.nodes[node_id].get("sfcs_list", [])
        except KeyError:
            raise ValueError(f"Nó {node_id} inexistente.")

    def get_node_sfc_vnf_list(self, node_id):
        try:
            node_sfcs = self.graph.nodes[node_id].get("sfcs_list", [])
        except KeyError:
            return []

        result = []
        for sfc_id in node_sfcs:
            if sfc_id not in self.sfc_dict: continue
            sfc = self.sfc_dict[sfc_id]
            route_info = self.sfc_route_info.get(sfc_id, {})
            for vnf_id, path in route_info.items():
                if vnf_id in ["src", "dst"] or not path: continue
                if path[0] == node_id:
                    vnf = sfc.get_vnf_by_id(vnf_id)
                    if vnf: result.append((sfc_id, vnf))
        return result

    def get_node_cpu_used(self, node_id):
        try: return self.graph.nodes[node_id]["cpu_used"]
        except KeyError: raise ValueError(f"Nó {node_id} inexistente.")

    def get_node_cpu_free(self, node_id):
        try: return self.graph.nodes[node_id]["cpu_capacity"] - self.graph.nodes[node_id]["cpu_used"]
        except KeyError: raise ValueError(f"Nó {node_id} inexistente.")

    def get_node_cpu_capacity(self, node_id):
        try: return self.graph.nodes[node_id]["cpu_capacity"]
        except KeyError: raise ValueError(f"Nó {node_id} inexistente.")

    def get_node_cache_used(self, node_id):
        try: return self.graph.nodes[node_id]["cache_used"]
        except KeyError: raise ValueError(f"Nó {node_id} inexistente.")

    def get_node_cache_free(self, node_id):
        try: return self.graph.nodes[node_id]["cache_capacity"] - self.graph.nodes[node_id]["cache_used"]
        except KeyError: raise ValueError(f"Nó {node_id} inexistente.")

    def get_node_is_active(self, node_id):
        try: return self.graph.nodes[node_id].get("is_active", True)
        except KeyError: raise ValueError(f"Nó {node_id} inexistente.")

    def get_node_cache_capacity(self, node_id):
        try: return self.graph.nodes[node_id]["cache_capacity"]
        except KeyError: raise ValueError(f"Nó {node_id} inexistente.")

    def get_link_latency(self, node1, node2):
        try: return self.graph.edges[node1, node2]["latency"]
        except KeyError: raise ValueError("Aresta inexistente.")

    def get_shortest_path_with_bw(self, source: str, target: str, required_bw: float) -> list[str] | None:
        if source == target: return [source]
        def filter_edge(u, v):
            if self.graph.has_edge(u, v):
                return (self.graph[u][v]["bandwidth_capacity"] - self.graph[u][v]["bandwidth_used"]) >= required_bw
            return False
        valid_view = nx.subgraph_view(self.graph, filter_edge=filter_edge)
        try: return nx.dijkstra_path(valid_view, source, target, weight="latency")
        except (nx.NetworkXNoPath, nx.NodeNotFound): return None


    # --- MÉTODOS AUXILIARES DE ORQUESTRAÇÃO E ESTADO ---

    def get_shareable_sfs(self):
        return self.shared_sfs

    def get_sfc_by_id(self, sfc_id):
        try:
            return self.sfc_dict[sfc_id]
        except KeyError:
            raise ValueError(f"SFC {sfc_id} não encontrada.")

    def get_number_actives_sfcs(self):
        return len(self.sfc_dict)

    def get_number_active_primary_sfcs(self):
        count = 0
        for sfc_id, sfc_obj in self.sfc_dict.items():
            is_backup_id = "backup" in sfc_id
            is_backup_attr = getattr(sfc_obj, "is_backup", False)

            if not is_backup_id and not is_backup_attr:
                count += 1
        return count

    def get_servers_reliability_dict(self):
        temp_list = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "server":
                if data.get("cpu_capacity", 0) == 0:
                    reliability = 0.0
                else:
                    reliability = self.get_node_reliability(node_id)
                temp_list.append((node_id, reliability))

        temp_list.sort(key=lambda x: x[1])
        return {node_id: rel for node_id, rel in temp_list}

    # =========================================================================
    # 6. RELATÓRIOS E MÉTRICAS (A SER EXTRAÍDO NO FUTURO)
    # =========================================================================

    def get_total_gpu_request(self): return self.metrics.total_gpu_requested
    def get_total_gpu_saved(self): return self.metrics.total_gpu_saved
    def get_total_cpu_request(self): return self.metrics.total_cpu_requested
    def get_total_cpu_saved(self): return self.metrics.total_cpu_saved
    def get_total_cache_request(self): return self.metrics.total_cache_requested
    def get_total_cache_saved(self): return self.metrics.total_cache_saved
    def get_cpu_network_used(self): return self.metrics.total_cpu_used
    def get_gpu_network_used(self): return self.metrics.total_gpu_used
    def get_cpu_total_used(self): return self.metrics.total_cpu_used + self.metrics.mobile_cpu_used
    def get_gpu_total_used(self): return self.metrics.total_gpu_used + self.metrics.mobile_gpu_used
    def get_cache_total_used(self): return self.metrics.total_cache_used + self.metrics.mobile_cache_used
    def get_cache_used(self): return self.metrics.total_cache_used

    def get_total_system_cpu_capacity(self):
        total = sum(d["cpu_capacity"] for n, d in self.graph.nodes(data=True) if "cpu_capacity" in d and not self._is_gpu_node(n))
        total += sum(d["cpu_capacity"] for n, d in self.md_graph.nodes(data=True) if "cpu_capacity" in d and not self._is_gpu_node(n))
        return total

    def get_total_system_gpu_capacity(self):
        total = sum(d["cpu_capacity"] for n, d in self.graph.nodes(data=True) if "cpu_capacity" in d and self._is_gpu_node(n))
        total += sum(d["cpu_capacity"] for n, d in self.md_graph.nodes(data=True) if "cpu_capacity" in d and self._is_gpu_node(n))
        return total

    def get_server_cpu_utilization_rate(self):
        return self.metrics.total_cpu_used / self.total_cpu_capacity if self.total_cpu_capacity > 0 else 0.0

    def get_total_system_utilization_cpu_rate(self):
        cap = self.get_total_system_cpu_capacity()
        return (self.metrics.total_cpu_used + self.metrics.mobile_cpu_used) / cap if cap > 0 else 0.0

    def get_total_system_utilization_gpu_rate(self):
        cap = self.get_total_system_gpu_capacity()
        return (self.metrics.total_gpu_used + self.metrics.mobile_gpu_used) / cap if cap > 0 else 0.0

    def get_total_system_utilization_cache_rate(self):
        return (self.metrics.total_cache_used + self.metrics.mobile_cache_used) / self.total_cache_capacity if self.total_cache_capacity > 0 else 0.0

    def get_bandwidth_utilization_rate(self):
        return self.metrics.total_bandwidth_used * 1.0 / self.total_bandwidth_capacity if self.total_bandwidth_capacity > 0 else 0.0

    def get_network_cpu_utilization_percentage(self):
        cap = sum(d.get("original_cpu_capacity", d.get("cpu_capacity", 0.0)) for n, d in self.graph.nodes(data=True) if not self._is_gpu_node(n))
        return (self.metrics.total_cpu_used / cap) * 100 if cap > 0 else 0.0

    def get_network_gpu_utilization_percentage(self):
        cap = sum(d.get("original_cpu_capacity", d.get("cpu_capacity", 0.0)) for n, d in self.graph.nodes(data=True) if self._is_gpu_node(n))
        return (self.metrics.total_gpu_used / cap) * 100 if cap > 0 else 0.0

    def get_network_cache_utilization_percentage(self):
        cap = sum(d.get("cache_capacity", 0.0) for n, d in self.graph.nodes(data=True))
        return (self.metrics.total_cache_used / cap) * 100 if cap > 0 else 0.0

    def calculate_jain_fairness(self, utilizations):
        if not utilizations or sum(x * x for x in utilizations) == 0: return 1.0
        return (sum(utilizations)**2) / (len(utilizations) * sum(x * x for x in utilizations))

    def get_cpu_jain_fairness(self):
        utils = [d["cpu_used"] / d["cpu_capacity"] for graph in (self.graph, self.md_graph) for n, d in graph.nodes(data=True) if not self._is_gpu_node(n) and d.get("cpu_capacity", 0) > 0]
        return self.calculate_jain_fairness(utils)

    def get_gpu_jain_fairness(self):
        utils = [d["cpu_used"] / d["cpu_capacity"] for graph in (self.graph, self.md_graph) for n, d in graph.nodes(data=True) if self._is_gpu_node(n) and d.get("cpu_capacity", 0) > 0]
        return self.calculate_jain_fairness(utils)

    def get_cache_jain_fairness(self):
        utils = [d["cache_used"] / d["cache_capacity"] for graph in (self.graph, self.md_graph) for n, d in graph.nodes(data=True) if d.get("cache_capacity", 0) > 0]
        return self.calculate_jain_fairness(utils)

    def get_bandwidth_jain_fairness(self):
        utils = [d["bandwidth_used"] / d["bandwidth_capacity"] for u, v, d in self.graph.edges(data=True) if d.get("bandwidth_capacity", 0) > 0]
        utils += [d["w_channel_used"] / d["w_channel_capacity"] for n, d in self.graph.nodes(data=True) if d.get("type") == "router" and d.get("w_channel_capacity", 0) > 0]
        return self.calculate_jain_fairness(utils)

    def get_acceptance_rate(self, success_arr):
        return np.mean(success_arr) * 100 if len(success_arr) > 0 else 0.0

    def is_shareable(self, service_name):
        return service_name.startswith(SHAREABLE_PREFIXES) if self.shareable_node else False

    def is_mobile_node(self, node):
        return node in self.md_graph

    def _is_gpu_node(self, node_id):
        return str(node_id).endswith(".1")

    def update(self):
        pass

    def get_mobile_cpu_utilization_percentage(self):
        cap = sum(d.get("cpu_capacity", 0.0) for n, d in self.md_graph.nodes(data=True) if not self._is_gpu_node(n))
        return (self.metrics.mobile_cpu_used / cap) * 100 if cap > 0 else 0.0

    def get_mobile_gpu_utilization_percentage(self):
        cap = sum(d.get("cpu_capacity", 0.0) for n, d in self.md_graph.nodes(data=True) if self._is_gpu_node(n))
        return (self.metrics.mobile_gpu_used / cap) * 100 if cap > 0 else 0.0

    def get_mobile_cache_utilization_percentage(self):
        cap = sum(d.get("cache_capacity", 0.0) for n, d in self.md_graph.nodes(data=True))
        return (self.metrics.mobile_cache_used / cap) * 100 if cap > 0 else 0.0

    def get_total_system_processing_utilization_rate(self):
        total_processing_used = self.get_cpu_total_used() + self.get_gpu_total_used()
        total_processing_capacity = (
            self.get_total_system_cpu_capacity() + self.get_total_system_gpu_capacity()
        )
        if total_processing_capacity == 0:
            return 0.0
        return total_processing_used / total_processing_capacity

    def get_cache_utilization_rate(self):
        if self.total_cache_capacity == 0:
            return 0.0
        return self.metrics.total_cache_used / self.total_cache_capacity
    
    def get_network_only_processing_utilization(self) -> float:
        """
        Calcula a taxa de utilização (0.0 a 1.0) do processamento (CPU + GPU)
        exclusivamente da infraestrutura de rede (servidores físicos), 
        ignorando os dispositivos móveis.
        """
        # Extração em tempo linear e segura (usando o EAFP nativo com .get)
        cap_cpu = sum(
            d.get("original_cpu_capacity", d.get("cpu_capacity", 0.0)) 
            for n, d in self.graph.nodes(data=True) if not self._is_gpu_node(n)
        )
        cap_gpu = sum(
            d.get("original_cpu_capacity", d.get("cpu_capacity", 0.0)) 
            for n, d in self.graph.nodes(data=True) if self._is_gpu_node(n)
        )

        total_network_capacity = cap_cpu + cap_gpu

        # Fail-Fast matemático: previne ZeroDivisionError caso o grafo esteja vazio
        if total_network_capacity <= 0:
            return 0.0

        total_network_used = self.metrics.total_cpu_used + self.metrics.total_gpu_used
        
        return total_network_used / total_network_capacity

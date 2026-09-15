import re
from typing import Any

from muar_sfc.core.infrastructure.enums import SHAREABLE_PREFIXES
from muar_sfc.core.infrastructure.topology import TopologyManager
from muar_sfc.core.network_metrics import NetworkMetrics
from muar_sfc.core.vnf import VNF


def extrair_sessao(s: str) -> str | None:
    match = re.search(r"p\d+_(\d+)(?:_|$)", s)
    return match.group(1) if match else None

class ResourceAllocator:
    """
    Motor de Infraestrutura responsável exclusivamente por deduzir e liberar 
    capacidades físicas (CPU, Banda, Cache) da Topologia.
    """
    def __init__(self, topology: TopologyManager, metrics: NetworkMetrics) -> None:
        self.topology = topology
        self.metrics = metrics
        self.shareable_node: bool = True

    def _is_shareable(self, service_name: str) -> bool:
        if not self.shareable_node:
            return False
        return service_name.startswith(SHAREABLE_PREFIXES)

    def _is_gpu_node(self, node_id: str | int) -> bool:
        return str(node_id).endswith(".1")

    def allocate_microservice(self, sfc: Any, vnf: VNF, node_id: str | int) -> None:
        sfc_id = sfc.id
        session = getattr(sfc, "session_id", extrair_sessao(sfc_id) or sfc_id.split("_")[-1])

        node = self.topology.get_node(node_id)

        if not node.get("is_active", True):
            raise ValueError(f"Falha crítica: nó {node_id} inativo para alocação.")

        service_id = vnf.id
        cpu_required = vnf.get_cpu_request()
        cache_required = vnf.get_cache_request()
        is_gpu = self._is_gpu_node(node_id)
        is_mobile = node.get("type") == "mobile_device"

        if is_gpu:
            self.metrics.total_gpu_requested = round(self.metrics.total_gpu_requested + cpu_required, 2)
        else:
            self.metrics.total_cpu_requested = round(self.metrics.total_cpu_requested + cpu_required, 2)
        self.metrics.total_cache_requested = round(self.metrics.total_cache_requested + cache_required, 2)

        def put_resource(cost_cpu: float, cost_cache: float):
            node["cpu_used"] = round(node["cpu_used"] + cost_cpu, 2)
            node["cache_used"] = round(node["cache_used"] + cost_cache, 2)

            if is_gpu:
                if is_mobile: self.metrics.mobile_gpu_used += cost_cpu
                else: self.metrics.total_gpu_used += cost_cpu
            else:
                if is_mobile: self.metrics.mobile_cpu_used += cost_cpu
                else: self.metrics.total_cpu_used += cost_cpu

            if is_mobile: self.metrics.mobile_cache_used += cost_cache
            else: self.metrics.total_cache_used += cost_cache

        clean_current_id = service_id.replace("_b", "")
        compatible_instance_found = False

        if self._is_shareable(service_id) or self._is_shareable(clean_current_id):
            for existing_id, existing_session in node.get("services", {}):
                if existing_id.replace("_b", "") == clean_current_id and existing_session == session:
                    compatible_instance_found = True
                    break

        if sfc_id not in node.setdefault("sfcs_list", []):
            node["sfcs_list"].append(sfc_id)

        service_key = (service_id, session)
        services_dict = node.setdefault("services", {})

        if service_key in services_dict:
            services_dict[service_key]["copys"] += 1

            if not self._is_shareable(service_id):
                if (node["cpu_used"] + cpu_required > node["cpu_capacity"] or
                    node["cache_used"] + cache_required > node["cache_capacity"]):
                    services_dict[service_key]["copys"] -= 1
                    node["sfcs_list"].remove(sfc_id)
                    raise ValueError(f"Sem capacidade no nó {node_id} para instância não-shared.")
                put_resource(cpu_required, cache_required)
            else:
                if is_gpu: self.metrics.total_gpu_saved += cpu_required
                else: self.metrics.total_cpu_saved += cpu_required
                self.metrics.total_cache_saved += cache_required
                self.metrics.shared_vnfs_count += 1

        else:
            cost_cpu = 0 if compatible_instance_found else cpu_required
            cost_cache = 0 if compatible_instance_found else cache_required

            if compatible_instance_found:
                if is_gpu: self.metrics.total_gpu_saved += cpu_required
                else: self.metrics.total_cpu_saved += cpu_required
                self.metrics.total_cache_saved += cache_required
                self.metrics.shared_vnfs_count += 1

            if (node["cpu_used"] + cost_cpu > node["cpu_capacity"] or
                node["cache_used"] + cost_cache > node["cache_capacity"]):
                if sfc_id in node["sfcs_list"]:
                    node["sfcs_list"].remove(sfc_id)
                raise ValueError(f"Sem capacidade no nó {node_id} para novo serviço.")

            services_dict[service_key] = {"cpu": cost_cpu, "cache": cost_cache, "copys": 1}
            put_resource(cost_cpu, cost_cache)

            if self._is_shareable(service_id):
                node.setdefault("reuse", []).append(vnf)

    def allocate_bandwidth(self, node1: str | int, node2: str | int, bw_required: float, ms_name: str, is_backup: bool = False) -> None:
        edge = self.topology.get_edge(node1, node2)

        current_reserved = edge.get("bandwidth_reserved", 0.0)
        total_committed = edge.get("bandwidth_used", 0.0) + current_reserved

        services_in_transit = edge.setdefault("services_in_transit", {})
        if ms_name in services_in_transit:
            total_committed -= services_in_transit[ms_name]["bw_used"]

        if total_committed + bw_required > edge.get("bandwidth_capacity", 0.0):
            raise ValueError(f"Banda insuficiente entre {node1} e {node2}.")

        if ms_name in services_in_transit:
            services_in_transit[ms_name]["copys"] += 1
            services_in_transit[ms_name]["bw_used"] += bw_required
        else:
            services_in_transit[ms_name] = {"copys": 1, "bw_used": bw_required, "is_backup": is_backup}

        if is_backup:
            edge["bandwidth_reserved"] = edge.get("bandwidth_reserved", 0.0) + bw_required
        else:
            edge["bandwidth_used"] = edge.get("bandwidth_used", 0.0) + bw_required
            self.metrics.total_bandwidth_used += bw_required

    def allocate_wireless_bandwidth(self, node1: str | int, node2: str | int, bw_required: float, ms_name: str) -> None:
        router = self.topology.get_node(node1)
        w_services = router.setdefault("w_services", {})

        if ms_name in w_services:
            w_services[ms_name]["copys"] += 1
            router["w_channel_used"] += bw_required
            self.metrics.total_bandwidth_used += bw_required
        else:
            if router.get("w_channel_used", 0.0) + bw_required > router.get("w_channel_capacity", 0.0):
                raise ValueError(f"Banda insuficiente entre {node1} e {node2}.")
            w_services[ms_name] = {"copys": 1, "bw_used": bw_required}
            router["w_channel_used"] = router.get("w_channel_used", 0.0) + bw_required
            self.metrics.total_bandwidth_used += bw_required

    def deallocate_microservice(self, node_id: str | int, sfc_id: str, vnf: VNF) -> None:
        try:
            node = self.topology.get_node(node_id)
        except ValueError:
            return

        service_id = vnf.id
        session_id = extrair_sessao(sfc_id)
        service_key = (service_id, session_id)

        if service_key not in node.get("services", {}):
            return

        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        is_gpu_node = self._is_gpu_node(node_id)
        is_mobile = node.get("type") == "mobile_device"

        service_info = node["services"][service_key]
        service_info["copys"] -= 1
        remove_physical_instance = service_info["copys"] <= 0
        is_shareable_service = self._is_shareable(service_id)

        if is_gpu_node:
            self.metrics.total_gpu_requested -= cpu_req
        else:
            self.metrics.total_cpu_requested -= cpu_req
        self.metrics.total_cache_requested -= cache_req

        stored_cache_cost = service_info.get("cache", 0.0)
        stored_cpu_cost = service_info.get("cpu", 0.0)

        was_subsidized_cache = stored_cache_cost < cache_req
        was_subsidized_cpu = stored_cpu_cost < cpu_req

        if not remove_physical_instance or was_subsidized_cache:
            self.metrics.total_cache_saved = max(0.0, self.metrics.total_cache_saved - cache_req)

        if not remove_physical_instance or was_subsidized_cpu:
            if is_gpu_node:
                self.metrics.total_gpu_saved = max(0.0, self.metrics.total_gpu_saved - cpu_req)
            else:
                self.metrics.total_cpu_saved = max(0.0, self.metrics.total_cpu_saved - cpu_req)

            if is_shareable_service:
                self.metrics.shared_vnfs_count = max(0, self.metrics.shared_vnfs_count - 1)

        if sfc_id in node.get("sfcs_list", []):
            node["sfcs_list"].remove(sfc_id)

        if remove_physical_instance:
            del node["services"][service_key]
            node["cpu_used"] = round(node["cpu_used"] - cpu_req, 2)
            node["cache_used"] = round(node["cache_used"] - cache_req, 2)

            if is_gpu_node:
                if is_mobile: self.metrics.mobile_gpu_used -= cpu_req
                else: self.metrics.total_gpu_used -= cpu_req
            else:
                if is_mobile: self.metrics.mobile_cpu_used -= cpu_req
                else: self.metrics.total_cpu_used -= cpu_req

            if is_mobile: self.metrics.mobile_cache_used -= cache_req
            else: self.metrics.total_cache_used -= cache_req

            if is_shareable_service and vnf in node.get("reuse", []):
                node["reuse"].remove(vnf)

    def release_bandwidth(self, node1: str | int, node2: str | int, ms_name: str) -> None:
        try:
            edge = self.topology.get_edge(node1, node2)
        except ValueError:
            return

        services = edge.get("services_in_transit", {})
        if ms_name not in services:
            return

        entry = services[ms_name]
        bw_to_release = entry["bw_used"]
        is_backup_entry = entry.get("is_backup", False)

        entry["copys"] -= 1

        if is_backup_entry:
            edge["bandwidth_reserved"] = max(0.0, edge.get("bandwidth_reserved", 0.0) - bw_to_release)
        else:
            edge["bandwidth_used"] = max(0.0, edge.get("bandwidth_used", 0.0) - bw_to_release)
            self.metrics.total_bandwidth_used = max(0.0, self.metrics.total_bandwidth_used - bw_to_release)

        if entry["copys"] <= 0:
            del services[ms_name]

    def release_wireless_bandwidth(self, node1: str | int, node2: str | int, ms_name: str) -> None:
        try:
            router = self.topology.get_node(node1)
        except ValueError:
            return

        services = router.get("w_services", {})
        if ms_name not in services:
            return

        services[ms_name]["copys"] -= 1
        bw_to_release = services[ms_name]["bw_used"]
        router["w_channel_used"] = max(0.0, router.get("w_channel_used", 0.0) - bw_to_release)

        self.metrics.total_bandwidth_used = max(0.0, self.metrics.total_bandwidth_used - bw_to_release)

        if services[ms_name]["copys"] == 0:
            del services[ms_name]

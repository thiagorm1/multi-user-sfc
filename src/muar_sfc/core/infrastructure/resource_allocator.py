import re
from types import SimpleNamespace

import networkx as nx

from muar_sfc.core.infrastructure.enums import SHAREABLE_PREFIXES
from muar_sfc.core.orchestration.placement import reserve_bandwidth, reserve_node, service_key


def extrair_sessao(s: str) -> str | None:
    match = re.search(r"p\d+_(\d+)(?:_|$)", s)
    return match.group(1) if match else None


class ResourceAllocator:
    """Reserve and release physical resources with symmetric shared reference counts."""

    def __init__(self, topology, metrics):
        self.topology = topology
        self.metrics = metrics
        self.shareable_node = True
        self.shareable_band = False

    def _is_shareable(self, service_name):
        return self.shareable_node and str(service_name).startswith(SHAREABLE_PREFIXES)

    def _is_gpu_node(self, node_id):
        return str(node_id).endswith(".1")

    def _processing_metric(self, node, node_id):
        return ("mobile_" if node.get("type") == "mobile_device" else "total_") + (
            "gpu_used" if self._is_gpu_node(node_id) else "cpu_used"
        )

    def _add_metric(self, field, delta):
        setattr(self.metrics, field, max(0.0, getattr(self.metrics, field) + delta))

    def allocate_microservice(self, sfc, vnf, node_id):
        node = self.topology.get_node(node_id)
        graph = nx.Graph()
        graph.add_node(node_id)
        graph.nodes[node_id].update(node)
        cpu, cache = reserve_node(graph, sfc, vnf, node_id, self.shareable_node)
        node.update(graph.nodes[node_id])
        node.setdefault("sfc_service_keys", {})[sfc.id, vnf.id] = service_key(sfc, vnf)
        if sfc.id not in node.setdefault("sfcs_list", []):
            node["sfcs_list"].append(sfc.id)
        self._add_metric(self._processing_metric(node, node_id), cpu)
        self._add_metric(
            "mobile_cache_used" if node.get("type") == "mobile_device" else "total_cache_used",
            cache,
        )
        processor = "gpu" if self._is_gpu_node(node_id) else "cpu"
        self._add_metric(f"total_{processor}_requested", vnf.get_cpu_request())
        self._add_metric("total_cache_requested", vnf.get_cache_request())
        self._add_metric(f"total_{processor}_saved", vnf.get_cpu_request() - cpu)
        self._add_metric("total_cache_saved", vnf.get_cache_request() - cache)
        if (cpu, cache) != (vnf.get_cpu_request(), vnf.get_cache_request()):
            self.metrics.shared_vnfs_count += 1

    def deallocate_microservice(self, node_id, sfc_id, vnf):
        try:
            node = self.topology.get_node(node_id)
        except ValueError:
            return
        keys = node.get("sfc_service_keys", {})
        key = keys.pop((sfc_id, vnf.id), service_key(SimpleNamespace(id=sfc_id), vnf))
        entry = node.get("services", {}).get(key)
        if entry is None:
            return
        shared = entry.get("shared", False)
        entry["copys"] -= 1
        cpu = (
            entry["cpu"]
            if shared and entry["copys"] == 0
            else (0.0 if shared else entry.get("cpu_per_copy", vnf.get_cpu_request()))
        )
        cache = (
            entry["cache"]
            if shared and entry["copys"] == 0
            else (0.0 if shared else entry.get("cache_per_copy", vnf.get_cache_request()))
        )
        node["cpu_used"] = max(0.0, node["cpu_used"] - cpu)
        node["cache_used"] = max(0.0, node["cache_used"] - cache)
        entry["cpu"] -= cpu
        entry["cache"] -= cache
        if entry["copys"] == 0:
            del node["services"][key]
        if not any(owner == sfc_id for owner, _ in keys) and sfc_id in node.get("sfcs_list", []):
            node["sfcs_list"].remove(sfc_id)
        processor = "gpu" if self._is_gpu_node(node_id) else "cpu"
        self._add_metric(self._processing_metric(node, node_id), -cpu)
        self._add_metric(
            "mobile_cache_used" if node.get("type") == "mobile_device" else "total_cache_used",
            -cache,
        )
        self._add_metric(f"total_{processor}_requested", -vnf.get_cpu_request())
        self._add_metric("total_cache_requested", -vnf.get_cache_request())
        self._add_metric(f"total_{processor}_saved", -(vnf.get_cpu_request() - cpu))
        self._add_metric("total_cache_saved", -(vnf.get_cache_request() - cache))
        if shared and entry["copys"] > 0:
            self.metrics.shared_vnfs_count = max(0, self.metrics.shared_vnfs_count - 1)

    def allocate_bandwidth(self, node1, node2, bw_required, ms_name, is_backup=False):
        edge = self.topology.get_edge(node1, node2)
        cost = reserve_bandwidth(edge, ms_name, bw_required, self.shareable_band, is_backup)
        if not is_backup:
            self._add_metric("total_bandwidth_used", cost)

    def release_bandwidth(self, node1, node2, ms_name):
        try:
            edge = self.topology.get_edge(node1, node2)
        except ValueError:
            return
        self._release(edge, ms_name)

    def _release(self, data, key):
        entry = data.get("services_in_transit", {}).get(key)
        if entry is None:
            return
        entry["copys"] -= 1
        cost = (
            (entry["bw_used"] if entry["copys"] == 0 else 0.0)
            if entry.get("shared", False)
            else entry.get("bw_per_copy", entry["bw_used"])
        )
        entry["bw_used"] = max(0.0, entry["bw_used"] - cost)
        field = "bandwidth_reserved" if entry.get("is_backup", False) else "bandwidth_used"
        data[field] = max(0.0, data.get(field, 0) - cost)
        if field == "bandwidth_used":
            self._add_metric("total_bandwidth_used", -cost)
        if entry["copys"] == 0:
            del data["services_in_transit"][key]

    def _wireless_proxy(self, router):
        return {
            "bandwidth_capacity": router.get("w_channel_capacity", 0),
            "bandwidth_used": router.get("w_channel_used", 0),
            "services_in_transit": router.setdefault("w_services", {}),
        }

    def allocate_wireless_bandwidth(self, node1, node2, bw_required, ms_name):
        router = self.topology.get_node(node1)
        proxy = self._wireless_proxy(router)
        cost = reserve_bandwidth(proxy, ms_name, bw_required, self.shareable_band)
        router["w_channel_used"] = proxy["bandwidth_used"]
        self._add_metric("total_bandwidth_used", cost)

    def release_wireless_bandwidth(self, node1, node2, ms_name):
        try:
            router = self.topology.get_node(node1)
        except ValueError:
            return
        proxy = self._wireless_proxy(router)
        self._release(proxy, ms_name)
        router["w_channel_used"] = proxy["bandwidth_used"]

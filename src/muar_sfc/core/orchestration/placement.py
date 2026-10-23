"""Common resource identities and reservations for planning and deployment."""

import re

import networkx as nx

from muar_sfc.core.infrastructure.enums import SHAREABLE_PREFIXES


def session_id(sfc):
    explicit = getattr(sfc, "session_id", None)
    if explicit is not None:
        return f"backup:{explicit}" if getattr(sfc, "is_backup", False) else str(explicit)
    match = re.search(r"p\d+_(\d+)(?:_|$)", str(sfc.id))
    value = match.group(1) if match else str(sfc.id).split("_")[-1]
    return f"backup:{value}" if getattr(sfc, "is_backup", False) else value


def service_key(sfc, vnf):
    return vnf.id, session_id(sfc)


def is_shareable(vnf):
    return str(vnf.id).startswith(SHAREABLE_PREFIXES)


def node_demand(graph, sfc, vnf, node, share):
    cpu, cache = vnf.get_cpu_request(), vnf.get_cache_request()
    entry = graph.nodes[node].get("services", {}).get(service_key(sfc, vnf))
    if (
        share
        and is_shareable(vnf)
        and entry is not None
        and (
            entry.get("cpu_per_copy", entry.get("cpu")) == cpu
            and entry.get("cache_per_copy", entry.get("cache")) == cache
        )
    ):
        return 0.0, 0.0
    return cpu, cache


def reserve_node(graph, sfc, vnf, node, share):
    data = graph.nodes[node]
    if not data.get("is_active", True) or data.get("type") not in (
        "server",
        "edge_server",
        "server_edge",
        "server_cloud",
        "mobile_device",
    ):
        raise ValueError(f"Nó {node} indisponível para processamento.")
    cpu, cache = node_demand(graph, sfc, vnf, node, share)
    if data.get("cpu_used", 0) + cpu > data.get("cpu_capacity", 0) + 1e-7:
        raise ValueError(f"CPU insuficiente no nó {node}.")
    if data.get("cache_used", 0) + cache > data.get("cache_capacity", 0) + 1e-7:
        raise ValueError(f"Cache insuficiente no nó {node}.")
    key = service_key(sfc, vnf)
    services = data.setdefault("services", {})
    old = services.get(key)
    shared = bool(share and is_shareable(vnf))
    if old and (
        old.get("cpu_per_copy", old.get("cpu")) != vnf.get_cpu_request()
        or old.get("cache_per_copy", old.get("cache")) != vnf.get_cache_request()
    ):
        raise ValueError("Mesmo ID de serviço possui demandas incompatíveis.")
    if old:
        entry = dict(old)
        entry["copys"] += 1
        entry["cpu"] += cpu
        entry["cache"] += cache
    else:
        entry = {
            "copys": 1,
            "cpu": cpu,
            "cache": cache,
            "cpu_per_copy": vnf.get_cpu_request(),
            "cache_per_copy": vnf.get_cache_request(),
            "shared": shared,
        }
    services[key] = entry
    data["cpu_used"] = data.get("cpu_used", 0) + cpu
    data["cache_used"] = data.get("cache_used", 0) + cache
    return cpu, cache


def stream_key(sfc, u, v, shareband):
    # Reuse only explicitly identical logical streams inside the same session.
    source_shared = u == "src" or is_shareable(sfc.get_vnf_by_id(u))
    target_shared = v != "dst" and is_shareable(sfc.get_vnf_by_id(v))
    bandwidth = sfc.graph.edges[u, v]["bandwidth"]
    owner = session_id(sfc) if shareband and source_shared and target_shared else str(sfc.id)
    return f"{owner}:{u}>{v}:{bandwidth:g}"


def bandwidth_demand(data, key, bandwidth, shareband):
    entry = data.get("services_in_transit", {}).get(key)
    if shareband and entry and entry.get("bw_per_copy", entry["bw_used"]) == bandwidth:
        return 0.0
    return bandwidth


def reserve_bandwidth(data, key, bandwidth, shareband, is_backup=False):
    cost = bandwidth_demand(data, key, bandwidth, shareband)
    committed = data.get("bandwidth_used", 0) + data.get("bandwidth_reserved", 0)
    if committed + cost > data.get("bandwidth_capacity", 0) + 1e-7:
        raise ValueError("Enlace sem banda residual suficiente.")
    services = data.setdefault("services_in_transit", {})
    old = services.get(key)
    if old:
        if old.get("bw_per_copy", old["bw_used"]) != bandwidth:
            raise ValueError("Mesmo fluxo possui demandas de banda incompatíveis.")
        entry = dict(old)
        entry["copys"] += 1
        entry["bw_used"] += cost
    else:
        entry = {
            "copys": 1,
            "bw_used": cost,
            "bw_per_copy": bandwidth,
            "shared": bool(shareband),
            "is_backup": is_backup,
        }
    services[key] = entry
    field = "bandwidth_reserved" if is_backup else "bandwidth_used"
    data[field] = data.get(field, 0) + cost
    return cost


def _available_weight(graph, bandwidth, key, shareband):
    def weight(_u, _v, data):
        if not graph.nodes[_u].get("is_active", True) or not graph.nodes[_v].get(
            "is_active", True
        ):
            return None
        cost = bandwidth_demand(data, key, bandwidth, shareband)
        free = (
            data.get("bandwidth_capacity", 0)
            - data.get("bandwidth_used", 0)
            - data.get("bandwidth_reserved", 0)
        )
        if free + 1e-7 < cost:
            return None
        # Prefer residual capacity while retaining propagation latency in the objective.
        pressure = cost / max(1.0, free) if cost else 0
        return data.get("latency", 1.0) + pressure * 0.1

    return weight


def available_paths(graph, source, bandwidth, key, shareband):
    """Exact residual-bandwidth paths for one immutable resource snapshot."""
    try:
        return nx.single_source_dijkstra(
            graph, source, weight=_available_weight(graph, bandwidth, key, shareband)
        )[1]
    except nx.NodeNotFound:
        return {}


def available_path(graph, source, target, bandwidth, key, shareband):
    if source == target:
        return [source] if source in graph else []

    try:
        return nx.dijkstra_path(
            graph, source, target, weight=_available_weight(graph, bandwidth, key, shareband)
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

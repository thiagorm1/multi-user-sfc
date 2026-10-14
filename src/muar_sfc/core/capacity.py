"""Explicit, auditable scaling of the simulated infrastructure."""

import math


def apply_capacity_profile(network, settings):
    if getattr(network, "capacity_profile_applied", False):
        raise ValueError("Perfil de capacidade já aplicado nesta rede.")
    for node, data in network.graph.nodes(data=True):
        if data.get("type") == "server":
            data["cpu_capacity"] *= settings.cpu_capacity_factor
            data["cache_capacity"] *= settings.cache_capacity_factor
            data["ips"] *= settings.cpu_capacity_factor
        if data.get("type") == "router":
            data["w_channel_capacity"] *= settings.wireless_capacity_factor
        for neighbor in network.graph.neighbors(node):
            if node == settings.src_node and settings.source_bandwidth_mbps > 0:
                network.graph.edges[node, neighbor]["bandwidth_capacity"] = (
                    settings.source_bandwidth_mbps / network.graph.degree(node)
                )
    for u, v, data in network.graph.edges(data=True):
        if settings.src_node in (u, v) and settings.source_bandwidth_mbps > 0:
            continue
        if math.isfinite(data["bandwidth_capacity"]):
            data["bandwidth_capacity"] *= settings.link_capacity_factor
    network.total_cpu_capacity = sum(
        d.get("cpu_capacity", 0)
        for _, d in network.graph.nodes(data=True)
        if d.get("type") == "server"
    )
    network.total_cache_capacity = sum(
        d.get("cache_capacity", 0)
        for _, d in network.graph.nodes(data=True)
        if d.get("type") == "server"
    )
    network.total_bandwidth_capacity = sum(
        d["bandwidth_capacity"]
        for _, _, d in network.graph.edges(data=True)
        if math.isfinite(d["bandwidth_capacity"])
    )
    network.capacity_profile_applied = True

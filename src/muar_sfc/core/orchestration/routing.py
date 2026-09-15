from typing import Any

import networkx as nx

from muar_sfc.core.infrastructure.topology import NetworkTopology


class RoutingService:
    """Provedor de algoritmos de roteamento e busca de caminhos."""

    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def get_shortest_path(self, source: Any, target: Any, weight: str = "latency") -> list[Any]:
        """Encontra o menor caminho baseado em um peso (ex: latência)."""
        try:
            return nx.dijkstra_path(self.topology.graph, source, target, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_bw_aware_path(
        self, source: Any, target: Any, required_bw: float
    ) -> list[Any] | None:
        """Busca o menor caminho considerando apenas links com banda disponível.

        Utiliza uma visualização filtrada (Subgraph View) para manter a eficiência[cite: 198].
        """

        def filter_edge(u: Any, v: Any) -> bool:
            edge = self.topology.graph[u][v]
            free_bw = edge["bandwidth_capacity"] - edge["bandwidth_used"]
            return free_bw >= required_bw

        valid_view = nx.subgraph_view(self.topology.graph, filter_edge=filter_edge)
        try:
            return nx.dijkstra_path(valid_view, source, target, weight="latency")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

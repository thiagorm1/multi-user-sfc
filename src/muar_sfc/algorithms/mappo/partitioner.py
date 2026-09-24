import math
from typing import Any

import networkx as nx
import numpy as np


class RegionalTopologyPartitioner:
    """
    Particiona uma topologia de rede substrata em K regiões/clusters de borda e 1 tier de Nuvem.
    Permite aos agentes locais do MAPPO terem escopo e observações delimitadas.
    """

    def __init__(self, graph: nx.Graph, n_regions: int = 4, cloud_node: int | str = 0):
        self.graph = graph
        self.n_regions = max(1, n_regions)
        self.cloud_node = cloud_node
        self.node_to_region: dict[Any, int] = {}
        self.region_to_nodes: dict[int, list[Any]] = {r: [] for r in range(self.n_regions)}
        self.partition_network()

    def partition_network(self) -> None:
        """Executa o particionamento dos nós da topologia."""
        # Se cloud_node fornecido for router ou inexistente, busca o melhor servidor para nuvem
        all_servers = [
            n for n in self.graph.nodes()
            if self.graph.nodes[n].get("type") in ("server", "edge_server")
        ]

        is_invalid_cloud = (
            self.cloud_node not in self.graph
            or self.graph.nodes[self.cloud_node].get("type") == "router"
        )
        if is_invalid_cloud and all_servers:
            c_servers = [n for n in all_servers if self.graph.nodes[n].get("level_server") == "c"]
            self.cloud_node = c_servers[0] if c_servers else max(
                all_servers, key=lambda n: self.graph.nodes[n].get("cpu_capacity", 0)
            )

        valid_nodes = [
            n for n in all_servers
            if n != self.cloud_node
        ]

        if not valid_nodes:
            valid_nodes = all_servers or [n for n in self.graph.nodes() if n != self.cloud_node]

        if not valid_nodes:
            self.node_to_region[self.cloud_node] = 0
            self.region_to_nodes[0] = [self.cloud_node]
            return

        # Verifica se nós possuem posições (x, y)
        has_positions = all("position" in self.graph.nodes[n] for n in valid_nodes)

        if has_positions:
            self._partition_by_coordinates(valid_nodes)
        else:
            self._partition_by_graph_structure(valid_nodes)

        # Associa nó de nuvem como acessível globalmente
        self.node_to_region[self.cloud_node] = -1

    def _partition_by_coordinates(self, nodes: list[Any]) -> None:
        """Agrupamento geográfico via K-Means."""
        coords = np.array([self.graph.nodes[n]["position"] for n in nodes], dtype=np.float32)

        k = min(self.n_regions, len(nodes))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(coords), size=k, replace=False)
        centroids = coords[indices]

        labels = np.zeros(len(nodes), dtype=int)
        for _ in range(20):
            dists = np.linalg.norm(coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for cluster_idx in range(k):
                mask = labels == cluster_idx
                if np.any(mask):
                    centroids[cluster_idx] = coords[mask].mean(axis=0)

        for node, label in zip(nodes, labels, strict=False):
            reg = int(label)
            self.node_to_region[node] = reg
            self.region_to_nodes[reg].append(node)

    def _partition_by_graph_structure(self, nodes: list[Any]) -> None:
        """Particionamento topológico via ordenação de nós."""
        k = min(self.n_regions, len(nodes))
        chunk_size = math.ceil(len(nodes) / k)
        sorted_nodes = sorted(nodes, key=lambda n: str(n))

        for idx, node in enumerate(sorted_nodes):
            reg = min(idx // chunk_size, k - 1)
            self.node_to_region[node] = reg
            self.region_to_nodes[reg].append(node)

    def get_region_for_node(self, node_id: Any) -> int:
        """Retorna o ID da região do nó (-1 para cloud)."""
        return self.node_to_region.get(node_id, -1)

    def get_nodes_in_region(self, region_id: int) -> list[Any]:
        """Retorna os nós pertencentes à região."""
        return self.region_to_nodes.get(region_id, [])

    def get_internal_links(self, region_id: int) -> list[tuple[Any, Any]]:
        """Retorna links cujas duas extremidades estão na mesma região."""
        region_nodes = set(self.get_nodes_in_region(region_id))
        links = []
        for u, v in self.graph.edges():
            if u in region_nodes and v in region_nodes:
                links.append((u, v))
        return links

    def get_border_links(self, region_id: int) -> list[tuple[Any, Any]]:
        """Retorna links conectando a região a outras regiões ou à nuvem."""
        region_nodes = set(self.get_nodes_in_region(region_id))
        links = []
        for u, v in self.graph.edges():
            u_in = u in region_nodes
            v_in = v in region_nodes
            if (u_in and not v_in) or (v_in and not u_in):
                links.append((u, v))
        return links

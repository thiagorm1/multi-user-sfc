import random

import networkx as nx
from loguru import logger

# Assumindo que você criou o enums.py no Passo 1
from muar_sfc.core.infrastructure.enums import NodeLevel, NodeType


class TopologyManager:
    """
    Gerencia exclusivamente a infraestrutura de rede (grafos, nós físicos e roteadores).
    Responsabilidade Única (SRP): Não sabe o que é uma SFC ou um VNF.
    """
    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.md_graph: nx.Graph = nx.Graph()

    def get_node(self, node_id: str | int) -> dict:
        """
        Recupera um nó usando a filosofia EAFP (É mais fácil pedir perdão que permissão).
        Tenta acessar o grafo principal, se falhar, tenta o mobile. Sem if/else verboso.
        """
        try:
            return self.graph.nodes[node_id]
        except KeyError:
            try:
                return self.md_graph.nodes[node_id]
            except KeyError:
                raise ValueError(f"Nó {node_id} inexistente na topologia.")

    def get_edge(self, u: str | int, v: str | int) -> dict:
        try:
            return self.graph.edges[u, v]
        except KeyError:
            raise ValueError(f"Aresta entre {u} e {v} inexistente.")

    def add_node(
        self,
        node_id: str | int,
        node_type: str,
        cpu_capacity: float = 0.00,
        cache_capacity: float = 0.00,
        w_channel_capacity: float = 0.0,
        position: tuple[float, float] = (0, 0),
        ips: float = 0.0,
    ) -> None:
        """Adiciona um nó aplicando Structural Pattern Matching (match/case) no lugar de if/elif."""
        node_type_clean = node_type.split("_")[0] if NodeType.SERVER.value in node_type else node_type

        # Refatoração: Uso do Match/Case introduzido no Python 3.10
        match node_type_clean:
            case NodeType.SERVER.value:
                node_level = node_type.split("_")[-1] if "_" in node_type else NodeLevel.DEFAULT.value
                self.graph.add_node(
                    node_id,
                    type=NodeType.SERVER.value,
                    cpu_capacity=float(cpu_capacity),
                    cache_capacity=float(cache_capacity),
                    cpu_used=0.00,
                    cache_used=0.00,
                    ips=float(ips) * 10e10,
                    position=position,
                    reuse=[],
                    services={},
                    sfcs_list=[],
                    level_server=node_level,
                    is_active=True,
                )

            case NodeType.MOBILE_DEVICE.value:
                sorteio = random.random()
                if sorteio <= 0.33:
                    cpu_capacity *= 0.75
                    cache_capacity *= 0.75
                    ips *= 0.75
                elif sorteio > 0.67:
                    cpu_capacity *= 1.25
                    cache_capacity *= 1.25
                    ips *= 1.25

                self.md_graph.add_node(
                    node_id,
                    type=NodeType.MOBILE_DEVICE.value,
                    cpu_capacity=float(cpu_capacity),
                    cache_capacity=float(cache_capacity),
                    cpu_used=0.00,
                    cache_used=0.00,
                    ips=float(ips) * 10e10,
                    position=position,
                    reuse=[],
                    services={},
                    sfcs_list=[],
                    is_active=True,
                )

            case NodeType.ROUTER.value:
                self.graph.add_node(
                    node_id,
                    type=NodeType.ROUTER.value,
                    cpu_capacity=float(cpu_capacity),
                    cache_capacity=float(cache_capacity),
                    cpu_used=0.00,
                    cache_used=0.00,
                    w_channel_capacity=float(w_channel_capacity),
                    w_channel_used=0.0,
                    position=position,
                    services={},
                    w_services={},
                    is_active=True,
                )

            case _:
                # Padrão irrefutável para terminação compulsória
                raise ValueError(f"Tipo de nó não reconhecido: {node_type_clean}")

    def remove_node(self, node_id: str | int) -> None:
        try:
            self.md_graph.remove_node(node_id)
        except nx.NetworkXError:
            logger.warning(f"Tentativa de remover nó móvel inexistente: {node_id}")

    def add_edge(self, node1: str | int, node2: str | int, bandwidth_capacity: float = 1000.00, latency: float = 1.0) -> None:
        self.graph.add_edge(
            node1, node2,
            bandwidth_capacity=float(bandwidth_capacity),
            bandwidth_used=0.00,
            bandwidth_reserved=0.00,
            latency=float(latency),
            services_in_transit={},
        )

    def connect_mobile_user(self, user_id: str | int, router_id: str | int, cpu_capacity: float) -> None:
        self.add_node(user_id, NodeType.MOBILE_DEVICE.value, cpu_capacity=cpu_capacity)
        self.add_edge(user_id, router_id, bandwidth_capacity=100.0, latency=5.0)

    def set_node_down(self, node_id: str) -> None:
        node = self.get_node(node_id)
        node["is_active"] = False
        node.setdefault("original_cpu_capacity", node.get("cpu_capacity", 0.0))
        node.setdefault("original_cache_capacity", node.get("cache_capacity", 0.0))
        node["cpu_capacity"] = 0.0
        node["cache_capacity"] = 0.0
        logger.warning(f"Simulação de Falha: Nó {node_id} caiu!")

    def restore_node(self, node_id: str, cpu_capacity: float | None = None, cache_capacity: float | None = None) -> None:
        node = self.get_node(node_id)
        node["is_active"] = True
        node["cpu_capacity"] = cpu_capacity if cpu_capacity is not None else node.get("original_cpu_capacity", 100.0)
        node["cache_capacity"] = cache_capacity if cache_capacity is not None else node.get("original_cache_capacity", 100.0)
        node.pop("original_cpu_capacity", None)
        node.pop("original_cache_capacity", None)

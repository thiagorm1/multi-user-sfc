import math
import random

from muar_sfc.core.net_v2 import Net2
from muar_sfc.topology.base.topology_base import TopologyBase

LIGHT_SPEED = 3e8

CLOUD_LATENCY = 1
BANDWIDTH_CAPACITY = 5000.0
W_BANDWIDTH_CAPACITY = 2000.0
CPU_CAPACITY = 100.0 * 1.0
CACHE_CAPACITY = 100.0
RELIABILITY_RANGE = [0.95, 0.99]


class LuxembourgV2(TopologyBase):
    def __init__(self, eco_effi_ratio: float) -> None:
        self.latency = 1
        self.eco_effi_ratio = eco_effi_ratio
        self.bandwidth_capacity = BANDWIDTH_CAPACITY
        self.w_bandwidth_capacity = W_BANDWIDTH_CAPACITY
        self.cpu_capacity = CPU_CAPACITY
        self.cache_capacity = CACHE_CAPACITY
        self.number_of_nodes = 35
        self.nodes = list(range(1, 35))

        self.edge_computing_servers = [25, 8, 14, 28, 2, 5, 9, 23, 18, 6, 33, 34]
        self.labels_nodes = {
            "high_level": [25, 8, 14, 28],
            "normal_level": [2, 5, 9, 23],
            "low_level": [18, 6, 33, 34],
        }

        self.positions = {
            0: (5000, 6000),
            1: (2884, 6739),
            2: (8081, 4302),
            3: (8881, 4995),
            4: (8956, 2900),
            5: (9693, 4040),
            6: (4351, 6749),
            7: (5341, 7590),
            8: (6053, 8420),
            9: (6230, 5831),
            10: (6446, 6969),
            11: (6776, 7725),
            12: (6785, 5631),
            13: (7000, 8517),
            14: (7034, 7203),
            15: (7041, 6100),
            16: (7019, 3541),
            17: (7295, 7026),
            18: (7234, 5147),
            19: (7336, 5733),
            20: (7477, 6206),
            21: (7467, 5394),
            22: (7605, 8238),
            23: (7551, 5914),
            24: (8266, 7723),
            25: (8403, 6861),
            26: (9639, 7772),
            27: (9868, 9199),
            28: (10138, 9166),
            29: (10433, 9612),
            30: (10652, 9036),
            31: (1690, 8623),
            32: (8839, 4462),
            33: (1492, 7139),
            34: (4500, 5500),
        }

        self.topology = [
            (0, 34),
            (14, 17),
            (25, 24),
            (8, 13),
            (8, 22),
            (14, 10),
            (14, 11),
            (28, 27),
            (28, 29),
            (28, 30),
            (2, 3),
            (2, 32),
            (5, 4),
            (9, 12),
            (23, 15),
            (23, 19),
            (23, 20),
            (18, 21),
            (6, 7),
            (2, 16),
            (25, 26),
            (33, 1),
            (33, 31),
            (34, 9),
            (9, 18),
            (18, 2),
            (2, 5),
            (5, 23),
            (23, 25),
            (25, 28),
            (28, 14),
            (14, 8),
            (8, 6),
            (6, 33),
            (33, 34),
        ]

        # Adiciona os nós de eficiência conectados entre si

        for node in self.edge_computing_servers:
            self.topology.append((node, node + 0.1))

        aux = []
        for chave, valor in self.positions.items():
            if chave in self.edge_computing_servers:
                aux.append((chave, valor))
        for tupla in aux:
            chave, valor = tupla
            self.positions[chave + 0.1] = valor

        cont = len(self.edge_computing_servers)
        for i in range(cont):
            self.edge_computing_servers.append(self.edge_computing_servers[i] + 0.1)

        for _chave, valor in self.labels_nodes.items():
            aux = valor[:]
            for node in aux:
                valor.append(node + 0.1)

        for node in self.edge_computing_servers:
            if node not in self.nodes:
                self.nodes.append(node)

    def generate_substrate_network(self):
        net = Net2()

        quantidade_cpu_eco = self.eco_effi_ratio * self.cpu_capacity
        quantidade_cache_eco = self.eco_effi_ratio * self.cache_capacity
        quantidade_cpu_effi = self.cpu_capacity - quantidade_cpu_eco
        quantidade_cache_effi = self.cache_capacity - quantidade_cache_eco

        boost_effi_factor = 17

        # Adiciona os nós
        net.add_node(0, node_type="router", cpu_capacity=0.0)

        for node in self.nodes:
            if node in self.edge_computing_servers:
                # Para o nó de eficiência
                if node in self.labels_nodes["normal_level"]:
                    type_node = "b"
                    level_factor = 1
                elif node in self.labels_nodes["low_level"]:
                    type_node = "a"
                    level_factor = 0.461
                else:
                    type_node = "c"
                    level_factor = 1.846

                if node % 1 == 0:
                    quanti_cpu = quantidade_cpu_eco * level_factor
                    quanti_cache = quantidade_cache_eco * level_factor
                    boost_effi = 1 * level_factor

                else:
                    quanti_cpu = quantidade_cpu_effi * level_factor
                    quanti_cache = quantidade_cache_effi * level_factor
                    boost_effi = 1 * level_factor

                    boost_effi = boost_effi_factor * level_factor

                net.add_node(
                    node,
                    node_type=f"server_{type_node}",
                    cpu_capacity=quanti_cpu,
                    cache_capacity=quanti_cache,
                    position=self.positions[node],
                    ips=random.uniform(0.1, 0.2) * boost_effi,
                )

                net.total_cpu_capacity += quanti_cpu
                net.total_cache_capacity += quanti_cache
                net.nodes_reliability[node] = random.uniform(0.95, 0.99)

            else:
                net.add_node(
                    node,
                    node_type="router",
                    w_channel_capacity=self.w_bandwidth_capacity,
                    position=self.positions[node],
                )

        # Adiciona as arestas com latência baseada na distância euclidiana
        for u, v in self.topology:
            if u == (v + 0.1) or v == (u + 0.1):
                net.add_edge(u, v, bandwidth_capacity=float("inf"), latency=0)
            else:
                latency_ms = random.uniform(1, 2)

                pos_u = self.positions[u]
                pos_v = self.positions[v]
                dist = math.dist(pos_u, pos_v)  # metros
                latency_ms = (dist / LIGHT_SPEED) * 1000  # converte para ms
                latency_ms += round(
                    random.uniform(0, 0.5), 3
                )  # simula pequena variação de latência
                if u == 0 or v == 0:
                    latency_ms *= 100
                net.add_edge(u, v, bandwidth_capacity=self.bandwidth_capacity, latency=latency_ms)

        # net.total_cpu_capacity = len(self.edge_computing_servers) * CPU_CAPACITY
        # net.total_cache_capacity = len(self.edge_computing_servers) * CACHE_CAPACITY
        net.total_bandwidth_capacity = (
            len(self.topology) - len(self.edge_computing_servers)
        ) * BANDWIDTH_CAPACITY
        net.pre_get_single_source_minimum_latency_path()

        return net

    def get_topology_info(self):
        """Retorna informações da topologia."""
        nodes = list(range(1, self.number_of_nodes))
        edges = self.topology  # Lista de arestas
        edge_computing_servers = self.edge_computing_servers  # Nós de processamento

        return {
            "nodes": nodes,
            "edges": edges,
            "ec_servers": edge_computing_servers,
            "routers": [
                node for node in nodes if node not in edge_computing_servers and node != 0
            ],
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx

    lux = LuxembourgV2(eco_effi_ratio=0.8)  # Exemplo de uso da proporção
    net = lux.generate_substrate_network()
    info = lux.get_topology_info()

    print("=== Topologia LuxembourgV2 ===")
    print(f"Nós totais: {lux.number_of_nodes}")
    print(f"Nós da rede (1 a {lux.number_of_nodes - 1}): {info['nodes']}")
    print(f"Servidores de borda (Edge Computing): {info['ec_servers']}")
    print(f"Roteadores: {info['routers']}")
    print(f"Arestas (conexões): {info['edges']}")
    print(f"Capacidade total CPU: {net.total_cpu_capacity}")
    print(f"Capacidade total Cache: {net.total_cache_capacity}")
    print(f"Capacidade total Banda: {net.total_bandwidth_capacity}")

    # Se quiser mostrar as posições dos nós
    print("\nPosições dos nós:")
    for node, pos in lux.positions.items():
        print(f"  Nó {node}: {pos}")

    # Mostrar a topologia graficamente:
    G = nx.Graph()
    G.add_nodes_from(range(lux.number_of_nodes))
    G.add_edges_from(lux.topology)

    pos = {node: (x / 1000, y / 1000) for node, (x, y) in lux.positions.items()}

    node_colors = []
    for node in G.nodes():
        if node == 0:
            node_colors.append("red")  # servidor principal
        elif node in lux.edge_computing_servers:
            node_colors.append("orange")  # servidores de borda
        else:
            node_colors.append("lightblue")  # roteadores

    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=400,
        font_size=8,
        font_weight="bold",
    )
    plt.title("Topologia LuxembourgV2")
    plt.show()

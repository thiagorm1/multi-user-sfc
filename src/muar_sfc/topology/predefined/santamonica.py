import numpy as np

from muar_sfc.core.net import Net
from muar_sfc.topology.base.topology_base import TopologyBase  # Importando a classe abstrata

LIGHT_SPEED = 3 * 10e8
CLOUD_LATENCY = 0
BANDWIDTH_CAPACITY = 1000
CPU_CAPACITY = 100
CACHE_CAPACITY = 100


class SantaMonica(TopologyBase):
    def __init__(self):
        self.latency = 1
        self.bandwidth_capacity = BANDWIDTH_CAPACITY
        self.cpu_capacity = CPU_CAPACITY
        self.cache_capacity = CACHE_CAPACITY
        self.processing_nodes = np.array([4, 14, 16, 18, 19, 20, 21, 25, 26, 32, 33, 35, 36]) - 1
        self.topology = [
            (1, 2),
            (1, 3),
            (4, 2),
            (4, 5),
            (20, 4),
            (5, 20),
            (19, 20),
            (21, 20),
            (11, 21),
            (11, 12),
            (12, 13),
            (14, 13),
            (15, 14),
            (14, 29),
            (35, 15),
            (29, 35),
            (24, 35),
            (24, 23),
            (33, 23),
            (33, 30),
            (30, 33),
            (31, 30),
            (16, 31),
            (16, 32),
            (34, 32),
            (32, 26),
            (33, 32),
            (32, 26),
            (25, 26),
            (25, 16),
            (17, 16),
            (17, 18),
            (10, 18),
            (18, 9),
            (6, 9),
            (9, 19),
            (19, 27),
            (25, 27),
            (8, 3),
            (8, 7),
            (7, 10),
            (22, 28),
            (21, 22),
            (28, 35),
            (36, 32),
        ]
        # Ajusta os índices dos nós para começar em 0
        self.topology = [(edge[0] - 1, edge[1] - 1) for edge in self.topology]

    def generate_substrate_network(self):
        substrate_network = Net()

        # Adicionando arestas e inicializando capacidades e latências
        for edge in self.topology:
            if edge == self.topology[-1]:
                substrate_network.init_link_latency(edge[0], edge[1], CLOUD_LATENCY)
            else:
                substrate_network.init_link_latency(edge[0], edge[1], self.latency)
            substrate_network.init_bandwidth_capacity(edge[0], edge[1], self.bandwidth_capacity)

        # Configurando os nós (CPU, cache)
        for node in range(36):  # Número total de nós
            if node in self.processing_nodes:
                substrate_network.init_node_cpu_capacity(node, self.cpu_capacity)
                substrate_network.init_node_cache_capacity(node, self.cache_capacity)
            else:
                substrate_network.init_node_cpu_capacity(node, 0)
                substrate_network.init_node_cache_capacity(node, 0)

        # Precomputar caminhos mínimos e atualizar a rede
        substrate_network.pre_get_single_source_minimum_latency_path()
        substrate_network.update()

        return substrate_network

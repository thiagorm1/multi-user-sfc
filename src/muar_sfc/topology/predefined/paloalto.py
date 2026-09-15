import numpy as np

from muar_sfc.core.net import Net
from muar_sfc.topology.base.topology_base import TopologyBase  # Importando a classe abstrata

LIGHT_SPEED = 3 * 10e8
CLOUD_LATENCY = 1
BANDWIDTH_CAPACITY = 10000
CPU_CAPACITY = 100
CACHE_CAPACITY = 100


class PaloAlto(TopologyBase):
    def __init__(self):
        self.latency = 1
        self.bandwidth_capacity = BANDWIDTH_CAPACITY
        self.cpu_capacity = CPU_CAPACITY
        self.cache_capacity = CACHE_CAPACITY
        self.edge_computing_servers = np.array([2, 3, 5, 7, 8, 11, 14, 18, 24, 27, 32, 33])
        self.number_of_nodes = 37
        self.nodes = np.arange(1, 37)
        self.topology = [
            (0, 3),  # Cloud
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 2),
            (2, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 3),
            (18, 21),
            (21, 22),
            (22, 23),
            (23, 24),
            (24, 20),
            (18, 24),
            (24, 25),
            (25, 26),
            (26, 22),
            (7, 27),
            (27, 28),
            (28, 8),
            (7, 29),
            (27, 30),
            (12, 31),
            (31, 32),
            (32, 33),
            (33, 14),
            (11, 34),
            (34, 32),
            (33, 35),
            (35, 36),
            (36, 14),
        ]
        self.positions = {
            0: (8000, 8000),
            1: (7278, 1622),
            2: (6400, 2341),
            3: (5000, 3364),
            4: (4440, 3825),
            5: (3932, 4212),
            6: (3400, 4620),
            7: (2290, 5440),
            8: (2190, 5760),
            9: (3287, 6056),
            10: (3868, 5622),
            11: (4476, 5613),
            12: (4845, 4811),
            13: (5712, 4120),
            14: (6700, 3375),
            15: (5975, 1833),
            16: (5380, 766),
            17: (4217, 110),
            18: (4000, 813),
            19: (3841, 1690),
            20: (4245, 2346),
            21: (3530, 1230),
            22: (3100, 1335),
            23: (3720, 2230),
            24: (3880, 2600),
            25: (3350, 2930),
            26: (3735, 3500),
            27: (1790, 5890),
            28: (2540, 6200),
            29: (2100, 5650),
            30: (1420, 5175),
            31: (5470, 5190),
            32: (6670, 5261),
            33: (7570, 4215),
            34: (7850, 3100),
            35: (7680, 2524),
            36: (7284, 1635),
        }

    def generate_substrate_network(self):
        substrate_network = Net()

        # Adicionando arestas e inicializando capacidades e latências
        for edge in self.topology:
            if edge == self.topology[0]:
                substrate_network.init_link_latency(edge[0], edge[1], CLOUD_LATENCY)
            else:
                substrate_network.init_link_latency(edge[0], edge[1], self.latency)
            substrate_network.init_bandwidth_capacity(edge[0], edge[1], self.bandwidth_capacity)

        # Configurando os nós (posição, capacidade de CPU e cache)
        for node in range(0, len(self.positions)):
            substrate_network.set_node_position(node, self.positions[node])
            if node in self.edge_computing_servers:
                substrate_network.init_node_cpu_capacity(node, self.cpu_capacity)
                substrate_network.init_node_cache_capacity(node, self.cache_capacity)
            else:
                substrate_network.init_node_cpu_capacity(node, 0)
                substrate_network.init_node_cache_capacity(node, 0)

        # Precomputar caminhos mínimos e atualizar a rede
        substrate_network.pre_get_single_source_minimum_latency_path()
        substrate_network.update()

        return substrate_network

    def get_topology_info(self):
        """Retorna informações da topologia."""
        nodes = list(np.arange(1, self.number_of_nodes))
        edges = self.topology  # Lista de arestas
        edge_computing_servers = self.processing_nodes.tolist()  # Nós de processamento

        return {"nodes": nodes, "edges": edges, "ec_servers": edge_computing_servers}

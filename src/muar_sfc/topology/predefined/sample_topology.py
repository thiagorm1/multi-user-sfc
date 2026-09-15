import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from muar_sfc.core.net import Net

LIGHT_SPEED = 3 * 10e8
CLOUD_LATENCY = 1
# bandwidth_capacity = 1000 # 1000
bandwidth_capacity = 1000  # 1000
cpu_capacity = 100
cache_capacity = 100


class SampleTopology:
    def __init__(self) -> None:
        self.number_of_nodes = 15
        self.processing_nodes = None

    def generate_substrate_network(self):

        # latency = 500 / LIGHT_SPEED
        latency = 1

        topology = [
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

        topology = [
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
            # (2,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,3),
            # (18,21),(21,22),(22,23),(23,24),(24,20),
            # (18,24),(24,25),(25,26),(26,22),
            # (7,27),(27,28),(28,8),
            # (7,29),
            # (27,30),
            # (12,31),(31,32),(32,33),(33,14),
            # (11,34),(34,32),
            # (33,35),(35,36),(36,14)
        ]
        position = {
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
            # 15:(5975,1833),16:(5380, 766),17:(4217, 110),18:(4000, 813),
            # 19:(3841,1690),20:(4245,2346),
            # 21:(3530,1230),22:(3100,1335),23:(3720,2230),24:(3880,2600),
            # 25:(3350,2930),26:(3735,3500),
            # 27:(1790,5890),28:(2540,6200),
            # 29:(2100,5650),
            # 30:(1420,5175),
            # 31:(5470,5190),32:(6670,5261),33:(7570,4215),
            # 34:(7850,3100),35:(7680,2524), 36:(7284,1635)
        }

        substrate_network = Net()
        for _i, edge in enumerate(topology):
            if edge == topology[0]:
                substrate_network.init_link_latency(edge[0], edge[1], CLOUD_LATENCY)
            else:
                substrate_network.init_link_latency(edge[0], edge[1], latency)
                # substrate_network.init_link_latency(edge[0] - 1, edge[1] - 1, abs(latency[i]))
            substrate_network.init_bandwidth_capacity(edge[0], edge[1], bandwidth_capacity)

        processing_nodes = np.array([2, 3, 5, 7, 8, 11, 14])
        # processing_nodes = np.array([2,3,5,7,8,11,14,18,24,27,32,33])
        # processing_nodes = np.array(np.arange(len(topology)))
        # processing_nodes = np.array([18,19,20,25,32,33,35,36]) - 1
        for node in range(0, 15):
            # print(node)
            # substrate_network.init_node_cpu_capacity(node, cpu_capacity)
            # substrate_network.init_node_cache_capacity(node, cache_capacity)
            ##print(position[node])
            substrate_network.set_node_position(node, position[node])
            if node in processing_nodes:
                substrate_network.init_node_cpu_capacity(node, cpu_capacity)
                substrate_network.init_node_cache_capacity(node, cache_capacity)
            else:
                substrate_network.init_node_cpu_capacity(node, 0)
                substrate_network.init_node_cache_capacity(node, 0)

        substrate_network.pre_get_single_source_minimum_latency_path()
        substrate_network.update()
        return substrate_network


if __name__ == "__main__":
    substrate_network = SampleTopology().generate_substrate_network()
    # for node in range(1,15):
    # print(substrate_network.get_node_cpu_free(node))
    # print(substrate_network.get_node_position(node))
    a = substrate_network.all_shortest_paths()
    print(substrate_network.get_all_node_positions())
    nx.draw(substrate_network, with_labels=True)  # networkx draw()
    # plt.draw()  # pyplot draw()
    plt.show()

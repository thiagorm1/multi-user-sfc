import networkx as nx

from muar_sfc.core.net import Net

MAX_BANDWIDTH = 50
LIGHT_SPEED = 3 * 10e8
MAX_CPU_CAPACITY = 100
MAX_NUMBER_NODES = 25


def generate_substrate_network():

    topology = [
        (1, 15),
        (1, 17),
        (1, 19),
        (1, 23),
        (2, 5),
        (2, 16),
        (3, 11),
        (3, 24),
        (3, 25),
        (4, 7),
        (4, 8),
        (4, 10),
        (4, 15),
        (4, 21),
        (5, 2),
        (5, 7),
        (5, 9),
        (5, 11),
        (5, 19),
        (5, 20),
        (6, 17),
        (6, 18),
        (6, 22),
        (7, 4),
        (7, 5),
        (7, 13),
        (7, 24),
        (8, 24),
        (8, 25),
        (9, 5),
        (9, 10),
        (9, 14),
        (9, 22),
        (10, 4),
        (10, 9),
        (10, 11),
        (10, 16),
        (10, 17),
        (11, 3),
        (11, 5),
        (11, 10),
        (11, 14),
        (11, 24),
        (12, 23),
        (12, 24),
        (13, 7),
        (13, 16),
        (13, 17),
        (14, 9),
        (14, 11),
        (14, 21),
        (14, 22),
        (15, 1),
        (15, 4),
        (15, 21),
        (16, 2),
        (16, 10),
        (16, 13),
        (17, 1),
        (17, 6),
        (17, 10),
        (17, 13),
        (17, 18),
        (17, 22),
        (17, 23),
        (18, 6),
        (18, 17),
        (19, 1),
        (19, 5),
        (19, 8),
        (19, 20),
        (19, 22),
        (19, 24),
        (20, 5),
        (20, 19),
        (20, 25),
        (21, 4),
        (21, 14),
        (21, 15),
        (22, 6),
        (22, 9),
        (22, 14),
        (22, 17),
        (22, 19),
        (22, 25),
        (23, 1),
        (23, 12),
        (23, 17),
        (24, 3),
        (24, 7),
        (24, 11),
        (24, 12),
        (24, 19),
        (25, 3),
        (25, 8),
        (25, 20),
        (25, 22),
    ]

    substrate_network = Net()
    substrate_network.add_nodes_from(topology)

    for edge in topology:
        substrate_network.set_link_bandwidth_capacity(edge[0], edge[1], MAX_BANDWIDTH)
    for node in range(MAX_NUMBER_NODES):
        substrate_network.set_node_cpu_capacity(node, MAX_CPU_CAPACITY)

    substrate_network.update()

    return substrate_network


if __name__ == "__main__":
    network = generate_substrate_network()
    nx.draw(network)

"""
Consideration:
Algorithm should not do any modification on substrate network

It should only use network information and sfc information to
solve and give out a mapping and route info, that

route info :=
{
    src:  [1, 2, 3],
    vnf1: [3, 4, 5],
    vnf2: [5, 6, 7],
    vnf3: [7, 8 ,9],
    dst:  []
}
"""

import logging
import random

import networkx as nx

from muar_sfc.config import ROOT_DIR

# Configuração de Observabilidade Estruturada
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Modernização Orientada a Objetos Multiplataforma (pathlib)
log_path = ROOT_DIR / "logs" / "RandomAlgorithm.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

ch = logging.FileHandler(log_path)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class RandomAlgorithm:
    """Random algorithm.

    Randomly select k number of substrate nodes from substrate network, where k is equal
    to the VNFs of an SFC. Note that these k nodes cannot be the substrate nodes who
    host src and dst of SFC.
    Start from src, find shortest path one by one and connect the shortest paths among
    selected nodes.

    By comparing with this random algorithm, the purpose of random allocation is to know
    whether our approach has a considerable impact on the load balance, or alternatively
    simply building servers in the preferred node by the network operator is enough to
    load balance the network.

    Refer to:
    Random fit placement
    F. Carpio, S. Dhahri, and A. Jukan, "VNF placement with replication for Load balancing
    in NFV networks," IEEE Int. Conf. Commun., pp. 1-6, 2017.
    """

    def __init__(self):
        self.name = "Random Algorithm"
        self.substrate_network = None
        self.sfc = None
        self.node_info = None
        self.route_info = None
        self.latency = None

    def clear_all(self):
        logger.info("clear all")
        self.substrate_network = None
        self.sfc = None
        self.node_info = None
        self.route_info = None
        self.latency = None

    def install_substrate_network(self, substrate_network):
        self.substrate_network = substrate_network
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        return self.sfc

    def start_algorithm(self):
        substrate_network = self.substrate_network
        sfc = self.sfc
        logger.info("Algorithm start")
        if self.algorithm(substrate_network, sfc):
            logger.info("Algorithm end, success")
            return True
        logger.info("Algorithm end, failed")
        return False

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def algorithm(self, substrate_network, sfc):
        nodes = substrate_network.nodes()
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        # (ingress and egress substrate network nodes)
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        number_of_vnfs = sfc.get_number_of_vnfs()

        # Randomly generated K number of substrate nodes from all substrate nodes except
        # ingress and egress. K is equal to the number of vnfs in sfc.
        nodes_list = list(nodes)
        try:
            nodes_list.remove(src_substrate_node)  # Remove ingress nodes from substrate node list
            nodes_list.remove(dst_substrate_node)  # Remove egress nodes from substrate node list
        except ValueError:
            # src or dst is not in the substrate network nodes
            logger.warning("src or dst is not in the substrate network nodes")
            return False

        if number_of_vnfs > len(nodes_list):
            # have not sufficient nodes for host vnfs
            return False

        # random choose k number of nodes, and append the egress substrate network nodes
        # for host dst vnf
        random_sampled_substrate_network_nodes = random.sample(nodes_list, k=number_of_vnfs)
        random_sampled_substrate_network_nodes.append(dst_substrate_node)

        route_info = {}
        bandwidth_usage_info = {}
        latency = 0

        current_vnf = src_vnf
        pre_substrate_node = src_substrate_node
        for node in random_sampled_substrate_network_nodes:
            # 🔴 Blindagem Arquitetural: Captura específica de erros do NetworkX
            try:
                # get shortest path length. here shortest path is weighted by latency.
                path_latency = substrate_network.get_shortest_path_length(
                    pre_substrate_node, node
                )
                path = substrate_network.get_shortest_path(pre_substrate_node, node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                logger.warning(
                    "have no path between two nodes: %s - %s", pre_substrate_node, node
                )
                return False

            pre_substrate_node = node

            latency = latency + path_latency
            vnf_id = current_vnf.id
            route_info[vnf_id] = path
            current_vnf = current_vnf.get_next_vnf()

            cpu_request = sfc.get_vnf_cpu_request(current_vnf)
            cpu_available = substrate_network.get_node_cpu_free(node)

            if cpu_request > cpu_available:
                logger.warning("cpu resources is not sufficient")
                return False

            bandwidth_request = sfc.get_link_bandwidth_request(vnf_id, current_vnf.id)

            length = len(path)
            for i in range(0, length - 1):
                edge_key = frozenset((path[i], path[i + 1]))
                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = (
                        substrate_network.get_link_bandwidth_free(path[i], path[i + 1])
                        - bandwidth_request
                    )
                if residual_bandwidth < 0:
                    logger.warning("Bandwidth resources is not sufficient")
                    return False
                bandwidth_usage_info[edge_key] = residual_bandwidth

        route_info[dst_vnf.id] = []  # make a placeholder for dst.
        self.route_info = route_info
        self.latency = latency
        return True

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
from pathlib import Path

import networkx as nx

from muar_sfc.config import ROOT_DIR

# Configuração de Observabilidade Estruturada
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Modernização Orientada a Objetos Multiplataforma (pathlib)
log_path = Path(ROOT_DIR) / "logs" / "GreedyAlgorithm.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

ch = logging.FileHandler(log_path)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class GreedyAlgorithm:
    """Greedy Algorithm.
    This algorithm starts from the substrate network node which hosts src of an SFC,
    checks its neighbor nodes, finds the neighbor node with a shortest latency edge,
    and use the node to host the vnf. The algorithm greedily finds all nodes for
    hosting vnf.
    Finally, the algorithm finds a shortest path from the substrate node who hosts
    the last vnf in the SFC to the substrate node who hosts dst of the SFC.

    Deploy VNF one by one, with a shortest path from the node to the previous
    substrate node.
    """

    def __init__(self):
        self.name = "Greedy Algorithm"
        self.substrate_network = None
        self.sfc = None
        self.node_info = None
        self.route_info = None
        self.latency = None

    def clear_all(self):
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
        logger.info("Start algorithm")
        if self.algorithm(substrate_network, sfc):
            logger.info("End algorithm, success")
            return True
        logger.info("End algorithm, failed")
        return False

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def algorithm(self, substrate_network, sfc):
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        route_info = {}

        latency = 0
        used_node = [src_substrate_node, dst_substrate_node]

        number_of_vnfs = sfc.get_number_of_vnfs()
        current_vnf = src_vnf
        current_substrate_node = src_substrate_node

        net_info = substrate_network
        old_server_resources = net_info._node
        servers = list(old_server_resources.keys())
        random.shuffle(servers)

        dist_src_dst = substrate_network.get_shortest_path_length(
            src_substrate_node, dst_substrate_node
        )
        if dist_src_dst > 4:
            servers = substrate_network.get_shortest_path(
                src_substrate_node, dst_substrate_node
            )

        # Inicializa o dicionário de recursos dos servidores
        # Atribuído à variável de descarte '_' para evitar erros de linting
        # em operações sem efeito colateral
        _ = {
            server: {
                "cpu_capacity": substrate_network.get_node_cpu_capacity(server),
                "cache_capacity": substrate_network.get_node_cache_capacity(server),
                "cpu_used": substrate_network.get_node_cpu_used(server),
                "cache_used": substrate_network.get_node_cache_used(server),
                "cpu_free": substrate_network.get_node_cpu_free(server),
                "cache_free": substrate_network.get_node_cache_free(server),
                "position": old_server_resources[server]["position"],
                "reuse": [],
            }
            for server in servers
        }

        node = 0
        for _ in range(0, number_of_vnfs):
            substrate_network.edges(current_substrate_node)
            next_vnf = current_vnf.get_next_vnf()

            cpu_request = sfc.get_vnf_cpu_request(next_vnf)
            cache_request = sfc.get_vnf_cache_request(next_vnf)

            sfc.get_link_bandwidth_request(current_vnf.id, next_vnf.id)

            min_latency = None

            for e in servers:
                # THIS SHOULD STAY DEACTIVATED TO ALLOW MULTIPLE VNFS
                # HOSTED IN THE SAME NODE
                if e in used_node:
                    # do not use the node used before, to avoid loop
                    continue

                cpu_available = substrate_network.get_node_cpu_free(e)
                cache_available = substrate_network.get_node_cache_free(e)

                if cpu_request > cpu_available:
                    # if node has not sufficient cpu, check next edge.
                    logger.debug("node %s has not %s cpu", e, cpu_request)
                    continue
                if cache_request > cache_available:
                    # if node has not sufficient cache, check next edge.
                    logger.debug("node %s has not %s cache", e, cache_request)
                    continue

                edge_latency = substrate_network.get_shortest_path_length(
                    current_substrate_node, e
                )
                if min_latency is None or edge_latency < min_latency:
                    min_latency = edge_latency
                    node = e

            if node is not None and min_latency is not None:
                # be careful that node can be 0
                path = substrate_network.get_shortest_path(current_substrate_node, node)
                route_info[current_vnf.id] = path
                used_node.append(node)
                latency = latency + min_latency

            else:
                logger.debug("node is not existing")
                self.route_info = {}
                self.latency = None
                return False

            current_substrate_node = node
            current_vnf = next_vnf

        try:
            # get shortest path length. here shortest path length is weighted by latency.
            path = substrate_network.get_shortest_path(node, dst_substrate_node)
            path_latency = substrate_network.get_shortest_path_length(
                node, dst_substrate_node
            )
        # 🔴 Substituição arquitetural: Captura focada do NetworkX
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning(
                "have no path between last vnf and dst: %s - %s", node, dst_substrate_node
            )
            self.route_info = {}
            self.latency = None
            return False

        route_info[current_vnf.id] = path
        route_info["dst"] = []
        latency = latency + path_latency

        self.route_info = route_info
        self.latency = latency

        path = self.route_info["src"]
        for i in range(len(path) - 1):
            edge_latency = self.substrate_network.get_link_latency(path[i], path[i + 1])
            self.latency = self.latency - edge_latency

        if len(self.route_info.keys()) != 6:
            self.latency = None
            self.route_info = {}
            return False

        if self.latency > sfc.get_latency_request():
            self.route_info = {}
            return False

        return True

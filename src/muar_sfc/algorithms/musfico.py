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

import copy
import logging

import networkx as nx

# Imports adicionados para lidar com a rede como um Grafo e
# calcular a latência de comunicação realista
from muar_sfc.algorithms.networkUtils import (
    calculate_latency_betwen_nodes,
    get_link_bandwidth_free,
)
from muar_sfc.config import ROOT_DIR

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.FileHandler(ROOT_DIR / "logs" / "musfico.log")
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class Musfico:
    def __init__(self):
        self.name = "musfico"
        self.graph = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.latency_minus_dst = None

    def clear_all(self):
        self.graph = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None

    def install_substrate_network(self, graph):
        self.graph = graph

        self.single_source_minimum_latency_path = {}
        for node in self.graph.nodes():
            self.single_source_minimum_latency_path[node] = nx.single_source_dijkstra(
                self.graph, source=node, cutoff=None, weight="latency"
            )
        return self.graph

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.route_info = {}

        src_vnf = self.sfc.get_src_vnf()
        src_substrate_node = self.sfc.get_substrate_node(src_vnf)
        dst_vnf = self.sfc.get_dst_vnf()
        dst_substrate_node = self.sfc.get_substrate_node(dst_vnf)

        for node in self.graph.nodes():
            self.node_info[node] = {}
            for vnf_id, _vnf in list(sfc.vnfs.items()):
                self.node_info[node][vnf_id] = {}
                self.node_info[node][vnf_id]["flag"] = False
                self.node_info[node][vnf_id]["latency"] = float("inf")
                self.node_info[node][vnf_id]["path"] = []
                self.node_info[node][vnf_id]["src_path"] = []
                self.node_info[node][vnf_id]["previous_substrate_node"] = None
                self.node_info[node][vnf_id]["current_substrate_nodes"] = []
                self.node_info[node][vnf_id]["bandwidth_usage_info"] = {}

            self.node_info[node][src_vnf.id] = {}
            self.node_info[node][src_vnf.id]["flag"] = False
            self.node_info[node][dst_vnf.id] = {}

        self.node_info[src_substrate_node][src_vnf.id]["flag"] = True
        self.node_info[src_substrate_node][src_vnf.id]["latency"] = 0
        self.node_info[src_substrate_node][src_vnf.id]["src_path"] = []
        self.node_info[src_substrate_node][src_vnf.id]["path"] = []
        self.node_info[src_substrate_node][src_vnf.id]["current_substrate_nodes"] = [
            src_substrate_node
        ]

        self.node_info[dst_substrate_node][dst_vnf.id]["flag"] = False
        self.node_info[dst_substrate_node][dst_vnf.id]["latency"] = float("inf")
        self.node_info[dst_substrate_node][dst_vnf.id]["src_path"] = []
        self.node_info[dst_substrate_node][dst_vnf.id]["path"] = []
        self.node_info[dst_substrate_node][dst_vnf.id]["current_substrate_nodes"] = []

        self.node_info[src_substrate_node][src_vnf.id]["bandwidth_usage_info"] = {}
        self.node_info[dst_substrate_node][dst_vnf.id]["bandwidth_usage_info"] = {}

        return self.sfc

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def start_algorithm(self):
        if self.algorithm():
            if self.latency is not None:
                if self.latency < 0:
                    print("Latencia negativa")
                    self.latency = None
                    self.route_info = False
                    return False

                if self.latency > self.sfc.get_latency_request():
                    self.latency = None
                    self.route_info = False
                    return False
            return True
        return False

    def algorithm(self):
        sfc = self.sfc

        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        self.src_substrate_node = src_substrate_node
        self.dst_substrate_node = dst_substrate_node

        (node_latency, node_path) = self.single_source_minimum_latency_path[dst_substrate_node]

        vnf1 = src_vnf.get_next_vnf()
        self._dp(src_substrate_node, vnf1)

        vnf = vnf1.get_next_vnf()
        while vnf.id != dst_vnf.id:
            for node in self.graph.nodes():
                self._dp(node, vnf)
            vnf = vnf.get_next_vnf()

        previous_vnf = sfc.get_previous_vnf(dst_vnf)
        previous_vnf_id = previous_vnf.id

        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, dst_vnf.id)

        for node in self.graph.nodes():
            if node in (dst_substrate_node, src_substrate_node):
                continue

            path = node_path.get(node)
            if not path:
                continue

            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[node][previous_vnf_id]["bandwidth_usage_info"]
            )
            length = len(path)

            real_comm_latency = 0.0

            for i in range(0, length - 1):
                u, v = path[i], path[i + 1]
                edge_key = frozenset((u, v))
                residual_bandwidth = None

                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = (
                        get_link_bandwidth_free(self.graph, u, v) - bandwidth_request
                    )

                if residual_bandwidth < 0:
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth

                # BOA PRÁTICA: Ignora a latência das arestas caso a VNF seja src ou dst,
                # mantendo paridade com o Instantiator
                if previous_vnf_id not in ["src", "dst"]:
                    real_comm_latency += calculate_latency_betwen_nodes(
                        self.graph, u, v, previous_vnf
                    )

            if not is_bandwidth_sufficient:
                continue

            _latency = self.node_info[node][previous_vnf_id]["latency"]
            total_latency = _latency + real_comm_latency

            if total_latency > 35:
                continue

            if (
                not self.node_info[dst_substrate_node][dst_vnf.id]["latency"]
                or total_latency < self.node_info[dst_substrate_node][dst_vnf.id]["latency"]
            ):
                self.node_info[dst_substrate_node][dst_vnf.id]["latency"] = total_latency
                self.node_info[dst_substrate_node][dst_vnf.id]["path"] = path
                self.node_info[dst_substrate_node][dst_vnf.id]["path"].reverse()
                self.node_info[dst_substrate_node][dst_vnf.id]["current_substrate_nodes"] = (
                    self.node_info[node][previous_vnf_id]["current_substrate_nodes"][:]
                )
                self.node_info[dst_substrate_node][dst_vnf.id]["current_substrate_nodes"].append(
                    dst_substrate_node
                )
                self.node_info[dst_substrate_node][dst_vnf.id]["src_path"] = (
                    self.node_info[node][previous_vnf_id]["src_path"][:]
                    + self.node_info[dst_substrate_node]["dst"]["path"][:]
                )
                self.node_info[dst_substrate_node][dst_vnf.id]["flag"] = True

        if self.node_info[dst_substrate_node][dst_vnf.id]["flag"]:
            previous_vnf = dst_vnf
            previous_substrate_node = dst_substrate_node

            while True:
                path = self.node_info[previous_substrate_node][previous_vnf.id]["path"]
                if not path:
                    break
                previous_substrate_node = path[0]
                previous_vnf = sfc.get_previous_vnf(previous_vnf)
                if previous_vnf:
                    self.route_info[previous_vnf.id] = path
                else:
                    break

            self.route_info[dst_vnf.id] = []
            self.latency = self.node_info[dst_substrate_node][dst_vnf.id]["latency"]

            # COMO AGORA A LATÊNCIA FOI ACUMULADA CORRETAMENTE (SEM O SRC),
            # AQUELE LAÇO CONFUSO QUE SUBTRAÍA A LATÊNCIA NO FINAL FOI DELETADO.

            if self.latency > sfc.get_latency_request() or self.latency < 0:
                self.route_info = {}
                self.latency = None
                return False
            return True
        else:
            return False

    def handle_failure(self):
        self.route_info = False
        self.latency = None

    def _dp(self, substrate_node, vnf):
        sfc = self.sfc
        previous_vnf = sfc.get_previous_vnf(vnf)
        previous_vnf_id = previous_vnf.id

        vnf_id = vnf.id
        (node_latency, node_path) = self.single_source_minimum_latency_path[substrate_node]

        _latency = self.node_info[substrate_node][previous_vnf_id]["latency"]

        cpu_request = sfc.get_vnf_cpu_request(vnf)
        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, vnf_id)
        cache_request = sfc.get_vnf_cache_request(vnf)

        for node in self.graph.nodes():
            # Guard Clauses
            if node == substrate_node:
                continue
            if node in self.node_info[substrate_node][previous_vnf_id]["current_substrate_nodes"]:
                continue
            if node == self.src_substrate_node or node == self.dst_substrate_node:
                continue

            cpu_available = (
                self.graph.nodes[node]["cpu_capacity"] - self.graph.nodes[node]["cpu_used"]
            )
            cache_available = (
                self.graph.nodes[node]["cache_capacity"] - self.graph.nodes[node]["cache_used"]
            )

            if cpu_available == 0 or cache_available == 0:
                continue

            if cpu_request > cpu_available or cache_request > cache_available:
                continue

            path = node_path.get(node)
            if not path:
                continue

            if path[0] != substrate_node:
                path.reverse()
            length = len(path)

            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[substrate_node][previous_vnf_id]["bandwidth_usage_info"]
            )

            real_comm_latency = 0.0

            for i in range(0, length - 1):
                u, v = path[i], path[i + 1]
                edge_key = frozenset((u, v))
                residual_bandwidth = None

                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = (
                        get_link_bandwidth_free(self.graph, u, v) - bandwidth_request
                    )

                if residual_bandwidth < 0:
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth

                # BOA PRÁTICA: Ignora a latência de comunicação se a VNF for src ou dst
                if previous_vnf_id not in ["src", "dst"]:
                    real_comm_latency += calculate_latency_betwen_nodes(
                        self.graph, u, v, previous_vnf
                    )

            if not is_bandwidth_sufficient:
                continue

            total_latency = _latency + real_comm_latency

            self.node_info[node][vnf_id]["bandwidth_usage_info"] = bandwidth_usage_info

            if (
                not self.node_info[node][vnf_id]["latency"]
                or total_latency <= self.node_info[node][vnf_id]["latency"]
            ):
                self.node_info[node][vnf_id]["latency"] = total_latency
                self.node_info[node][vnf_id]["path"] = path
                self.node_info[node][vnf_id]["flag"] = True
                self.node_info[node][vnf_id]["previous_substrate_node"] = substrate_node
                self.node_info[node][vnf_id]["current_substrate_nodes"] = self.node_info[
                    substrate_node
                ][previous_vnf_id]["current_substrate_nodes"][:]
                self.node_info[node][vnf_id]["current_substrate_nodes"].append(node)
                self.node_info[node][vnf_id]["src_path"] = (
                    self.node_info[substrate_node][previous_vnf_id]["src_path"][:] + path[:-1]
                )

        return True

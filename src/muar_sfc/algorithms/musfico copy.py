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

from muar_sfc.algorithms.greedy_algorithm import GreedyAlgorithm
from muar_sfc.config import ROOT_DIR
from muar_sfc.utils.k_shortest_paths import k_shortest_paths

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler(ROOT_DIR + "./logs/musfico.log")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class Musfico:
    def __init__(self):
        self.name = "musfico"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.latency_minus_dst = None

    def calculate_route_last_vnf(self):
        """
        Calculate shortest path between last vnf and previous vnf
        for a new destination.
        """

        # get destination vnf from sfc
        dst_vnf = self.sfc.get_dst_vnf()
        # get previous vnf from old object
        previous_vnf = self.sfc.get_previous_vnf(dst_vnf)
        prev_vnf_node = self.route_info[previous_vnf.id][0]
        # applies the algorithm
        shortest_path = k_shortest_paths(
            self.substrate_network, prev_vnf_node, self.dst_substrate_node, k=1, weight="latency"
        )
        self.route_info[previous_vnf.id] = shortest_path[0]

    def clear_all(self):
        # logger.debug('clear all')
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None

    def install_substrate_network(self, substrate_network):
        self.substrate_network = substrate_network
        self.single_source_minimum_latency_path = (
            self.substrate_network.single_source_minimum_latency_path
        )
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc

        src_vnf = self.sfc.get_src_vnf()
        src_substrate_node = self.sfc.get_substrate_node(src_vnf)
        dst_vnf = self.sfc.get_dst_vnf()
        dst_substrate_node = self.sfc.get_substrate_node(dst_vnf)

        for node in self.substrate_network.nodes():
            self.node_info[node] = {}
            for vnf_id, _vnf in list(sfc.vnfs.items()):
                # Not include src and dst.
                self.node_info[node][vnf_id] = {}
                self.node_info[node][vnf_id]["flag"] = (
                    False  # whether vnf/id can be placed on node
                )
                self.node_info[node][vnf_id]["latency"] = float("inf")
                self.node_info[node][vnf_id]["path"] = []
                self.node_info[node][vnf_id]["src_path"] = []
                self.node_info[node][vnf_id]["previous_substrate_node"] = None
                self.node_info[node][vnf_id][
                    "current_substrate_nodes"
                ] = []  # The meta information
                # in which is a set of substrate node
                # has been assigned to VNFs in order
                self.node_info[node][vnf_id]["bandwidth_usage_info"] = {}

            self.node_info[node][src_vnf.id] = {}
            self.node_info[node][src_vnf.id]["flag"] = (
                False  # src cannot be placed on the node except src node
            )
            self.node_info[node][dst_vnf.id] = {}

        self.node_info[src_substrate_node][src_vnf.id]["flag"] = (
            True  # src can be placed on the src node
        )
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
        substrate_network = self.substrate_network
        sfc = self.sfc
        # logger.info('Algorithm start')
        return self.algorithm(substrate_network, sfc)

    def algorithm(self, substrate_network, sfc):
        nodes = substrate_network.nodes()
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        self.src_substrate_node = src_substrate_node
        self.dst_substrate_node = dst_substrate_node

        vnf1 = src_vnf.get_next_vnf()
        self._dp(src_substrate_node, vnf1)

        vnf = vnf1.get_next_vnf()
        while vnf.id != dst_vnf.id:
            for node in nodes:
                self._dp(node, vnf)
            vnf = vnf.get_next_vnf()

        # For dst:
        (node_latency, node_path) = self.single_source_minimum_latency_path[
            dst_substrate_node
        ]
        # Get single source path from substrate node to all other substrate node
        # here node in latency and path results is the node host previous vnf
        previous_vnf = sfc.get_previous_vnf(dst_vnf)
        previous_vnf_id = previous_vnf.id

        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, dst_vnf.id)

        for node, latency in list(node_latency.items()):
            if node in (dst_substrate_node, src_substrate_node):
                # if node is ingress or egress, continue
                continue

            # Check bandwidth resources
            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[node][previous_vnf_id]["bandwidth_usage_info"]
            )
            path = node_path[node]
            length = len(path)
            for i in range(0, length - 1):
                edge_key = frozenset((path[i], path[i + 1]))
                residual_bandwidth = None
                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = (
                        bandwidth_usage_info[edge_key] - bandwidth_request
                    )
                else:
                    residual_bandwidth = (
                        self.substrate_network.get_link_bandwidth_free(
                            path[i], path[i + 1]
                        )
                        - bandwidth_request
                    )
                if residual_bandwidth < 0:
                    # logger.warning('Bandwidth resources is not sufficient to dst')
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth
            if not is_bandwidth_sufficient:
                # check next path
                continue
            _latency = self.node_info[node][previous_vnf_id]["latency"]
            if not self.node_info[dst_substrate_node][dst_vnf.id][
                "latency"
            ] or _latency + latency < self.node_info[dst_substrate_node][dst_vnf.id].get(
                "latency", float("inf")
            ):
                self.node_info[dst_substrate_node][dst_vnf.id]["latency"] = _latency + latency
                self.node_info[dst_substrate_node][dst_vnf.id]["path"] = node_path[node]
                self.node_info[dst_substrate_node][dst_vnf.id]["path"].reverse()
                self.node_info[dst_substrate_node][dst_vnf.id][
                    "current_substrate_nodes"
                ] = self.node_info[node][previous_vnf_id][
                    "current_substrate_nodes"
                ][
                    :
                ]
                self.node_info[dst_substrate_node][dst_vnf.id]["current_substrate_nodes"].append(
                    dst_substrate_node
                )
                self.node_info[dst_substrate_node][dst_vnf.id][
                    "src_path"
                ] = (
                    self.node_info[node][previous_vnf_id]["src_path"][:]
                    + self.node_info[dst_substrate_node]["dst"]["path"][:]
                )
                self.node_info[dst_substrate_node][dst_vnf.id]["flag"] = True

        if self.node_info[dst_substrate_node][dst_vnf.id]["flag"]:
            return self._process_solution(sfc, dst_vnf, dst_substrate_node)
        return False

    def _process_solution(self, sfc, dst_vnf, dst_substrate_node):
        # There is a solution
        # Backtracking
        # Start from dst to backtracking to src
        previous_vnf = dst_vnf
        previous_substrate_node = dst_substrate_node
        print("backtrack pvs node:", previous_substrate_node)
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
        self.route_info[previous_vnf.id][0]

        # Se chegou aqui e `self.route_info` não está vazio
        # (ou a flag de que achou caminho é True):
        oversubscribed = self.check_resource_excess(sfc)

        if oversubscribed != []:
            print(f"Nós que extrapolaram recursos: {oversubscribed}")
            return bool(self.solucao_paliativa(sfc))

        # remove latency from dst to previous vnf
        # self.latency_minus_dst = self.latency - len(self.route_info[previous_vnf.id])
        if "src" not in self.route_info:
            return True
        path = self.route_info["src"]
        for i in range(len(path) - 1):
            edge_latency = self.substrate_network.get_link_latency(
                path[i], path[i + 1]
            )
            self.latency = self.latency - edge_latency
        if self.latency > sfc.get_latency_request():
            self.route_info = {}
            return False
        return True

    def _dp(self, substrate_node, vnf):
        """
        Start from substrate node substrate_node, calculate all paths and
        latency from substrate_node to other nodes N.
        update information in nodes N for vnf, if latency is minimum.
        """
        # Get precedent of the vnf
        sfc = self.sfc
        previous_vnf = sfc.get_previous_vnf(vnf)
        previous_vnf_id = previous_vnf.id
        # if not self.node_info[substrate_node][previous_vnf_id]['flag']:
        #    # This substrate node cannot host precedent vnf, thus, no need to exam further.
        #    return False

        vnf_id = vnf.id
        # Get single source path from substrate node to all other substrate node
        (node_latency, node_path) = self.single_source_minimum_latency_path[substrate_node]

        _latency = self.node_info[substrate_node][previous_vnf_id]["latency"]

        cpu_request = sfc.get_vnf_cpu_request(vnf)
        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, vnf_id)
        cache_request = sfc.get_vnf_cache_request(vnf)

        for node, latency in list(node_latency.items()):
            # if latency > 3:
            #     continue

            # if node == substrate_node:
            #    # Cannot use the current substrate node to host this vnf.
            #    continue

            # if node in self.node_info[substrate_node][previous_vnf_id][
            #     'current_substrate_nodes'
            # ]:
            #    # If node has been used, cannot host this vnf
            #    # Current_substrate_nodes contains the nodes that have been used
            #    continue
            if node == self.src_substrate_node or node == self.dst_substrate_node:
                # Ingress and egress cannot host this vnf
                continue

            # Check CPU and cache resources
            cpu_available = self.substrate_network.get_node_cpu_free(node)
            cache_available = self.substrate_network.get_node_cache_free(node)
            if cpu_request > cpu_available or cache_request > cache_available:
                # if node has not sufficient cpu, check next node.
                continue

            # Check bandwidth resources
            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[substrate_node][previous_vnf_id]["bandwidth_usage_info"]
            )

            path = node_path[node]
            if path[0] != substrate_node:
                path.reverse()
            length = len(path)
            for i in range(0, length - 1):
                edge_key = frozenset((path[i], path[i + 1]))
                residual_bandwidth = None
                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = (
                        bandwidth_usage_info[edge_key] - bandwidth_request
                    )
                else:
                    residual_bandwidth = (
                        self.substrate_network.get_link_bandwidth_free(
                            path[i], path[i + 1]
                        )
                        - bandwidth_request
                    )
                if residual_bandwidth < 0:
                    # logger.warning('Bandwidth resources is not sufficient')
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth
            if not is_bandwidth_sufficient:
                continue
            self.node_info[node][vnf_id]["bandwidth_usage_info"] = bandwidth_usage_info
            if (
                not self.node_info[node][vnf_id]["latency"]
                or (_latency + latency) <= self.node_info[node][vnf_id]["latency"]
            ):
                self.node_info[node][vnf_id]["latency"] = _latency + latency
                self.node_info[node][vnf_id]["path"] = node_path[node]
                self.node_info[node][vnf_id]["flag"] = True
                self.node_info[node][vnf_id]["previous_substrate_node"] = substrate_node
                self.node_info[node][vnf_id][
                    "current_substrate_nodes"
                ] = self.node_info[substrate_node][previous_vnf_id][
                    "current_substrate_nodes"
                ][
                    :
                ]
                self.node_info[node][vnf_id]["current_substrate_nodes"].append(node)
                self.node_info[node][vnf_id]["src_path"] = (
                    self.node_info[substrate_node][previous_vnf_id]["src_path"][:]
                    + node_path[node][:-1]
                )
        return True

    def check_resource_excess(self, sfc):
        """
        Verifica se a alocação em self.route_info extrapola a CPU ou Cache
        livre nos nós do Substrate. Retorna lista de nós que foram excedidos
        ou lista vazia se estiver tudo correto.
        """

        if not self.route_info:
            # Se não houve solução (route_info vazio), não há o que checar
            return []

        # Dicionários para acumular consumo de CPU/Cache em cada nó
        node_cpu_usage = {}
        node_cache_usage = {}

        # Inicializa cada nó com consumo zero
        for n in self.substrate_network.nodes():
            node_cpu_usage[n] = 0
            node_cache_usage[n] = 0

        hosts = []
        # Para cada VNF (chave do dict), some o consumo no nó que hospeda
        for vnf_id, path in self.route_info.items():
            # Ignorar caso especial src/dst se esses IDs estão no route_info
            if vnf_id in ["src", "dst"]:
                continue

            # No seu exemplo, "path[0]" é o nó que hospeda a VNF
            host_node = path[0]
            hosts.append(host_node)
            # Obtem o objeto da VNF
            vnf_obj = sfc.vnfs[vnf_id]

            # Soma as requisições
            cpu_req = sfc.get_vnf_cpu_request(vnf_obj)
            cache_req = sfc.get_vnf_cache_request(vnf_obj)

            node_cpu_usage[host_node] += cpu_req
            node_cache_usage[host_node] += cache_req

        # Agora, verificar se algum nó excedeu a capacidade
        oversubscribed_nodes = []
        hosts = list(set(hosts))
        for node in hosts:
            cpu_free = self.substrate_network.get_node_cpu_free(node)
            cache_free = self.substrate_network.get_node_cache_free(node)

            cpu_capacity = self.substrate_network.get_node_cpu_capacity(node)
            cache_capacity = self.substrate_network.get_node_cache_capacity(node)

            solution_cpu_req = node_cpu_usage[node]
            solution_cache_req = node_cache_usage[node]

            if cpu_capacity <= 0 or cache_capacity <= 0:
                oversubscribed_nodes.append(node)

            if (cpu_free < solution_cpu_req) or (cache_free < solution_cache_req):
                oversubscribed_nodes.append(node)
        return oversubscribed_nodes

    def solucao_paliativa(self, sfc):
        # Se houver algum nó que estourou CPU ou Cache, descarta a solução
        # Como solução parcial, caso o MSF dê uma solução inválida, iremos
        # usar a abordagem greedy para alocação
        greedy_alg = GreedyAlgorithm()
        # self.clear_all()
        greedy_alg.clear_all()
        greedy_alg.install_substrate_network(self.substrate_network)
        greedy_alg.install_SFC(self.sfc)
        greedy_alg.mono = False
        self.old_greedy = True
        greedy_alg.start_algorithm()
        self.route_info = greedy_alg.get_route_info()  # Routes choosen by the alg
        self.latency = greedy_alg.get_latency()  # latency of the solution     # indica falha

        if self.latency is None:
            self.route_info = {}
            return False

        if self.latency > sfc.get_latency_request():
            self.route_info = {}
            return False

        oversubscribed = self.check_resource_excess(sfc)

        if oversubscribed != []:
            print("Greedy retornou uma solução errada")
        return True

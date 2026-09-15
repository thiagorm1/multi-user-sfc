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

from muar_sfc.algorithms.algorithm import Algorithm
from muar_sfc.config import ROOT_DIR
from muar_sfc.utils.k_shortest_paths import k_shortest_paths

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler(ROOT_DIR / "logs" / "BrunoAlg.log")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


"""
Funcionamento:

Para cada SF pertencente à uma SFC

Verifique se para a SF atual, existe uma SF existente compatível em algum lugar da topologia.
Ordene os nós de acordo com a distância para a SF anterior.

Caso exista, aloque-a nesse ponto e faça o shortest path entre ela e a SF anterior.

Caso contrário, aloque-a o mais próximo possível da SF anterior em um nó com recurso
disponível e por um caminho com banda disponível.

Se SF for a penúltima, aplique k shortest path para chegar na dst

Repita o algoritmo ate o DST

"""


class BrunoAlg(Algorithm):
    def __init__(self):
        self.name = "Unknown 2"
        self.substrate_network = None
        self.sfc = None
        self.path_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.saved_cpu = 0
        self.saved_cache = 0
        self.saved_band = 0
        self.latency_minus_dst = 0
        self.shareable_sfs = None
        self.fail_for_band = 0
        self.fail_for_cache = 0
        self.fail_for_cpu = 0
        self.weights = None

    def clear_all(self):
        # logger.debug('clear all')
        self.substrate_network = None
        self.sfc = None
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.path_info = {}

    def install_substrate_network(self, substrate_network):
        self.substrate_network = substrate_network
        self.single_source_minimum_latency_path = (
            self.substrate_network.single_source_minimum_latency_path
        )
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        src_vnf = self.sfc.get_src_vnf()
        dst_vnf = self.sfc.get_dst_vnf()
        for vnf_id, _vnf in list(sfc.vnfs.items()):
            self.path_info[vnf_id] = {}
            self.path_info[vnf_id]["path"] = []
            self.path_info[vnf_id]["latency"] = float("inf")
            self.path_info[vnf_id]["substrate_node"] = None
            self.path_info[vnf_id]["flag"] = False
        self.path_info[src_vnf.id] = {}
        self.path_info[dst_vnf.id] = {}
        self.path_info[src_vnf.id]["path"] = []
        self.path_info[src_vnf.id]["latency"] = 0
        self.path_info[src_vnf.id]["substrate_node"] = None
        self.path_info[src_vnf.id]["flag"] = True
        self.path_info[dst_vnf.id]["path"] = []
        self.path_info[dst_vnf.id]["latency"] = float("inf")
        self.path_info[dst_vnf.id]["substrate_node"] = None
        self.path_info[dst_vnf.id]["flag"] = True
        return self.sfc

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def start_algorithm(self, shareable_sfs=None, **kwargs):
        substrate_network = self.substrate_network
        sfc = self.sfc
        # logger.info('Algorithm start')
        if kwargs:
            self.weights = kwargs["weights"]
        if shareable_sfs is not None:
            self.shareable_sfs = shareable_sfs

        # O método self.algorithm já retorna True ou False
        return self.algorithm(substrate_network, sfc)

    def algorithm(self, substrate_network, sfc):
        # informações sf = (nó, sf_id, sf)
        substrate_network.nodes()
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        self.src_substrate_node = src_substrate_node
        self.dst_substrate_node = dst_substrate_node

        vnf1 = src_vnf.get_next_vnf()
        self._alg(vnf1)

        vnf = vnf1.get_next_vnf()
        while vnf.id != dst_vnf.id:
            success = self._alg(vnf)
            if not success:
                self.route_info = {}
                return False
            vnf = vnf.get_next_vnf()
        self.route_info = {}

        previous_vnf = dst_vnf.get_previous_vnf()
        previous_vnf_node = self.path_info[previous_vnf.id]["substrate_node"]

        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf.id, dst_vnf.id)

        """
                     dst
                      '
                      7
                      '        5 - dst
                      '      /
        src - 1 - 2 - 3 - 4
                             \
                               6 - dst

        """

        shortest_paths = k_shortest_paths(
            self.substrate_network,
            source=previous_vnf_node,
            target=dst_substrate_node,
            weight="latency",
            k=5,
        )
        for path in shortest_paths:
            is_enough_bandwidth = True
            path_latency = 0
            for i in range(len(path) - 1):
                required_bandwidth = (
                    self.substrate_network.get_link_bandwidth_free(path[i], path[i + 1])
                    - bandwidth_request
                )
                path_latency += self.substrate_network.get_link_latency(path[i], path[i + 1])
                if required_bandwidth < 0:
                    is_enough_bandwidth = False
                    break
            if not is_enough_bandwidth:
                continue
            self.path_info[previous_vnf.id]["path"] = path
            self.path_info[previous_vnf.id]["latency"] = path_latency
            break
        self.latency = 0
        previous_vnf = dst_vnf.get_previous_vnf()
        while True:
            path = self.path_info[previous_vnf.id]["path"]
            if not path:
                break
            if previous_vnf:
                self.route_info[previous_vnf.id] = path
                self.latency += self.path_info[previous_vnf.id]["latency"]
            else:
                break
            if previous_vnf.id == "src":
                break
            previous_vnf = previous_vnf.get_previous_vnf()
        self.route_info[dst_vnf.id] = []
        return True

    def _alg(self, vnf):
        sfc = self.sfc
        # get previous sf
        previous_vnf = vnf.get_previous_vnf()
        # get resources from vnf
        cpu_request = sfc.get_vnf_cpu_request(vnf)
        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf.id, vnf.id)
        cache_request = sfc.get_vnf_cache_request(vnf)
        # get node from previous sf
        previous_vnf_node = (
            self.src_substrate_node
            if previous_vnf.id == "src"
            else self.path_info[previous_vnf.id]["substrate_node"]
        )
        # print("previous vnf node is ", previous_vnf_node)
        # get all shortest paths from previous node to all other nodes
        (node_latency, node_path) = self.single_source_minimum_latency_path[previous_vnf_node]
        # store smallest latency
        total_latency = 100000
        # check if shareable sf found
        is_shareable_sf = False
        # O(n x m x k ) where n = number of nodes in the topology,
        # m = number of shareable sfs in node, k = number of SFs for a SFC
        for node, sfs in self.shareable_sfs.items():
            if vnf.id in list(map(lambda sf: sf.id, sfs)):
                candidate_path = node_path[node]
                if candidate_path[0] != previous_vnf_node:
                    candidate_path.reverse()
                candidate_latency = node_latency[node]
                # new smallest latency
                total_latency = candidate_latency
                # check bandwidth for the path
                is_enough_bandwidth = True
                for i in range(len(candidate_path) - 1):
                    required_bandwidth = (
                        self.substrate_network.get_link_bandwidth_free(
                            candidate_path[i], candidate_path[i + 1]
                        )
                        - bandwidth_request
                    )
                    if required_bandwidth < 0:
                        is_enough_bandwidth = False
                        break
                if not is_enough_bandwidth:
                    continue
                # if possible path has a greater latency then it's not a
                # suitable candidate
                if candidate_latency > total_latency:
                    continue
                total_latency = candidate_latency
                self.path_info[previous_vnf.id]["path"] = candidate_path
                self.path_info[previous_vnf.id]["latency"] = total_latency
                self.path_info[vnf.id]["substrate_node"] = candidate_path[-1]
                self.path_info[vnf.id]["flag"] = True
                is_shareable_sf = True
        if is_shareable_sf:
            return True
        # if there are no shareable SFs available, then we try to allocate the SF
        # as close as possible to the previouss SF.
        for _idx, (node, latency) in enumerate(node_latency.items()):
            # print("currently testing node ", idx)
            if node == self.src_substrate_node or node == self.dst_substrate_node:
                continue
            cpu_available = self.substrate_network.get_node_cpu_free(node)
            cache_available = self.substrate_network.get_node_cache_free(node)
            if cpu_request > cpu_available:
                # print("not enough cpu")
                continue
            elif cache_request > cache_available:
                # print("not enough cache")
                continue
            # check bandwidth resources
            is_enough_bandwidth = True
            path = node_path[node]
            if path[0] != previous_vnf_node:
                path.reverse()
            for i in range(len(path) - 1):
                required_bandwidth = (
                    self.substrate_network.get_link_bandwidth_free(path[i], path[i + 1])
                    - bandwidth_request
                )
                if required_bandwidth < 0:
                    is_enough_bandwidth = False
                    break
            if not is_enough_bandwidth:
                # print("not enough bandwidth")
                continue
            # finally, minimizes the latency for the given choice
            if latency > total_latency:
                # print("latency is way too high..")
                continue
            total_latency = latency
            # print("hey, we finally have a path ", path)
            self.path_info[previous_vnf.id]["path"] = path
            self.path_info[previous_vnf.id]["latency"] = latency
            self.path_info[vnf.id]["substrate_node"] = path[-1]
            self.path_info[vnf.id]["flag"] = True
        return len(self.path_info[previous_vnf.id]["path"]) != 0

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
ch = logging.FileHandler(ROOT_DIR / "logs" / "BrunoAlgNew.log")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)



class BrunoAlgNew(Algorithm):
    def __init__(self):
        self.name = "Unknown 3"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
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
        self.node_info = {}

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
        for node in self.substrate_network.nodes():
            self.node_info[node] = {}
            for vnf_id, _vnf in list(sfc.vnfs.items()):
                self.node_info[node][vnf_id] = {}
                self.node_info[node][vnf_id]["path"] = []
            self.node_info[node][src_vnf.id] = {}
            self.node_info[node][dst_vnf.id] = {}
            self.node_info[node][src_vnf.id]["path"] = []
            self.node_info[node][dst_vnf.id]["path"] = []
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
        return self.algorithm(substrate_network, sfc)

    def algorithm(self, substrate_network, sfc):
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        self.src_substrate_node = src_substrate_node
        self.dst_substrate_node = dst_substrate_node
        paths = k_shortest_paths(
            self.substrate_network,
            source=self.src_substrate_node,
            target=self.dst_substrate_node,
            weight="latency",
            k=3,
        )

        for path in paths:
            if self._alg(path, self.sfc):
                break

    def _alg(self, path, sfc):
        """
        Initially, we try to allocate as many sfs as possible in a single node. When
        nodes are starting to get full, we spread sfs around the topology, which
        increases bandwidth cost.
        """
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()
        deployed_sfs = 0
        reverse_path = path.copy()
        reverse_path.reverse()
        # start allocating from dst to src
        for idx, node in enumerate(reverse_path):
            if node == self.src_substrate_node or node == self.dst_substrate_node:
                continue
            cpu_available = self.substrate_network.get_node_cpu_free(node)
            cache_available = self.substrate_network.get_node_cache_free(node)
            total_cpu_request = 0
            total_cache_request = 0
            total_bandwidth_request = 0
            # iterate over the sfs
            vnf = dst_vnf.get_previous_vnf()
            while vnf.id != src_vnf.id:
                # print("running for vnf ", vnf.id)
                # if vnf placed already, continue.
                if len(self.node_info[node][vnf.id]["path"]) != 0:
                    continue
                # get resources used by sf
                total_cpu_request += sfc.get_vnf_cpu_request(vnf)
                total_cache_request += sfc.get_vnf_cache_request(vnf)
                # see if there are enough resources
                if total_cpu_request > cpu_available:
                    break
                if total_cache_request > cache_available:
                    break
                self.node_info[node][vnf.id]["path"].append(node)
                # count number of sfs placed in the node
                deployed_sfs += 1
                vnf = vnf.get_previous_vnf()
            # if all sfs placed we're basically done.
            if deployed_sfs == sfc.get_number_of_vnfs():
                return True
            # if not all sfs could be placed in node, check bandwidth resources
            else:
                bandwidth_request = sfc.get_link_bandwidth_request(vnf.id, vnf.get_next_vnf().id)
                residual_bandwidth = (
                    self.substrate_network.get_link_bandwidth_free(reverse_path[idx + 1], node)
                    - bandwidth_request
                )
                if residual_bandwidth < 0:
                    continue
                total_bandwidth_request += bandwidth_request
        return False

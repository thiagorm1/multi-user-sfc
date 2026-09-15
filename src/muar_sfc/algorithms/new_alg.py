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

from muar_sfc.algorithms.algorithm import Algorithm
from muar_sfc.config import ROOT_DIR

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler(ROOT_DIR / "logs" / "NewAlg.log")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)



class NewAlg(Algorithm):
    def __init__(self):
        self.name = "Unknown"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.saved_band = 0
        self.saved_cpu = 0
        self.saved_cache = 0
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
                self.node_info[node][vnf_id]["cost"] = float("inf")
                self.node_info[node][vnf_id]["saved_band"] = 0
                self.node_info[node][vnf_id]["saved_cpu"] = 0
                self.node_info[node][vnf_id]["saved_cache"] = 0
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
        self.node_info[src_substrate_node][src_vnf.id]["cost"] = 0
        self.node_info[src_substrate_node][src_vnf.id]["saved_band"] = 0
        self.node_info[src_substrate_node][src_vnf.id]["saved_cpu"] = 0
        self.node_info[src_substrate_node][src_vnf.id]["saved_cache"] = 0
        self.node_info[src_substrate_node][src_vnf.id]["src_path"] = []
        self.node_info[src_substrate_node][src_vnf.id]["path"] = []
        self.node_info[src_substrate_node][src_vnf.id]["current_substrate_nodes"] = [
            src_substrate_node
        ]
        self.node_info[dst_substrate_node][dst_vnf.id]["flag"] = False
        self.node_info[dst_substrate_node][dst_vnf.id]["latency"] = float("inf")
        self.node_info[dst_substrate_node][dst_vnf.id]["cost"] = float("inf")
        self.node_info[dst_substrate_node][dst_vnf.id]["saved_band"] = 0
        self.node_info[dst_substrate_node][dst_vnf.id]["saved_cpu"] = 0
        self.node_info[dst_substrate_node][dst_vnf.id]["saved_cache"] = 0
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

    def start_algorithm(self, shareable_sfs=None, **kwargs):
        substrate_network = self.substrate_network
        sfc = self.sfc
        if kwargs:
            self.weights = kwargs["weights"]
        if shareable_sfs is not None:
            self.shareable_sfs = shareable_sfs

        return self.algorithm(substrate_network, sfc)

    def _compute_cost(self, length=0, bw_saved=0, cpu_saved=0, cache_saved=0, latency=0):
        # normalizar custos
        # if length > 1:
        #    cost = 1 / (cpu_saved + 1) + 1 / (cache_saved + 1) + length / (bw_saved + 1)
        # else:
        #    cost = 1 / (cpu_saved + 1) + 1 / (cache_saved + 1)
        # bw 70 cpu 50 cache 50
        #
        # self.weights
        bw_saved / (500 * length)
        cpu_saved / 100
        normalized_cache = cache_saved / 100

        cost = 1000 if normalized_cache == 0 else 0

        return cost

    def algorithm(self, substrate_network, sfc):
        # informações sf = (nó, sf_id, sf)
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
        ]  # Get single source path from substrate node to all other substrate node
        # here node in latency and path results is the node host previous vnf
        previous_vnf = sfc.get_previous_vnf(dst_vnf)
        previous_vnf_id = previous_vnf.id

        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, dst_vnf.id)
        saved_band = 0
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
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = (
                        self.substrate_network.get_link_bandwidth_free(path[i], path[i + 1])
                        - bandwidth_request
                    )
                if residual_bandwidth < 0:
                    # logger.warning('Bandwidth resources is not sufficient to dst')
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth
            if not is_bandwidth_sufficient:
                continue

            _latency = self.node_info[node][previous_vnf_id]["latency"]
            _saved_cpu = self.node_info[node][previous_vnf_id]["saved_cpu"]
            _saved_cache = self.node_info[node][previous_vnf_id]["saved_cache"]
            _saved_band = self.node_info[node][previous_vnf_id]["saved_band"]
            _cost = self.node_info[node][previous_vnf_id]["cost"]

            total_latency = _latency + latency
            total_band_saved = _saved_band + saved_band
            total_cpu_saved = _saved_cpu
            total_cache_saved = _saved_cache
            cost = self._compute_cost(
                length=length,
                bw_saved=total_band_saved,
                cpu_saved=total_cpu_saved,
                cache_saved=total_cache_saved,
                latency=total_latency,
            )
            if (
                not self.node_info[dst_substrate_node][dst_vnf.id]["cost"]
                or (_cost + cost) <= self.node_info[dst_substrate_node][dst_vnf.id]["cost"]
            ):
                self.node_info[dst_substrate_node][dst_vnf.id]["latency"] = total_latency
                self.node_info[dst_substrate_node][dst_vnf.id]["cost"] = _cost + cost
                self.node_info[dst_substrate_node][dst_vnf.id]["saved_band"] = total_band_saved
                self.node_info[dst_substrate_node][dst_vnf.id]["saved_cpu"] = total_cpu_saved
                self.node_info[dst_substrate_node][dst_vnf.id]["saved_cache"] = total_cache_saved
                self.node_info[dst_substrate_node][dst_vnf.id]["path"] = node_path[node]
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
            # There is a solution
            # Backtracking
            # Start from dst to backtracking to src
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
            self.saved_band = self.node_info[dst_substrate_node][dst_vnf.id]["saved_band"]
            self.saved_cache = self.node_info[dst_substrate_node][dst_vnf.id]["saved_cache"]
            self.saved_cpu = self.node_info[dst_substrate_node][dst_vnf.id]["saved_cpu"]
            # refuse if latency is too high
            if "src" not in self.route_info:
                return True
            path = self.route_info["src"]
            for i in range(len(path) - 1):
                edge_latency = self.substrate_network.get_link_latency(path[i], path[i + 1])
                self.latency = self.latency - edge_latency
            # if self.latency > sfc.get_latency_request():
            #    self.route_info = {}
            #    return False
            return True
        else:
            return False

    def _dp(self, substrate_node, vnf):
        """
        Start from substrate node substrate_node, calculate all paths and latency
        from substrate_node to other nodes N.
        update information in nodes N for vnf, if latency is minimum.
        """

        # Get precedent of the vnf
        sfc = self.sfc
        previous_vnf = sfc.get_previous_vnf(vnf)
        previous_vnf_id = previous_vnf.id

        vnf_id = vnf.id
        # Get single source path from substrate node to all other substrate node
        (node_latency, node_path) = self.single_source_minimum_latency_path[substrate_node]

        _latency = self.node_info[substrate_node][previous_vnf_id]["latency"]
        _cost = self.node_info[substrate_node][previous_vnf_id]["cost"]
        _saved_cpu = self.node_info[substrate_node][previous_vnf_id]["saved_cpu"]
        _saved_cache = self.node_info[substrate_node][previous_vnf_id]["saved_cache"]
        _saved_band = self.node_info[substrate_node][previous_vnf_id]["saved_band"]

        saved_cpu = 0
        saved_cache = 0

        cpu_request = sfc.get_vnf_cpu_request(vnf)
        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, vnf_id)
        cache_request = sfc.get_vnf_cache_request(vnf)

        is_enough_cpu, is_enough_cache, is_enough_band = True, True, True

        # check shareable sfs.
        # SUBSTITUIR CAMINHO DE MENOR LATENCIA POR CAMINHO DE MENOR CUSTO
        # CUSTO = 1 / ECONOMIA DE BANDA DO CAMINHO + 1 / ECONOMIA DE
        for node, latency in list(node_latency.items()):
            saved_band = 0
            if node == self.src_substrate_node or node == self.dst_substrate_node:
                continue
            # Check CPU and cache resources
            cpu_available = self.substrate_network.get_node_cpu_free(node)
            cache_available = self.substrate_network.get_node_cache_free(node)
            if cpu_request > cpu_available:
                # if node has not sufficient cpu, check next node.
                # logger.warning("not sufficient CPU in Node " + str(node))
                is_enough_cpu = False
                self.fail_for_cpu += 1
            if cache_request > cache_available:
                # if node has not sufficient cache, check next node.
                # logger.warning("not sufficient cache in Node " + str(node))
                is_enough_cache = False
                self.fail_for_cache += 1

            # map resources saved (SIM102 resolvido aqui)
            if self.shareable_sfs is not None and vnf_id in list(map(lambda sf: sf.id,
                                                                     self.shareable_sfs[node])):
                saved_cpu = vnf.get_cpu_request()
                saved_cache = vnf.get_cache_request()

            # check bandwidth resources
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
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    # if link data can be reused, then we request no bandwidth
                    flux_info = self.substrate_network.sfs_flux_info
                    flux_ids = [flux_vnf.id for flux_vnf in flux_info[(path[i], path[i + 1])]]
                    if vnf_id in flux_ids:
                        saved_band += bandwidth_request
                        bandwidth_request = 0
                    residual_bandwidth = (
                        self.substrate_network.get_link_bandwidth_free(path[i], path[i + 1])
                        - bandwidth_request
                    )
                if residual_bandwidth < 0:
                    # logger.warning('dp Bandwidth resources is not sufficient')
                    is_bandwidth_sufficient = False
                    break
                # route_info = {}
                # condition to check latency
                bandwidth_usage_info[edge_key] = residual_bandwidth
            if not is_bandwidth_sufficient:
                self.fail_for_band += 1
                is_enough_band = False
            if not is_enough_band or not is_enough_cache or not is_enough_cpu:
                is_enough_band = True
                is_enough_cache = True
                is_enough_cpu = True
                continue
            self.node_info[node][vnf_id]["bandwidth_usage_info"] = bandwidth_usage_info
            total_latency = _latency + latency
            total_cpu_saved = _saved_cpu + saved_cpu
            total_cache_saved = _saved_cache + saved_cache
            total_band_saved = _saved_band + saved_band
            cost = self._compute_cost(
                length=length,
                bw_saved=total_band_saved,
                cpu_saved=total_cpu_saved,
                cache_saved=total_cache_saved,
                latency=total_latency,
            )
            if (
                not self.node_info[node][vnf_id]["cost"]
                or (_cost + cost) <= self.node_info[node][vnf_id]["cost"]
            ):
                self.node_info[node][vnf_id]["latency"] = total_latency
                self.node_info[node][vnf_id]["cost"] = _cost + cost
                self.node_info[node][vnf_id]["saved_band"] = total_band_saved
                self.node_info[node][vnf_id]["saved_cpu"] = total_cpu_saved
                self.node_info[node][vnf_id]["saved_cache"] = total_cache_saved
                self.node_info[node][vnf_id]["path"] = node_path[node]
                self.node_info[node][vnf_id]["flag"] = True
                self.node_info[node][vnf_id]["previous_substrate_node"] = substrate_node
                self.node_info[node][vnf_id]["current_substrate_nodes"] = self.node_info[
                    substrate_node
                ][previous_vnf_id]["current_substrate_nodes"][:]
                self.node_info[node][vnf_id]["current_substrate_nodes"].append(node)
                self.node_info[node][vnf_id]["src_path"] = (
                    self.node_info[substrate_node][previous_vnf_id]["src_path"][:]
                    + node_path[node][:-1]
                )
        return True

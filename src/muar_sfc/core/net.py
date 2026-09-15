import logging
import math
import re

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

"""
node data structure:

network
sfc_dict = {sfc_id: sfc_object}

route_info
{
    sfc_id :
        {
            src:  [1, 2, 3],
            vnf1: [3, 4, 5],
            vnf2: [5, 6, 7],
            vnf3: [7, 8 ,9],
            dst:  []
        }
}

node
{
    cpu_capacity: xx,
    cpu_used: xx,
    cpu_free: xx,
    cache_capacity: xx,
    cache_used: xx,
    cache_free: xx,
    #vnf_list: [vnfObject]
    sfc_vnf_list: [(sfc_id, vnf)]
}

"""


class Net(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)
        self.sfc_dict = {}
        self.sfc_route_info = {}  # sfc_id, route_info
        self.nodes_reliability = {}

        self.total_cpu_used = 0
        self.total_cpu_capacity = 0

        self.total_cache_used = 0
        self.total_cache_capacity = 0

        self.total_bandwidth_used = 0
        self.total_bandwidth_capacity = 0

        self.max_cpu_overload = 1
        self.max_cache_overload = 1

        self.cpu_saved = 0
        self.cache_saved = 0
        self.shared_vnfs_count = 0
        self.Graph = 0
        self.single_source_minimum_latency_path = None
        self.nodes_positions = 0
        self.kd_positions = 0

        # stores processing delay information for each node in the topology
        self.processing_delay_info = []
        self.shareable_sf_sfc = {}  # stores sfc information for a shareable sf for
        # later undeploy.
        self.sf_route_info = {}
        self.sfs_flux_info = {}
        self.shared_sfs = {}
        self.shareable_band = False
        self.shareable_node = True
        self.verbose = False
        # self.lock = threading.Lock()

    def set_verbose(self, verbose):
        self.verbose = verbose

    def reset_sfs_flux_info(self):
        for edge in self.edges():
            self.sfs_flux_info[(edge[0], edge[1])] = []
            self.sfs_flux_info[(edge[1], edge[0])] = []

    def reset_shared_sfs(self):
        for node in self.nodes():
            self.shared_sfs[node] = []

    def set_node_processing_delay(self):
        pass

    def get_node_processing_delay(self):
        pass

    def get_nodes_processing_delay(self):
        pass

    def set_max_cpu_overload(self, ratio):
        self.max_cpu_overload = ratio

    def get_max_cpu_overload(self):
        return self.max_cpu_overload

    def get_sfc_by_id(self, sfc_id):
        return self.sfc_dict[sfc_id]

    def set_sfc(self, sfc):
        self.sfc_dict[sfc.id] = sfc

    def _get_node_attribute(self, node_id, attr):
        attribute = nx.get_node_attributes(self, attr)
        return attribute[node_id]

    def reset_node_cpu_capacity(self, node_id, cpu_capacity):
        self.set_node_cpu_capacity(node_id, cpu_capacity)
        self.set_node_cpu_used(node_id, 0)
        try:
            self.set_node_reuse(node_id, [])
        except (KeyError, AttributeError):
            logger.warning(f"Nó {node_id} não possui o atributo de reúso.")
        self.set_node_cpu_free(node_id, cpu_capacity)
        self._set_node_attribute(node_id, sfc_vnf_list=[])
        return

    def reset_node_cell_bandwidth_capacityy(self, node_id, cell_bw_capacity):
        self.set_node_cell_bandwidth_capacity(node_id, cell_bw_capacity)
        self.set_node_cell_bandwidth_used(node_id, 0)

        try:
            self.set_node_reuse(node_id, [])
        except (KeyError, AttributeError):  # <-- CORRIGIDO: Captura Estrita
            logger.warning(f"Nó {node_id} não possui o atributo de reúso de banda.")

        self.set_node_cell_bandwidth_free(node_id, cell_bw_capacity)
        self._set_node_attribute(node_id, sfc_vnf_list=[])
        return

    def init_node_cpu_capacity(self, node_id, cpu_capacity):
        self.reset_node_cpu_capacity(node_id, cpu_capacity)
        return

    def init_node_cell_bandwidth_capacity(self, node_id, cell_bandwidth_capacity):
        self.reset_node_cell_bandwidth_capacityy(node_id, cell_bandwidth_capacity)
        return

    def init_node_reliability(self, node_id, reliability):
        self.set_node_reliability(node_id, reliability)
        return

    def set_node_position(self, node_id, position):
        self._set_node_attribute(node_id, position=position)

    def get_node_position(self, node_id):
        return self._get_node_attribute(node_id, "position")

    def get_all_node_positions(self):
        if self.nodes_positions == 0 or self.kd_positions == 0:
            node_positions = {}
            for node_id in self.nodes:
                if node_id != 0:
                    node_positions[node_id] = self.get_node_position(node_id)
            self.nodes_positions = node_positions
            server_coords = np.array(list(node_positions.values()))
            self.kd_positions = KDTree(server_coords)
            return node_positions
        else:
            return self.nodes_positions

    def set_node_cpu_capacity(self, node_id, cpu_capacity):
        self._set_node_attribute(node_id, cpu_capacity=cpu_capacity)
        return cpu_capacity

    def set_node_cpu_used(self, node_id, cpu_used):
        self._set_node_attribute(node_id, cpu_used=cpu_used)

    def set_node_cell_bandwidth_capacity(self, node_id, cell_bw_capacity):
        self._set_node_attribute(node_id, cell_bw_capacity=cell_bw_capacity)
        return cell_bw_capacity

    def set_node_cell_bandwidth_used(self, node_id, cell_bw_used):
        self._set_node_attribute(node_id, cell_bw_used=cell_bw_used)

    def set_node_cell_bandwidth_free(self, node_id, cell_bw_free):
        return self._set_node_attribute(node_id, cell_bw_free=cell_bw_free)

    def set_node_reuse(self, node_id, reuse=None):
        if reuse is None:
            reuse = []
        self._set_node_attribute(node_id, reuse=reuse)

    def set_node_reliability(self, node_id, reliability):
        self._set_node_attribute(node_id, reliability=reliability)
        return

    def get_node_reliability(self, node_id):
        return self._get_node_attribute(node_id, "reliability")

    def set_node_cpu_free(self, node_id, cpu_free):
        return self._set_node_attribute(node_id, cpu_free=cpu_free)

    def get_node_cpu_capacity(self, node_id):
        return self._get_node_attribute(node_id, "cpu_capacity")

    def get_node_cpu_used(self, node_id):
        return self._get_node_attribute(node_id, "cpu_used")

    def get_node_cpu_free(self, node_id):
        return self._get_node_attribute(node_id, "cpu_free")

    def allocate_cpu_resource(self, node_id, cpu_amount):

        self.get_node_cpu_capacity(node_id)
        cpu_free = self.get_node_cpu_free(node_id)
        cpu_used = self.get_node_cpu_used(node_id)

        if cpu_amount > cpu_free:
            return False
        else:
            self.set_node_cpu_free(node_id, cpu_free - cpu_amount)
            self.set_node_cpu_used(node_id, cpu_used + cpu_amount)
            return True

    def deallocate_cpu_resource(self, node_id, cpu_amount):
        self.get_node_cpu_capacity(node_id)
        cpu_free = self.get_node_cpu_free(node_id)
        cpu_used = self.get_node_cpu_used(node_id)
        if cpu_amount > cpu_free:
            return False
        else:
            self.set_node_cpu_free(node_id, cpu_free + cpu_amount)
            self.set_node_cpu_used(node_id, cpu_used - cpu_amount)
            return True

    def change_node_cache_capacity(self, node_id):
        pass

    def reset_node_cache_capacity(self, node_id, cache_capacity):
        self.set_node_cache_capacity(node_id, cache_capacity)
        self.set_node_cache_used(node_id, 0)
        self.set_node_cache_free(node_id, cache_capacity)
        self._set_node_attribute(node_id, sfc_vnf_list=[])
        return

    def init_node_cache_capacity(self, node_id, cache_capacity):
        self.reset_node_cache_capacity(node_id, cache_capacity)
        return

    def set_node_cache_capacity(self, node_id, cache_capacity):
        self._set_node_attribute(node_id, cache_capacity=cache_capacity)
        return cache_capacity

    def get_shareable_sfs(self):
        return self.shared_sfs

    def get_flux_info(self):
        return self.sfs_flux_info

    def set_node_cache_used(self, node_id, cache_used):
        return self._set_node_attribute(node_id, cache_used=cache_used)

    def set_node_cache_free(self, node_id, cache_free):
        return self._set_node_attribute(node_id, cache_free=cache_free)

    def get_node_cache_capacity(self, node_id):
        return self._get_node_attribute(node_id, "cache_capacity")

    def get_node_cache_used(self, node_id):
        return self._get_node_attribute(node_id, "cache_used")

    def get_node_cache_free(self, node_id):
        return self._get_node_attribute(node_id, "cache_free")

    def allocate_cache_resource(self, node_id, cache_amount):
        cache_free = self.get_node_cache_free(node_id)
        cache_used = self.get_node_cache_used(node_id)
        if cache_amount > cache_free:
            return False
        else:
            self.set_node_cache_free(node_id, cache_free - cache_amount)
            self.set_node_cache_used(node_id, cache_used + cache_amount)
            return True

    def deallocate_cache_resource(self, node_id, cache_amount):
        cache_free = self.get_node_cache_free(node_id)
        cache_used = self.get_node_cache_used(node_id)
        if cache_amount > cache_free:
            return False
        else:
            self.set_node_cache_free(node_id, cache_free + cache_amount)
            self.set_node_cache_used(node_id, cache_used - cache_amount)
            return True

    def _set_link_attribute(self, u, v, **attr):
        self.add_edge(u, v, **attr)

    def _get_link_attribute(self, u, v, attr):
        attribute = nx.get_edge_attributes(self, attr)
        if (u, v) in attribute:
            return attribute[(u, v)]
        elif (v, u) in attribute:
            return attribute[(v, u)]
        return attribute[(u, v)]

    def get_link_bandwidth_capacity(self, u, v):
        return self._get_link_attribute(u, v, "bandwidth_capacity")

    def get_link_bandwidth_used(self, u, v):
        return self._get_link_attribute(u, v, "bandwidth_used")

    def get_link_bandwidth_free(self, u, v):
        return self._get_link_attribute(u, v, "bandwidth_free")

    def reset_bandwidth_capacity(self, u, v, bw_c):
        self.set_link_bandwidth_capacity(u, v, bw_c)
        self.set_link_bandwidth_used(u, v, 0)
        self.set_link_bandwidth_free(u, v, bw_c)
        return bw_c

    def init_bandwidth_capacity(self, u, v, bw_c):
        return self.reset_bandwidth_capacity(u, v, bw_c)

    def reset_bandwidth(self, u, v):
        # reset used and free bandwidth by keep capacity not changed
        capacity = self.get_link_bandwidth_capacity(u, v)
        # capacity = self.get_00_bandwidth_capacity(u, v)
        self.set_link_bandwidth_free(u, v, capacity)
        self.set_link_bandwidth_used(u, v, 0)

    def set_link_bandwidth_capacity(self, u, v, bw_c):
        self._set_link_attribute(u, v, bandwidth_capacity=bw_c)
        return bw_c

    def set_link_bandwidth_used(self, u, v, bw_u):
        self._set_link_attribute(u, v, bandwidth_used=bw_u)
        return bw_u

    def set_link_bandwidth_free(self, u, v, bw_f):
        self._set_link_attribute(u, v, bandwidth_free=bw_f)
        return bw_f

    def allocate_bandwidth_resource(self, u, v, bw_amount):
        self.get_link_bandwidth_capacity(u, v)
        bw_u = self.get_link_bandwidth_used(u, v)
        bw_f = self.get_link_bandwidth_free(u, v)
        # if bw_amount > bw_f:
        #     return False
        # else:
        self.set_link_bandwidth_used(u, v, bw_u + bw_amount)
        self.set_link_bandwidth_free(u, v, bw_f - bw_amount)
        return True

    def allocate_bandwidth_resource_path(self, path, bw_amount):
        length = len(path)
        for i in range(0, length - 1):
            self.allocate_bandwidth_resource(path[i], path[i + 1], bw_amount)

    def deallocate_bandwidth_resource(self, u, v, bw_amount):
        self.get_link_bandwidth_capacity(u, v)
        bw_u = self.get_link_bandwidth_used(u, v)
        bw_f = self.get_link_bandwidth_free(u, v)
        if bw_amount > bw_f:
            return False
        else:
            self.set_link_bandwidth_used(u, v, bw_u - bw_amount)
            self.set_link_bandwidth_free(u, v, bw_f + bw_amount)
            return True

    def deallocate_bandwidth_resource_path(self, path, bw_amount):
        length = len(path)
        for i in range(0, length - 1):
            self.deallocate_bandwidth_resource(path[i], path[i + 1], bw_amount)

    def get_link_latency(self, u, v):
        return self._get_link_attribute(u, v, "latency")

    def set_link_latency(self, u, v, bw_l):
        self._set_link_attribute(u, v, latency=bw_l)

    def init_link_latency(self, u, v, bw_l):
        self.set_link_latency(u, v, bw_l)

    def get_neighbours(self, node_id):
        return self.neighbors(node_id)

    def all_shortest_paths(self):
        r = nx.all_pairs_dijkstra(self, weight="latency")
        print([n for n in r])

    def get_shortest_paths(self, src, dst, weight):
        try:
            return nx.shortest_path(self, src, dst, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_minimum_latency_path(self, src, dst):
        return self.get_shortest_paths(src, dst, "latency")

    def get_minimum_free_bandwidth(self, path):
        length = len(path)
        minimum_free_bandwidth = float("inf")
        for i in range(0, length - 1):
            free_bandwidth = self.get_link_bandwidth_free(path[i], path[i + 1])
            if free_bandwidth < minimum_free_bandwidth:
                minimum_free_bandwidth = free_bandwidth
        return minimum_free_bandwidth

    def get_shortest_path_length(self, source, target):
        return nx.shortest_path_length(self, source=source, target=target, weight="latency")

    def get_shortest_path(self, source, target):
        return nx.shortest_path(self, source=source, target=target, weight="latency")

    def get_single_source_minimum_latency_path(self, src):
        return nx.single_source_dijkstra(self, source=src, cutoff=None, weight="latency")

    def pre_get_single_source_minimum_latency_path(self):
        # print "pre_get_single_source_minimum_latency_path"
        single_source_minimum_latency_path = {}
        for node in self.nodes():
            single_source_minimum_latency_path[node] = nx.single_source_dijkstra(
                self, source=node, cutoff=None, weight="latency"
            )
        self.single_source_minimum_latency_path = single_source_minimum_latency_path
        return single_source_minimum_latency_path

    def deploy_sfc(self, sfc, route_info):
        # with self.lock:
        if not route_info:
            print("route info is None")
            return
        if sfc.id not in self.sfc_dict:
            self.sfc_dict[sfc.id] = sfc
        if sfc.id not in self.sfc_route_info:
            self.sfc_route_info[sfc.id] = route_info
        list(route_info.items())
        for vnf_id, path in list(route_info.items()):
            if vnf_id == "dst":
                # self.nodes[sfc.dst.substrate_node]['sfc_vnf_list'].append((sfc.id, sfc.dst))
                tmp = self._get_node_attribute(sfc.dst.substrate_node, "sfc_vnf_list")
                tmp.append((sfc.id, sfc.dst))
                self._set_node_attribute(sfc.dst.substrate_node, sfc_vnf_list=tmp)

                continue
            vnf = sfc.get_vnf_by_id(vnf_id)
            # print path
            # self.nodes[path[0]]['sfc_vnf_list'].append((sfc.id, vnf))
            tmp = self._get_node_attribute(path[0], "sfc_vnf_list")
            tmp.append((sfc.id, vnf))
            self._set_node_attribute(path[0], sfc_vnf_list=tmp)
        # sfc.start()

    def _set_node_attribute(self, node_id, **attr):
        self.add_node(node_id, **attr)
        return

    def get_node_sfc_vnf_list(self, node_id):
        return self._get_node_attribute(node_id, "sfc_vnf_list")

    def reset_vnf_cpu_request(self, node_id, sfc_id, vnf_id, income_bw=0, outcome_bw=0):
        # Passo 1: Recupere a lista de VNFs associadas ao nó
        sfc_vnf_list = self._get_node_attribute(node_id, "sfc_vnf_list")

        # Passo 2: Localize a VNF específica dentro da lista
        for i, (current_sfc_id, vnf) in enumerate(sfc_vnf_list):
            if current_sfc_id == sfc_id and vnf.id == vnf_id:
                # Passo 3: Zere os atributos cpu_request e cache_request da VNF
                vnf.cpu_request = 0
                vnf.cache_request = 0

                # Passo 4: Atualize as interfaces income e outcome
                vnf.set_income_interface_bandwidth(income_bw)
                vnf.set_outcome_interface_bandwidth(outcome_bw)

                # Passo 5: Atualize a lista no nó
                sfc_vnf_list[i] = (current_sfc_id, vnf)  # Sobrescreve com a VNF atualizada
                self._set_node_attribute(node_id, sfc_vnf_list=sfc_vnf_list)
                break
                return  # Encerra após a modificação bem-sucedida
        sfc_vnf_list = self._get_node_attribute(node_id, "sfc_vnf_list")
        print()
        return
        # Caso a VNF não seja encontrada, levanta um erro
        raise ValueError(f"VNF {vnf_id} da SFC {sfc_id} não encontrada no nó {node_id}.")

    def undeploy_sfc(self, sfc_id):
        if sfc_id not in self.sfc_route_info or sfc_id not in self.sfc_dict:
            print("Not in both")
        else:
            # after undeployed sfc, sfc need to be deleted from following dicts
            route_info = self.sfc_route_info[sfc_id]
            sfc = self.sfc_dict[sfc_id]
            # sfc.stop()
            # recovery cpu resources
            # no need to actually modify used and free cpu resource.
            # the substrate network will be updated once the vnf removed from node
            for vnf_id, _path in list(route_info.items()):
                if vnf_id == "dst":
                    self.nodes[sfc.dst.substrate_node]["sfc_vnf_list"].remove((sfc_id, sfc.dst))
                    continue
                save_node = None
                for node in self.nodes():
                    for sfc_vnfs in self.get_node_sfc_vnf_list(node):
                        for vnf in sfc_vnfs:
                            if vnf == sfc.get_vnf_by_id(vnf_id):
                                save_node = node
                                break  # Para de procurar assim que encontra
                    if save_node is not None:
                        break  # Para de procurar em outros nós quando encontra
                if save_node is not None:
                    vnf_to_remove = sfc.get_vnf_by_id(vnf_id)
                    self.nodes[save_node]["sfc_vnf_list"].remove((sfc_id, vnf_to_remove))
                else:
                    print(f"VNF {vnf_id} não encontrado.")

                # print('#########  sfc net.py  ###########')
            self.sfc_route_info.pop(sfc_id, None)
            self.sfc_dict.pop(sfc_id, None)

            if self.verbose:
                print("sfc size", len(self.sfc_dict))

            # if(len(self.sfc_dict) == 0):
            # print('quitting')
            # return
            # quit()
            # recovery bandwidth resources

    def update_network_state(self):
        self.update_nodes_state()
        self.update_bandwidth_state()

    def get_sfc_id_resource_saved(self, sfc_id):
        cpu_saved = 0
        cache_saved = 0

        total_cpu_req = 0
        total_cache_req = 0

        route_info = self.sfc_route_info[sfc_id]
        for key, item in route_info.items():
            if key not in ["src", "dst"]:
                node_vnf_list = self.get_node_sfc_vnf_list(item[0])
                vnf = node_vnf_list[0][1]
                vnf_id = vnf.id
                if vnf_id in list(map(lambda sf: sf.id, self.shared_sfs[item[0]])):
                    cpu_saved += vnf.get_cpu_request()
                    cache_saved += vnf.get_cache_request()

                total_cpu_req += vnf.get_cpu_request()
                total_cache_req += vnf.get_cache_request()
        conta = (cpu_saved + cache_saved) / (total_cpu_req + total_cache_req)
        return conta

    def update_nodes_state(self):
        pattern = re.compile(r"_p")
        self.reset_shared_sfs()

        # Estruturas adicionadas
        total_cpu_used = 0
        total_cache_used = 0

        cpu_saved = 0
        cache_saved = 0
        shared_vnfs_count = 0
        shared_vnfs_details = {}
        all_vnfs_on_nodes = {}

        for node in self.nodes():
            cpu_used = 0
            cache_used = 0

            cpu_rel = cpu_used
            cache_rel = cache_used

            cpu_capacity = self.get_node_cpu_capacity(node)
            cache_capacity = self.get_node_cache_capacity(node)
            self.get_node_sfc_vnf_list(node)
            for sfc_vnf in self.get_node_sfc_vnf_list(node):
                sfc_id = sfc_vnf[0]
                is_backup = (sfc_id.split("_")[2]) == "backup"
                vnf_id = sfc_vnf[1].id

                if self.shareable_node:
                    if vnf_id not in list(map(lambda sf: sf.id, self.shared_sfs[node])):
                        cpu_used += sfc_vnf[1].get_cpu_request()
                        cache_used += sfc_vnf[1].get_cache_request()
                        if not is_backup:
                            cpu_rel = cpu_used
                            cache_rel = cache_used
                        if re.search(pattern, vnf_id) is None and vnf_id not in ("src", "dst"):
                            self.shared_sfs[node].append(sfc_vnf[1])
                    else:
                        # Lógica para contabilizar recursos poupados e detalhes das
                        # VNFs compartilhadas
                        cpu_request = sfc_vnf[1].get_cpu_request()
                        cache_request = sfc_vnf[1].get_cache_request()
                        cpu_saved += cpu_request
                        cache_saved += cache_request
                        shared_vnfs_count += 1
                        if vnf_id in shared_vnfs_details:
                            shared_vnfs_details[vnf_id].add(
                                sfc_vnf[0]
                            )  # Adiciona SFC se já existe
                        else:
                            shared_vnfs_details[vnf_id] = {sfc_vnf[0]}
                else:
                    cpu_used += sfc_vnf[1].get_cpu_request()
                    cache_used += sfc_vnf[1].get_cache_request()
                    if not is_backup:
                        cpu_rel = cpu_rel + cpu_used
                        cache_rel = cache_rel + cache_used
                # Atualiza o dicionário de todas as VNFs nos nós
                if vnf_id not in all_vnfs_on_nodes:
                    all_vnfs_on_nodes[vnf_id] = [node]
                elif node not in all_vnfs_on_nodes[vnf_id]:
                    all_vnfs_on_nodes[vnf_id].append(node)

            time = 0.01  # padrão por simplificação

            base_failure_rate = 1 - self.get_node_reliability(node)

            alpha_base = 1000
            alpha_cpu = 1
            alpha_mem = 1.5

            lambda_total = (
                alpha_base * base_failure_rate + alpha_cpu * cpu_rel + alpha_mem * cache_rel
            )

            reliability = math.exp(-lambda_total * time)
            p_falha = 1 - reliability

            self.nodes_reliability[node] = p_falha

            self.set_node_cpu_used(node, cpu_used)
            cpu_free = cpu_capacity - cpu_used
            self.set_node_cpu_free(node, cpu_free)
            total_cpu_used = total_cpu_used + cpu_used

            self.set_node_cache_used(node, cache_used)
            cache_free = cache_capacity - cache_used
            total_cache_used = total_cache_used + cache_used
            self.set_node_cache_free(node, cache_free)

        self.cpu_saved = cpu_saved
        self.cache_saved = cache_saved
        self.shared_vnfs_count = shared_vnfs_count
        self.total_cpu_used = total_cpu_used
        self.total_cache_used = total_cache_used

        # Após o loop, imprime os resultados adicionais
        # print(f"CPU poupado: {}, Cache poupado: {cache_saved}")
        # print(f"VNFs compartilhadas: {shared_vnfs_count}, Detalhes: {shared_vnfs_details}")
        # print(f"IDs de todas as VNFs nos nós: {all_vnfs_on_nodes}")

        total_cpu_used = 0
        total_cache_used = 0

        # for node in self.nodes():
        #     node_cpu_used = self.get_node_cpu_used(node)
        #     node_cpu_capacity = self.get_node_cpu_capacity(node)
        #     node_cache_used =  self.get_node_cache_used(node)
        #     node_cache_capacity = self.get_node_cache_capacity(node)

        #     total_cpu_used += node_cpu_used
        #     total_cpu_capacity += node_cpu_capacity
        #     total_cache_used += node_cache_used
        #     total_cache_capacity += node_cache_capacity

        # self.total_cpu_used = total_cpu_used
        # self.total_cpu_capacity = total_cpu_capacity
        # self.total_cache_used = total_cache_used
        # self.total_cache_capacity = total_cache_capacity

        # if node_cpu_used != 0:
        #     practical_cpu_used += node_cpu_used
        #     practical_cpu_capacity += node_cpu_capacity
        # if node_cache_used != 0:
        #     practical_cache_used += node_cache_used
        #     practical_cache_capacity += node_cache_capacity
        # self.practical_cpu_used = practical_cpu_used
        # self.practical_cpu_capacity = practical_cpu_capacity
        # self.practical_cache_used = practical_cache_used
        # self.practical_cache_capacity = practical_cache_capacity

    def update_bandwidth_state(self):
        pattern = re.compile(r"_p")
        # print(self.sfs_flux_info)
        self.reset_sfs_flux_info()
        for edge in self.edges():
            self.reset_bandwidth(edge[0], edge[1])
        for node in self.nodes():
            for sfc_vnf in self.get_node_sfc_vnf_list(node):
                sfc_id = sfc_vnf[0]
                # Here, we are assuming a sfc that is not in the
                # dict indicated a situation where a sfc is no longer present,
                # but there's still information about it because one of its
                # sfs is being shared by someone else.
                if sfc_id not in self.sfc_dict:
                    continue
                try:
                    sfc = self.get_sfc_by_id(sfc_id)
                except KeyError:  # <-- CORRIGIDO: Exceção Tipada
                    continue

                vnf = sfc_vnf[1]
                if vnf.id == "dst":
                    continue
                route_info = self.sfc_route_info[sfc_id]
                path = route_info[vnf.id]
                # if next sf is in the same node
                # then there's no bandwidth to be spent
                if len(path) <= 1:
                    continue
                for i in range(len(path) - 1):
                    if self.shareable_band:
                        link_sfs_info = self.sfs_flux_info[(path[i], path[i + 1])]
                        flux_ids = [flux_vnf.id for flux_vnf in link_sfs_info]
                        # if link does not contain a shareable flux then we allocate bandwidth
                        if vnf.id not in flux_ids:
                            self.allocate_bandwidth_resource(
                                path[i],
                                path[i + 1],
                                sfc.get_link_bandwidth_request(vnf.id, vnf.next_vnf.id),
                            )
                            if re.search(pattern, vnf.id) is None and vnf.id != "src":
                                self.sfs_flux_info[(path[i], path[i + 1])].append(vnf)
                        else:
                            pass
                    else:
                        self.allocate_bandwidth_resource(
                            path[i],
                            path[i + 1],
                            sfc.get_link_bandwidth_request(vnf.id, vnf.next_vnf.id),
                        )
        total_bandwidth_used = 0
        total_bandwidth_capacity = 0

        for edge in self.edges():
            total_bandwidth_used += self.get_link_bandwidth_used(edge[0], edge[1])
            total_bandwidth_capacity += self.get_link_bandwidth_capacity(edge[0], edge[1])

            # if  self.get_link_bandwidth_used(edge[0], edge[1]) != 0:
            #     practical_bw_used  += self.get_link_bandwidth_used(edge[0], edge[1])
            #     practical_bw_capacity += self.get_link_bandwidth_capacity(edge[0], edge[1])

        self.total_bandwidth_capacity = total_bandwidth_capacity
        self.total_bandwidth_used = total_bandwidth_used

        # self.practical_bw_capacity = practical_bw_capacity
        # self.practical_bw_used = practical_bw_used

    def print_out_nodes_information(self, failure_cpu=None, failure_cache=None):
        # for node in self.nodes():
        #     node_id = node
        #     cpu_used = self.get_node_cpu_used(node)
        #     cpu_free = self.get_node_cpu_free(node)
        #     cpu_capacity = self.get_node_cpu_capacity(node)
        #     sfc_vnf_list = self.get_node_sfc_vnf_list(node)
        # print "node id:", node_id, ":", "CPU: used:", cpu_used, "free:", cpu_free,
        # "capacity:", cpu_capacity, "vnf", sfc_vnf_list
        # print "total cpu used: ", self.total_cpu_used, "total cpu capacity: ",
        # self.total_cpu_capacity
        if failure_cpu is None:
            print(
                "CPU       utilization: ",
                str(round(self.total_cpu_used * 1.0 / self.total_cpu_capacity * 100, 3)) + "%",
            )
        else:
            print(
                "CPU       utilization: ",
                str(round(self.total_cpu_used * 1.0 / self.total_cpu_capacity * 100, 3)) + "%",
                end=" ",
            )
            # print(f"     Failure for CPU: {failure_cpu}%")
        # print(("CPU over utilization: ",
        # str(self.get_cpu_overloaded_utilization_rate()*100) +'%'))
        if failure_cache is None:
            print(
                "Cache     utilization: ",
                str(round(self.total_cache_used * 1.0 / self.total_cache_capacity * 100, 3)) + "%",
            )
        else:
            print(
                "Cache     utilization: ",
                str(round(self.total_cache_used * 1.0 / self.total_cache_capacity * 100, 3)) + "%",
                end=" ",
            )
            # print(f"     Failure for Cache: {failure_cache}%")

    def print_out_acceptance_information(self, success_arr):
        if len(success_arr) != 0:
            media = np.mean(success_arr)
            media_porc = media * 100
            print("Acceptance: ", str(round(media_porc, 3)) + "%", end=" ")

    def print_out_edges_information(self, failure_band=None):
        for edge in self.edges():
            self.get_link_bandwidth_capacity(edge[0], edge[1])
            self.get_link_bandwidth_free(edge[0], edge[1])
            self.get_link_bandwidth_used(edge[0], edge[1])
            self.get_link_latency(edge[0], edge[1])
            # print "edge:", edge, ":", "BW: used:", ud, "free:", fr, "capacity:", cp,
            # "latency:", lt
        # print "total bandwidth used: ", self.total_bandwidth_used,
        # "total bandwidth capacity: ", self.total_bandwidth_capacity
        if failure_band is None:
            print(
                "Bandwidth utilization: ",
                str(
                    round(self.total_bandwidth_used * 1.0 / self.total_bandwidth_capacity * 100, 3)
                )
                + "%",
            )
        else:
            print(
                "Bandwidth utilization: ",
                str(
                    round(self.total_bandwidth_used * 1.0 / self.total_bandwidth_capacity * 100, 3)
                )
                + "%",
                end=" ",
            )
            print(f"     Failure for Band: {failure_band}%")
        # print(("Bandwidth utilization(used): ", str(self.get_network_utility()*100)+'%'))

    def update(self):
        self.update_network_state()

    def get_cpu_utilization_rate(self):
        return self.total_cpu_used * 1.0 / self.total_cpu_capacity

    def get_cache_utilization_rate(self):
        return self.total_cache_used * 1.0 / self.total_cache_capacity

    def get_resilient_cpu_utilization(self):
        return self.total_cpu_used * 1.0 / 1200

    def get_resilient_cache_utilization(self):
        return self.total_cache_used * 1.0 / 1200

    def get_resilient_bandwidth_utilization(self):
        return self.total_bandwidth_used * 1.0 / 35000

    def get_active_servers_cpu_rate(self):
        if self.practical_cpu_capacity != 0:
            return self.practical_cpu_used * 1.0 / self.practical_cpu_capacity
        else:
            return 0

    def get_active_servers_cache_rate(self):
        if self.practical_cache_capacity != 0:
            return self.practical_cache_used * 1.0 / self.practical_cache_capacity
        else:
            return 0

    def get_active_links_bw_rate(self):
        if self.practical_bw_capacity != 0:
            return self.practical_bw_used * 1.0 / self.practical_bw_capacity
        else:
            return 0

    def get_bandwidth_utilization_rate(self):
        self.update()
        return self.total_bandwidth_used * 1.0 / self.total_bandwidth_capacity

    def get_cpu_overloaded_utilization_rate(self):
        self.update()
        tmp_used = 0
        cpu_capacity = 100
        for node in self.nodes():
            if self.get_node_cpu_used(node) > cpu_capacity:
                tmp_used += self.get_node_cpu_used(node)
            # print(self.get_node_cpu_used(node))
        return tmp_used * 1.0 / cpu_capacity

    def get_network_utility(self):
        self.update()
        tmp_used = 0
        tmp_capacity = 0
        for edge in self.edges():
            if self.get_link_bandwidth_used(edge[0], edge[1]) > 0:
                tmp_used = tmp_used + self.get_link_bandwidth_used(edge[0], edge[1])
                tmp_capacity = tmp_capacity + self.get_link_bandwidth_capacity(edge[0], edge[1])
        if tmp_capacity == 0:
            return 0
        return tmp_used * 1.0 / tmp_capacity


if __name__ == "__main__":
    substrate_network = Net()
    substrate_network.init_bandwidth_capacity(1, 6, 100)
    substrate_network.init_bandwidth_capacity(1, 2, 100)
    substrate_network.init_bandwidth_capacity(2, 3, 100)
    substrate_network.init_bandwidth_capacity(3, 4, 100)
    substrate_network.init_bandwidth_capacity(4, 5, 100)
    substrate_network.init_bandwidth_capacity(5, 6, 100)
    substrate_network.init_bandwidth_capacity(2, 6, 100)
    substrate_network.init_bandwidth_capacity(2, 5, 100)
    substrate_network.init_bandwidth_capacity(3, 5, 100)

    substrate_network.init_link_latency(1, 6, 2)
    substrate_network.init_link_latency(1, 2, 2)
    substrate_network.init_link_latency(2, 3, 2)
    substrate_network.init_link_latency(3, 4, 2)
    substrate_network.init_link_latency(4, 5, 2)
    substrate_network.init_link_latency(5, 6, 2)
    substrate_network.init_link_latency(2, 6, 2)
    substrate_network.init_link_latency(2, 5, 2)
    substrate_network.init_link_latency(3, 5, 2)

    substrate_network.init_node_cpu_capacity(1, 100)
    substrate_network.init_node_cpu_capacity(2, 100)
    substrate_network.init_node_cpu_capacity(3, 100)
    substrate_network.init_node_cpu_capacity(4, 100)
    substrate_network.init_node_cpu_capacity(5, 100)
    substrate_network.init_node_cpu_capacity(6, 100)

    substrate_network.init_node_cache_capacity(1, 100)
    substrate_network.init_node_cache_capacity(2, 100)
    substrate_network.init_node_cache_capacity(3, 100)
    substrate_network.init_node_cache_capacity(4, 100)
    substrate_network.init_node_cache_capacity(5, 100)
    substrate_network.init_node_cache_capacity(6, 100)

    # print(substrate_network.nodes())

    print(substrate_network.nodes())
    print(substrate_network.edges())

    print(substrate_network.get_link_latency(1, 2))
    # print(substrate_network.get_link_latency(2, 1))
    print(substrate_network.get_link_bandwidth_free(1, 2))
    # print(substrate_network.get_link_bandwidth_free(2, 1))
    print([n for n in substrate_network.get_neighbours(6)])
    # print substrate_network.all_shortest_paths()
    print("shortestpath")
    print(substrate_network.get_shortest_paths(1, 5, weight="latency"))
    path = substrate_network.get_minimum_latency_path(1, 5)
    print(path)
    print(substrate_network.get_minimum_free_bandwidth(path))
    print("single source")
    print(substrate_network.get_single_source_minimum_latency_path(2))
    print(substrate_network.edges())
    substrate_network.allocate_bandwidth_resource_path(path, 10)
    print(substrate_network.edges())
    print("Finished")
    print(substrate_network.get_link_bandwidth_capacity(3, 5))
    print(substrate_network.get_link_bandwidth_capacity(5, 3))
    print(substrate_network.get_node_cache_capacity(2))
    print("****************************************************")
    print(substrate_network.get_shortest_path_length(1, 6))

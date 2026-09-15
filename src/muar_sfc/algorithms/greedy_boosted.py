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

from muar_sfc.algorithms.algorithm import Algorithm
from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
    get_available_shortest_path,
    get_link_latency,
    get_shortest_path,
    get_shortest_path_length,
)
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

SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")


class GreedyOptAlgorithm(Algorithm):
    """Greedy Algorithm.
    This algorithm starts from the substrate network node which hosts src of an SFC,
    checks its neighbor nodes, finds the neighbor node with a shortest latency edge,
    and use the node to host the vnf.
    The algorithm greedily finds all nodes for hosting vnf.
    Finally, the algorithm finds a shortest path from the substrate node who hosts
    the last vnf in the SFC to the substrate node who hosts dst of the SFC.

    Deploy VNF one by one, with a shortest path from the node to the previous
    substrate node.
    """

    def __init__(self):
        self.name = "Greedy Algorithm"
        self.sfc = None
        self.route_info = None
        self.latency = None
        self.mono = False
        self.old_greedy = True
        self.forbidden_matches = {}
        self.is_backup = False
        self.graph = None
        self.single_source_minimum_latency_path = None

    def clear_all(self):
        self.sfc = None
        self.graph = None
        self.route_info = None
        self.latency = None
        self.is_backup = False
        self.single_source_minimum_latency_path = None

    def install_substrate_network(self, graph, shareable_sfs=None):
        if shareable_sfs is None:
            shareable_sfs = []
        self.graph = graph
        return self.graph

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.route_info = {}
        self.latency = None
        return self.sfc
        # is_backup = True if sfc.id.split("_")[2] == 'backup' else False
        # self.is_backup = is_backup
        # if is_backup:
        #     split = sfc.id.split("_")
        #     original_sfc_id = f"{split[0]}_{split[1]}_{split[3]}_{split[4]}"
        #     route_info = self.substrate_network.sfc_route_info[original_sfc_id]
        #     for vnf, rf in route_info.items():
        #         if vnf not in ['src','dst']:
        #             node_used = route_info[vnf][0]
        #             correct_name =  vnf + "_b"
        #             self.forbidden_matches[correct_name] = node_used
        # return self.sfc

    def check_solution(self):
        if self.latency is None:
            return False
        if abs(self.latency) < 0.0001:  # Correção de erros de ponto flutuante
            self.latency = max(0, self.latency)
        if (
            not isinstance(self.latency, (int, float))
            or self.latency < 0
            or self.latency > self.sfc.get_latency_request()
            or not self.route_info
        ):
            return False
        return len(list(self.route_info.keys())) == 6

    def handle_failure(self):
        self.route_info = False
        self.latency = None

        # for vnf, server_forbidden in self.forbidden_matches.items():
        #     try:
        #         server_used = self.route_info[vnf][0]
        #         if server_used == server_forbidden:
        #             self.route_info = False
        #             self.latency = None
        #             return False
        #     except:
        #         self.route_info = False
        #         self.latency = None
        #         return False

    def start_algorithm(self):
        # logger.info("Start algorithm")
        self.algorithm()
        is_success = self.check_solution()
        if is_success:
            try:
                # logger.info("Finished algorithm, success")
                logger.debug(f"Route info greedyB: {self.route_info}")
                return True
            except Exception as e:
                logger.exception(f"Erro ao exibir sucesso do algoritmo GreedyOpt: {e}")
                self.handle_failure()
                return False
        else:
            self.handle_failure()
            # logger.info("End algorithm, failed")
            return False

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def algorithm(self):
        # Get src and dst vnf
        src_vnf = self.sfc.get_src_vnf()
        dst_vnf = self.sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advance
        src_substrate_node = self.sfc.get_substrate_node(src_vnf)
        dst_substrate_node = self.sfc.get_substrate_node(dst_vnf)

        route_info = {}

        latency = 0
        used_node = [dst_substrate_node, src_substrate_node]

        number_of_vnfs = self.sfc.get_number_of_vnfs()
        current_vnf = dst_vnf
        current_substrate_node = dst_substrate_node
        servers = list(self.graph.nodes())

        # Inicializa o dicionário de recursos dos servidores
        for server in servers:
            aux = self.graph.nodes[server]
            _ = aux["cpu_capacity"]

        server_resources = {
            server: {
                "cpu_capacity": self.graph.nodes[server]["cpu_capacity"],
                "cache_capacity": self.graph.nodes[server]["cache_capacity"],
                "cpu_used": self.graph.nodes[server]["cpu_used"],
                "cache_used": self.graph.nodes[server]["cache_used"],
                "reuse": [],
            }
            for server in servers
            if self.graph.nodes[server]["cpu_capacity"] > 0
            and self.graph.nodes[server]["cache_capacity"]
        }

        nodes_used = []
        servers_to_check = list(server_resources.keys())

        for _ in range(number_of_vnfs - 1, -1, -1):
            prev_vnf = current_vnf.get_previous_vnf()
            cpu_request = self.sfc.get_vnf_cpu_request(prev_vnf)
            cache_request = self.sfc.get_vnf_cache_request(prev_vnf)
            bandwidth_request = self.sfc.get_link_bandwidth_request(prev_vnf.id, current_vnf.id)

            min_path = []
            min_latency = float("inf")
            node = None
            random.shuffle(servers_to_check)
            for node_a in servers_to_check:
                if node_a == dst_substrate_node:
                    continue
                cpu_used = server_resources[node_a]["cpu_used"]
                cache_used = server_resources[node_a]["cache_used"]

                cpu_cap = server_resources[node_a]["cpu_capacity"]
                cache_cap = server_resources[node_a]["cache_capacity"]

                # Agora buscamos valores diretamente em server_resources:
                cpu_available = round(cpu_cap - cpu_used, 2)
                cache_available = round(cache_cap - cache_used, 2)

                if cpu_cap <= 0 or cache_cap <= 0:
                    continue

                if cpu_available <= 0 or cache_available <= 0:
                    continue

                if cpu_used + cpu_request > cpu_cap or cache_used + cache_request > cache_cap:
                    continue

                if cpu_request > cpu_available:
                    logger.debug("Node %s não tem CPU suficiente para %s", node_a, cpu_request)
                    continue
                if cache_request > cache_available:
                    logger.debug("Node %s não tem CACHE suficiente para %s", node_a, cache_request)
                    continue

                # path = get_shortest_path(self.graph, current_substrate_node, node_a)
                # if current_substrate_node == node_a or node_a in nodes_used:
                #     continue

                # Por segurança
                path = get_available_shortest_path(
                    self.graph, node_a, current_substrate_node, bandwidth_request
                )
                if not path:
                    continue
                # Verificando link (apenas se não for laço no mesmo nó)
                comp_latency = calculate_computational_latency(self.graph, node_a, prev_vnf)
                edge_latency = 0
                if len(path) > 1:
                    for u, v in zip(path[:-1], path[1:], strict=False):
                        edge_latency += calculate_latency_betwen_nodes(self.graph, u, v, prev_vnf)

                total_latency = comp_latency + edge_latency

                # Verifica se a latência desse caminho é a menor
                if total_latency < min_latency:
                    min_path = path
                    min_latency = total_latency
                    node = node_a

            # Se encontrou um nó para alocar
            if node is not None:
                nodes_used.append(node)
                # Atualizamos o dicionário de recursos
                server_resources[node]["cpu_used"] += cpu_request
                server_resources[node]["cache_used"] += cache_request

                # Se for usar a banda, você também decrementa a banda do enlace
                # se node != current_substrate_node, por exemplo
                # Ajuste do route_info e soma de latência
                if node == current_substrate_node:
                    route_info[prev_vnf.id] = [node]
                else:
                    route_info[prev_vnf.id] = min_path
                used_node.append(node)
                latency += min_latency
            else:
                logger.debug("Não foi possível alocar VNF")
                self.route_info = {}
                self.latency = None
                return False

            current_substrate_node = node
            current_vnf = prev_vnf

        try:
            path = get_shortest_path(self.graph, src_substrate_node, node)
            path_latency = get_shortest_path_length(self.graph, src_substrate_node, node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning(
                "Não há caminho entre src e primeira VNF: %s - %s", src_substrate_node, node
            )
            self.route_info = {}
            self.latency = None
            return False

        # Adicionamos esse path como 'src' no route_info
        route_info["src"] = path
        route_info["dst"] = []
        latency += path_latency

        # Define route_info e latency no objeto
        self.route_info = route_info
        self.latency = latency

        # Se você precisa fazer algum ajuste de latência baseado em edges do path:
        path = self.route_info["src"]
        for i in range(len(path) - 1):
            edge_latency = get_link_latency(self.graph, path[i], path[i + 1])
            self.latency = self.latency - edge_latency

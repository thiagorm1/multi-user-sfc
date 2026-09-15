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

import networkx as nx

from muar_sfc.algorithms.algorithm import Algorithm
from muar_sfc.config import ROOT_DIR

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler(ROOT_DIR / "logs" / "RodrigoAlg.log")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)




class Rodrigo(Algorithm):
    def __init__(self):
        self.name = "rodrigo"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.latency_request = 0
        self.latency_minus_dst = 0
        self.fail_for_band = 0
        self.fail_for_cache = 0
        self.fail_for_cpu = 0
        self.fail_resources = 0
        self.saved_band = 0
        self.saved_cpu = 0
        self.saved_cache = 0
        self.server_resources = 0
        self.services_requirements = 0
        self.G = 0
        self.services = 0
        self.cpu_factor = 1
        self.cache_factor = 1
        self.band_factor = 1
        self.boot_factor = 1

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
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        return self.sfc

    def set_costs(self, costs_parameters):
        self.cpu_weight = costs_parameters[0]
        self.cache_weight = costs_parameters[1]
        self.band_weight = costs_parameters[2]
        self.boot_weight = costs_parameters[3]

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def start_algorithm(self, shareable_sfs=None, **kwargs):
        substrate_network = self.substrate_network
        sfc = self.sfc
        # logger.info('Algorithm start')
        return self.algorithm(substrate_network, sfc, shareable_sfs)

    def algorithm(self, substrate_network, sfc, shareable_sfs=None):
        sfs_dict = self.sfc.vnfs_dict
        net_info = substrate_network

        server_resources = net_info._node

        shareable_sfs = (
            shareable_sfs
            if shareable_sfs is not None
            else {node_id: [] for node_id in server_resources}
        )

        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        sfc.get_substrate_node(src_vnf)
        dst = sfc.get_substrate_node(dst_vnf)

        # Inicialização da topologia da rede
        network_topology = net_info._adj

        # Criação do grafo representando a rede com capacidade de banda
        G = nx.Graph()
        for node, edges in network_topology.items():
            for target, edge_attr in edges.items():
                G.add_edge(node, target, bandwidth=edge_attr["bandwidth_free"], weight=1)

        service_requirements = {}

        services = []  # Lista para guardar os nomes

        for item in sfs_dict:
            nome = item["name"]
            services.append(nome)  # Adiciona o nome à lista de nomes
            service_requirements[nome] = {
                "CPU": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        service_requirements["dst"] = {"CPU": 0, "cache": 0, "out_bw": 0, "in_bw": 0}
        services = list(reversed(services))

        # Adicionando o campo 'reuse' no node_table
        for node_id, node_info in server_resources.items():
            # Inicializando o campo 'reuse' como uma lista vazia para cada nó
            node_info["reuse"] = []
            # Se existirem VNFs compartilháveis para este nó em shareable_sfs
            if node_id in shareable_sfs:
                # Iterar sobre cada VNF em shareable_sfs para este nó
                for vnf in shareable_sfs[node_id]:
                    # Adicionar o ID (nome) da VNF ao campo 'reuse' do nó
                    node_info["reuse"].append(vnf.id)  # Acessando o atributo 'id' da VNF

        route_info, latency = self.fit_path(
            G, service_requirements, server_resources, services, dst
        )

        if latency > self.latency_request:
            self.latency = None
            self.route_info = False
            return False
        else:
            self.latency = latency
            self.route_info = route_info
            return True

    # Início da função fit_path modificado
    def fit_path(self, G, service_requirements, server_resources, services, dst):
        found_path = True

        current_location = dst  # Initial location for the path fitting

        service_requirements_local = service_requirements
        server_resources_local = server_resources
        services_list = services

        allocation_results = {"dst": {"allocated_server": dst, "path": [], "cost": 0}}
        current_location = dst

        for service in services_list:
            bandwidth_requirement = service_requirements[service]["out_bw"]
            best_server, best_path, total_cost = self.find_best_server_for_service(
                G,
                server_resources_local,
                service_requirements_local,
                current_location,
                service,
                bandwidth_requirement,
            )
            if best_server is None:
                # Handle the case when no server is found
                print(f"No suitable server found for service {service}")
                found_path = False
                break

            allocation_results[service] = {
                "allocated_server": best_server,
                "path": best_path,
                "cost": total_cost,
            }
            current_location = best_server
            # Atualiza os recursos alocados no servidor escolhido
            if best_server:  # Verifica se um servidor foi escolhido
                server_resources[best_server]["cpu_used"] += service_requirements[service]["CPU"]
                server_resources[best_server]["cache_used"] += service_requirements[service][
                    "cache"
                ]

        if found_path:
            # ultima iteração para o src
            path_to_src = nx.dijkstra_path(G, current_location, 0, weight="weight")

            route_info = {
                key: list(reversed(value["path"])) for key, value in allocation_results.items()
            }

            # calculo da latencia antes do src
            total_latency = sum(len(path) - 1 for path in route_info.values() if path)

            route_info["src"] = list(reversed(path_to_src))

            # colocando o dst no final
            first_key, first_value = next(iter(route_info.items()))
            del route_info[first_key]
            route_info[first_key] = first_value

            return route_info, total_latency
        else:
            return False, 1000

    def find_best_server_for_service(
        self,
        G,
        server_resources,
        service_requirements,
        current_location,
        service,
        bandwidth_requirement,
    ):

        def calculate_latency_cost(distance):
            if distance <= self.latency_request:
                return 0
            else:
                return 1000

        paths = dict(nx.single_source_shortest_path_length(G, current_location, cutoff=6))
        min_cost = float("inf")
        best_server = None
        best_path = None
        cost_details = None  # Para armazenar detalhes do custo

        # Inclui o servidor atual na verificação para permitir reutilização sem movimento
        paths[current_location] = 0  # Custo de 'mover' para o mesmo servidor é 0

        for server, num_hops in paths.items():
            path = nx.shortest_path(G, current_location, server, weight="weight")
            if all(
                G[u][v]["bandwidth"] >= bandwidth_requirement
                for u, v in zip(path, path[1:], strict=False)
            ):
                reuse = service in server_resources[server]["reuse"]
                node_resource_cost = (
                    0 if reuse else 1
                )  # Assuming cpu_cost and cache_cost should always be the same

                # Initial setup
                boot_cost = 0
                cpu_required = 0 if reuse else service_requirements[service]["CPU"]
                cache_required = (
                    0 if reuse else service_requirements[service]["cache"]
                )  # Assuming there's a cache requirement

                # Calculate available resources
                available_cpu = server_resources[server][
                    "cpu_free"
                ]  # - server_resources[server]['cpu_used']
                available_cache = server_resources[server][
                    "cache_free"
                ]  # - server_resources[server]['cache_used']

                if server_resources[server]["cpu_used"] >= 17.27 or reuse:  # já estava ligado
                    boot_cost = 0
                elif (
                    server_resources[server]["cpu_used"] + cpu_required
                ) >= 17.27:  # tem que ligar
                    boot_cost = 1
                else:
                    boot_cost = 0

                if available_cpu >= cpu_required and available_cache >= cache_required:
                    bandwidth_cost = num_hops  # Custo de 1 por salto
                    latency_cost = calculate_latency_cost(num_hops)

                    total_cost = (
                        node_resource_cost * self.cpu_factor
                        + (node_resource_cost * self.cache_factor)
                        + (boot_cost * self.boot_factor)
                        + (bandwidth_cost * self.band_factor)
                        + latency_cost
                    )

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_server = server
                        best_path = path
                        # Armazena o detalhamento dos custos para o melhor servidor atual
                        cost_details = {
                            "cpu_cost": node_resource_cost,
                            "cache_cost": node_resource_cost,
                            "boot_cost": boot_cost,
                            "bandwidth_cost": bandwidth_cost,
                            "latency_cost": latency_cost,
                            "total_cost": total_cost,
                        }

                # Atualiza a banda usada ao longo do melhor caminho, se aplicável
        if (
            best_path and best_server != current_location
        ):  # Evita atualizar banda se alocação é no mesmo servidor
            for u, v in zip(best_path, best_path[1:], strict=False):
                G[u][v]["bandwidth"] -= bandwidth_requirement
                G[v][u]["bandwidth"] -= bandwidth_requirement  # Bidirecional

        return best_server, best_path, cost_details

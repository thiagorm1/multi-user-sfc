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

from muar_sfc.algorithms.algorithm import Algorithm
from muar_sfc.config import ROOT_DIR

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler(ROOT_DIR / "logs" / "MSF.log")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)



class Osfem(Algorithm):
    def __init__(self):
        self.name = "gr"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.latency_request = 0

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

        self.using_bit_rate = False
        self.bitrate_cut = 1.0
        self.bw_used = 0

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
        self.using_bit_rate = False
        self.bitrate_cut = 1.0
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

    def get_bit_rate_used(self):
        return self.bitrate_cut

    def is_using_bit_rate_cut(self):
        return self.using_bit_rate

    def get_transcode_bw(self):
        return self.bw_used

    def start_algorithm(self, shareable_sfs=None, **kwargs):
        substrate_network = self.substrate_network
        sfc = self.sfc
        # logger.info('Algorithm start')
        return self.algorithm(substrate_network, sfc, shareable_sfs)

    def algorithm(self, substrate_network, sfc, shareable_sfs=None):
        # Parte 1: Preparação dos dados de entrada
        sfs_dict = self.sfc.vnfs_dict
        net_info = substrate_network
        server_resources = net_info._node
        shareable_sfs = (
            shareable_sfs
            if shareable_sfs is not None
            else {node_id: [] for node_id in server_resources}
        )

        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        sfc.get_substrate_node(src_vnf)
        dst = sfc.get_substrate_node(dst_vnf)

        # Parte 4: Inicialização da topologia da rede
        network_topology = net_info._adj

        # Parte 5: Criação do grafo da rede
        G = self.create_network_graph(network_topology)

        # Parte 6: Preparação dos requisitos de serviço
        services, service_requirements = self.prepare_service_requirements(sfs_dict)

        self.configure_shareable_sfs(server_resources, shareable_sfs)

        # Parte 8: Encontrando a rota e calculando a latência
        bit_rate_trials = [1.0]
        is_success = False

        transcode_bw_used = 0

        # Supondo que bit_rate_trials e outras variáveis estejam definidas
        for bitrate in bit_rate_trials:
            service_requirements_altered = copy.deepcopy(service_requirements)

            # Percorre o dicionário e altera o valor de out_bw para chaves que começam com "EC_TC"
            for chave in service_requirements_altered:
                if chave.startswith("EC_TC"):
                    transcode_bw_used = service_requirements_altered[chave]["out_bw"] * bitrate
                    service_requirements_altered[chave]["out_bw"] = transcode_bw_used

            route_info, latency = self.find_best_allocation_for_sfc(
                G, service_requirements_altered, server_resources, services, dst
            )

            # Parte 9: Avaliação do resultado com base na latência
            is_success = self.evaluate_result(latency, route_info)

            # if bitrate == 0.3:
            #     print(bitrate)

            if is_success and bitrate != 1.0:
                self.using_bit_rate = True
                self.bitrate_cut = bitrate
            # elif is_success and bitrate == 1.0:
            #     print(bitrate)

            self.bw_used = transcode_bw_used
            if is_success:
                self.bitrate_cut = bitrate
                return is_success
            else:
                continue

        return is_success

    def create_network_graph(self, network_topology):
        import networkx as nx

        G = nx.Graph()
        for node, edges in network_topology.items():
            for target, edge_attr in edges.items():
                G.add_edge(node, target, bandwidth=edge_attr["bandwidth_free"], weight=1)
        return G

    def prepare_service_requirements(self, sfs_dict):
        service_requirements = {}
        services = []  # Lista para guardar os nomes dos serviços
        for item in sfs_dict:
            nome = item["name"]
            services.append(nome)  # Adiciona o nome à lista
            service_requirements[nome] = {
                "CPU": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }
        service_requirements["dst"] = {"CPU": 0, "cache": 0, "out_bw": 0, "in_bw": 0}
        services = list(reversed(services))
        return services, service_requirements

    def configure_shareable_sfs(self, server_resources, shareable_sfs):
        for node_id, node_info in server_resources.items():
            node_info["reuse"] = []  # Inicializa o campo 'reuse'
            if node_id in shareable_sfs:
                for vnf in shareable_sfs[node_id]:
                    node_info["reuse"].append(vnf.id)

    def evaluate_result(self, latency, route_info):
        if latency > self.latency_request:
            self.latency = None
            self.route_info = False
            return False
        else:
            self.latency = latency
            self.route_info = route_info
            return True

    # Modificando a função de alocação para usar a nova lógica de exploração
    def find_best_allocation_for_sfc(
        self, G, service_requirements, server_resources, services, dst
    ):
        allocation_results = {"dst": {"allocated_server": dst, "path": [], "cost": 0}}
        current_location = dst  # começa a alocação de trás pra frente
        success = True

        for i, service in enumerate(services):
            # Verifica se há um próximo serviço na lista usando operador ternário
            next_service = services[i + 1] if i + 1 < len(services) else None

            try:
                # Chama a função com o serviço atual e o próximo serviço (ou None)
                best_server, best_path, cost_details = (
                    self.find_best_server_for_service_with_exploration(
                        G,
                        server_resources,
                        service_requirements,
                        current_location,
                        service,
                        next_service,
                    )
                )
            except Exception as e:
                print("***************************************************")
                print("***************************************************")
                print()
                print(f"Erro na alocação do osfem: {e}")
                print()
                print("***************************************************")
                print("***************************************************")

            allocation_results[service] = {
                "allocated_server": best_server,
                "path": best_path,
                "cost": cost_details,
            }

            current_location = best_server

            # Se existe um servidor ótimo
            if best_server:  # Verifica se um servidor foi escolhido
                server_resources[best_server]["cpu_used"] += service_requirements[service]["CPU"]
                server_resources[best_server]["cache_used"] += service_requirements[service][
                    "cache"
                ]
            else:
                # print(f"Falha.")
                success = False
                break

        if not success:
            return [], 1000

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

    def find_best_server_for_service_with_exploration(
        self,
        G,
        server_resources,
        service_requirements,
        current_location,
        service,
        next_service,
        exploration_margin=1,
    ):
        best_cost, best_candidate, candidates = self.find_candidates_serves_for_sf(
            G, server_resources, service_requirements, current_location, service
        )

        if best_cost == float("inf") or best_candidate == float("inf") or candidates is None:
            return False, False, False

        exploration_threshold = best_cost * (1 + exploration_margin)
        filtered_candidates = [
            candidate for candidate in candidates if candidate[2] <= exploration_threshold
        ]

        if next_service is not None:
            best_c_cost = float("inf")
            for candidato in filtered_candidates:
                (
                    candidate_future_cost,
                    _,
                    _,
                ) = self.find_candidates_serves_for_sf(
                    G, server_resources, service_requirements, candidato[0], next_service
                )
                candidate_atual_cost = candidato[3]["total_cost"]
                custo_conjunto = candidate_atual_cost + candidate_future_cost

                if custo_conjunto < best_c_cost:
                    best_c_cost = custo_conjunto
                    best_candidate = candidato

        server_choose = best_candidate[0]
        path_to = best_candidate[1]
        min_cost = best_candidate[2]

        return server_choose, path_to, min_cost

    def find_candidates_serves_for_sf(
        self, G, server_resources, service_requirements, current_location, service
    ):
        paths = dict(nx.single_source_shortest_path_length(G, current_location, cutoff=10))
        paths[current_location] = 0  # Custo de 'mover' para o mesmo servidor é 0

        candidates = []
        best_cost = float("inf")

        # Função para calcular o custo total
        def calculate_total_cost(node_resource_cost, boot_cost, bandwidth_cost, latency_cost):
            return (
                (node_resource_cost * self.cpu_factor)
                + (node_resource_cost * self.cache_factor)
                + (boot_cost * self.boot_factor)
                + (bandwidth_cost * self.band_factor)
                + latency_cost
            )

        # Função para verificar disponibilidade de largura de banda e recursos do servidor
        def check_resources(server, path, bandwidth_requirement, cpu_required, cache_required):
            has_bandwidth = all(
                G[u][v]["bandwidth"] >= bandwidth_requirement
                for u, v in zip(path, path[1:], strict=False)
            )
            if has_bandwidth:
                available_cpu = server_resources[server]["cpu_free"]
                available_cache = server_resources[server]["cache_free"]
                if available_cpu >= cpu_required and available_cache >= cache_required:
                    return True
            return False

        bandwidth_requirement = service_requirements[service]["out_bw"]
        for server, num_hops in paths.items():
            path = nx.shortest_path(G, current_location, server, weight="weight")
            reuse = service in server_resources[server]["reuse"]
            node_resource_cost = (
                0 if reuse else 1
            )  # Assuming cpu_cost and cache_cost should always be the same

            boot_cost = 0
            cpu_required = 0 if reuse else service_requirements[service]["CPU"]
            cache_required = 0 if reuse else service_requirements[service]["cache"]

            if server_resources[server]["cpu_used"] >= 17.27 or reuse:
                boot_cost = 0
            elif (server_resources[server]["cpu_used"] + cpu_required) >= 17.27:
                boot_cost = 1
            else:
                boot_cost = 0

            if check_resources(server, path, bandwidth_requirement, cpu_required, cache_required):
                bandwidth_cost = num_hops  # Custo de 1 por salto
                latency_cost = self.calculate_latency_cost(num_hops)
                total_cost = calculate_total_cost(
                    node_resource_cost, boot_cost, bandwidth_cost, latency_cost
                )

                if total_cost < best_cost:
                    best_cost = total_cost

                candidates.append(
                    (
                        server,
                        path,
                        total_cost,
                        {
                            "cpu_cost": node_resource_cost,
                            "cache_cost": node_resource_cost,
                            "boot_cost": boot_cost,
                            "bandwidth_cost": bandwidth_cost,
                            "latency_cost": latency_cost,
                            "total_cost": total_cost,
                        },
                    )
                )
            else:
                # print("Sem recurso")
                pass
            # Se encontrou candidatos viáveis, não tenta com bitrate menor
            if best_cost < float("inf"):
                break

        best_candidate = min(candidates, key=lambda x: x[2], default=(None, None, None, None))

        return best_cost, best_candidate, candidates

    def calculate_latency_cost(self, distance):
        if distance <= self.latency_request:
            return 0
        else:
            return 1000

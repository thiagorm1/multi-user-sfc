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


class Goku(Algorithm):
    def __init__(self):
        self.name = "goku"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.latency_request = 0
        self.fail_reason = None

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

        self.cpu_factor = 2
        self.cache_factor = 2
        self.band_factor = 1.0
        self.boot_factor = 0

        self.using_bit_rate = False
        self.bitrate_cut = 1.0

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
        # self.single_source_minimum_latency_path = (
        #     self.substrate_network.single_source_minimum_latency_path
        # )
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        self.using_bit_rate = False
        self.bitrate_cut = 1.0
        return self.sfc

    # def check_sfc(self,sfc):
    #     backup = False
    #     sfc_id = sfc.id
    #     is_backup = True if int(sfc_id.split("_")[-1]) % 2 == 0 else False

    #     new_sfc = 0
    #     if is_backup == True:
    #         prefixo, x = sfc_id.rsplit('_', 1)
    #         real_sfc_id = f"{prefixo}_{int(x) - 1}"
    #         route_info = self.substrate_network.sfc_route_info

    #         if real_sfc_id in list(route_info.keys()):
    #             route_info = route_info[real_sfc_id]
    #         else:
    #             return sfc
    #         servers_used = list(set([
    #             item for chave, valor in route_info.items()
    #             if chave not in ['src', 'dst'] for item in valor
    #         ]))
    #         return new_sfc
    #     else:
    #         return sfc

    def set_costs(self, costs_parameters):
        self.cpu_factor = costs_parameters[0]
        self.cache_factor = costs_parameters[1]
        self.band_factor = costs_parameters[2]

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def get_fail_reason(self):
        return self.fail_reason

    def get_bit_rate_used(self):
        return self.bitrate_cut

    def is_using_bit_rate_cut(self):
        return self.using_bit_rate

    def start_algorithm(self, shareable_sfs=None, **kwargs):
        substrate_network = self.substrate_network
        sfc = self.sfc
        # Retorna diretamente o resultado booleano (resolve SIM103)
        return bool(self.algorithm(substrate_network, sfc, shareable_sfs))

    def set_nodes_resources(self, substrate_network, shareable_sfs):
        net_info = substrate_network
        old_server_resources = net_info._node
        servers = old_server_resources.keys()

        # Inicializa o dicionário de recursos dos servidores
        server_resources = {
            server: {
                "cpu_capacity": round(
                    self.substrate_network.get_node_cpu_capacity(server), 3
                ),
                "cache_capacity": round(
                    self.substrate_network.get_node_cache_capacity(server), 3
                ),
                "cpu_used": round(
                    self.substrate_network.get_node_cpu_used(server), 3
                ),
                "cache_used": round(
                    self.substrate_network.get_node_cache_used(server), 3
                ),
                "cpu_free": round(
                    self.substrate_network.get_node_cpu_free(server), 3
                ),
                "cache_free": round(
                    self.substrate_network.get_node_cache_free(server), 3
                ),
                "position": old_server_resources[server]["position"],
                "reuse": [],
            }
            for server in servers
        }

        # Preenche o campo 'reuse' para os servidores com SFs compartilháveis
        for node_id, node_info in server_resources.items():
            if node_id in shareable_sfs:
                node_info["reuse"] = [vnf.id for vnf in shareable_sfs[node_id]]

        return server_resources

    def algorithm(self, substrate_network, sfc, shareable_sfs):
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()
        sfc.get_substrate_node(src_vnf)
        dst = sfc.get_substrate_node(dst_vnf)

        nodes_resource = self.set_nodes_resources(substrate_network, shareable_sfs)
        network_links = copy.deepcopy(substrate_network._adj)

        G = self.create_network_graph(network_links)

        sfs_dict = self.sfc.vnfs_dict
        services, service_requirements = self.prepare_service_requirements(sfs_dict)

        # Parte 8: Encontrando a rota e calculando a latência
        # bit_rate_trials = [1.0]
        is_success = False

        # # Supondo que bit_rate_trials e outras variáveis estejam definidas
        # for bitrate in bit_rate_trials:
        # service_requirements_altered = copy.deepcopy(service_requirements)

        # Percorre o dicionário e altera o valor de out_bw
        # for chave in service_requirements_altered:
        #     if chave.startswith('EC_TC'):
        #         service_requirements_altered[chave]['out_bw'] = (
        #             service_requirements_altered[chave]['out_bw'] * bitrate
        #         )

        route_info, latency = self.find_best_allocation_for_sfc(
            G, service_requirements, nodes_resource, network_links, services, dst
        )

        # if bitrate == 0.3:
        #     print(bitrate)

        # if is_success and bitrate != 1.0:
        #     self.using_bit_rate = True
        #     self.bitrate_cut = bitrate
        # elif is_success and bitrate == 1.0:
        #     print(bitrate)

        return self.evaluate_result(latency, route_info)
        # self.bitrate_cut = bitrate
        # return is_success
        # else:
        #     continue

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
                "latency": item["latency"],
            }
        service_requirements["dst"] = {
            "CPU": 0,
            "cache": 0,
            "out_bw": 0,
            "in_bw": 0,
            "latency": 0,
        }
        services = list(reversed(services))
        return services, service_requirements

    def evaluate_result(self, latency, route_info):
        if latency is None:
            self.fail_reason = "resource"
            self.route_info = False
            self.latency = None
            return False
        if latency > self.latency_request:
            self.latency = None
            self.route_info = False
            self.fail_reason = "latency"
            return False
        else:
            self.latency = latency
            self.route_info = route_info
            return True

    # Modificando a função de alocação para usar a nova lógica de exploração
    def find_best_allocation_for_sfc(
        self, G, service_requirements, server_resources, network_links, services, dst
    ):
        allocation_results = {"dst": {"allocated_server": dst, "path": [], "cost": 0}}
        current_location = dst  # começa a alocação de trás pra frente
        success = True

        for _i, service in enumerate(services):
            # # Verifica se há um próximo serviço na lista
            # if i + 1 < len(services):
            #     next_service = services[i + 1]
            # else:
            #     next_service = None

            best_server, best_path, min_cost, cost_details = self.allocate_sf(
                G,
                service_requirements,
                server_resources,
                network_links,
                current_location,
                service,
                services,
            )

            allocation_results[service] = {
                "allocated_server": best_server,
                "path": best_path,
                "cost": min_cost,
            }

            current_location = best_server

            # Se existe um servidor ótimo
            if best_server:  # Verifica se um servidor foi escolhido
                if cost_details["reuse"]:
                    server_resources[best_server]["cpu_used"] += service_requirements[
                        service
                    ]["CPU"]
                    server_resources[best_server]["cache_used"] += service_requirements[
                        service
                    ]["cache"]
                    server_resources[best_server]["cpu_free"] -= service_requirements[
                        service
                    ]["CPU"]
                    server_resources[best_server]["cache_free"] -= service_requirements[
                        service
                    ]["cache"]
            else:
                # print(f"Falha.")
                success = False
                self.fail_reason = "resource"
                break

        if not success:
            return [], None

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

    def allocate_sf(
        self,
        G,
        service_requirements,
        server_resources,
        network_links,
        current_location,
        service,
        services,
        exploration_margin=1,
    ):
        best_cost, best_candidate, candidates = self.find_candidates_serves_for_sf(
            G, service_requirements, server_resources, current_location, service
        )

        if best_cost == float("inf") or best_candidate == float("inf") or candidates is None:
            return False, False, False, False

        server_choose = best_candidate[0]
        path_to = best_candidate[1]
        min_cost = best_candidate[2]
        cost_details = best_candidate[3]
        return server_choose, path_to, min_cost, cost_details

    def find_candidates_serves_for_sf(
        self, G, service_requirements, server_resources, current_location, service
    ):
        # service_requirements = copy.deepcopy(service_requirements)

        paths = dict(nx.single_source_shortest_path_length(G, current_location, cutoff=8))
        paths[current_location] = 0  # Custo de 'mover' para o mesmo servidor é 0

        candidates = []
        best_cost = float("inf")

        # Função para calcular o custo total
        def calculate_total_cost(node_resource_cost, boot_cost, bandwidth_cost, latency_cost):
            boot_cost = 0
            return (
                (node_resource_cost * self.cpu_factor) / 2
                + (node_resource_cost * self.cache_factor) / 2
                + (boot_cost * self.boot_factor)
                + (bandwidth_cost * self.band_factor)
                + latency_cost
            )

        # Função para verificar disponibilidade de largura de banda e recursos do servidor
        def check_resources(server, path, bandwidth_requirement, cpu_required, cache_required):
            if all(
                G[u][v]["bandwidth"] > bandwidth_requirement
                for u, v in zip(path, path[1:], strict=False)
            ):
                available_cpu = (
                    server_resources[server]["cpu_capacity"]
                    - server_resources[server]["cpu_used"]
                )
                available_cache = (
                    server_resources[server]["cache_capacity"]
                    - server_resources[server]["cache_used"]
                )
                if available_cpu > cpu_required and available_cache > cache_required:
                    return True
            return False

        def calculate_bandwidth_cost(path, bandwidth_requirement):
            cost = 0
            epsilon = 1e-6  # Pequeno valor para evitar divisão por zero

            for u, v in zip(path, path[1:], strict=False):
                available_bandwidth = G[u][v]["bandwidth"]
                if available_bandwidth >= bandwidth_requirement:
                    cost += bandwidth_requirement / (available_bandwidth + epsilon)
                else:
                    return float("inf")  # Link não disponível

            # Normalizar o custo acumulado para ficar entre 0 e 1
            # normalized_cost = cost / (len(path) - 1)
            # normalized_cost = normalized_cost / (bandwidth_requirement / max_bandwidth)

            return cost * 3.0

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

            # if server_resources[server]['cpu_used'] >= 17.27 or reuse:
            #     boot_cost = 0
            # elif (server_resources[server]['cpu_used'] + cpu_required) >= 17.27:
            #     boot_cost = 1
            # else:
            #     boot_cost = 0

            # Ajustar o cálculo de node_resource_cost para evitar divisão por zero
            available_cpu = (
                server_resources[server]["cpu_capacity"] - server_resources[server]["cpu_used"]
            )
            (
                server_resources[server]["cache_capacity"]
                - server_resources[server]["cache_used"]
            )

            if available_cpu > 0:
                node_resource_cost = cpu_required / available_cpu  #
            else:
                node_resource_cost = float(
                    "inf"
                )  # Penalizar fortemente se não houver CPU disponível

            boot_cost = 0
            if check_resources(server, path, bandwidth_requirement, cpu_required, cache_required):
                bandwidth_cost = calculate_bandwidth_cost(path, bandwidth_requirement)
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
                            "reuse": reuse,
                        },
                    )
                )
            else:
                # Sem recurso disponível
                pass

            # Se encontrou candidatos viáveis, não tenta com bitrate menor
            # if best_cost < float('inf'):
            #     break

        best_candidate = min(candidates, key=lambda x: x[2], default=(None, None, None, None))
        return best_cost, best_candidate, candidates

    # latency as restriction
    def calculate_latency_cost(self, distance):
        if distance <= self.latency_request:
            return 0
        else:
            return 1000

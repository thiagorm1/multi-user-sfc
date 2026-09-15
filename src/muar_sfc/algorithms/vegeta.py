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
ch = logging.FileHandler(ROOT_DIR / "logs" / "Vegeta.log")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)




class Vegeta(Algorithm):
    def __init__(self):
        self.name = "vegeta"
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
        self.server_resources = 0
        self.services_requirements = 0
        self.G = 0
        self.services = 0
        self.servers_used = []

        self.cpu_factor = 1
        self.cache_factor = 1
        self.band_factor = 1
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

        # Filtro estilo GA: remove os nós que não podem hospedar VNFs
        self.valid_nodes = [
            node
            for node in self.substrate_network.nodes()
            if self.substrate_network.nodes[node]["type"] != "router"
        ]

        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        self.using_bit_rate = False
        self.bitrate_cut = 1.0
        return self.sfc

    SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")

    def is_shareable(self, service_name: str) -> bool:
        """Verifica se o prefixo da VNF permite compartilhamento."""
        return service_name.startswith(self.SHAREABLE_PREFIXES)

    def _extract_session(self, sfc_id: str) -> str:
        """Extrai o ID da sessão da SFC de forma robusta."""
        parts = sfc_id.split("_")
        if "backup" in sfc_id:
            return parts[3] if len(parts) > 3 else parts[-1]
        return parts[-1]

    def get_fail_reason(self):
        return self.fail_reason

    def set_costs(self, costs_parameters):
        self.cpu_factor = costs_parameters[0]
        self.cache_factor = costs_parameters[1]
        self.band_factor = costs_parameters[2]

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def get_bit_rate_used(self):
        return self.bitrate_cut

    def is_using_bit_rate_cut(self):
        return self.using_bit_rate

    def start_algorithm(self, backup=False, **kwargs):
        substrate_network = self.substrate_network
        sfc = self.sfc

        if backup:
            return bool(self.backup_algorithm(substrate_network, sfc))

        # Removido o shareable_sfs da chamada
        return bool(self.algorithm(substrate_network, sfc))

    def set_nodes_resources(self, substrate_network):
        """
        Lê os recursos e serviços disponíveis diretamente do grafo da rede.
        Usa o dicionário 'services' real como Single Source of Truth.
        """
        net_info = substrate_network
        servers = net_info.nodes()

        server_resources = {}

        for server in servers:
            node_data = net_info.nodes[server]

            cap_cpu = node_data.get("cpu_capacity", 0.0)
            cap_cache = node_data.get("cache_capacity", 0.0)
            used_cpu = node_data.get("cpu_used", 0.0)
            used_cache = node_data.get("cache_used", 0.0)

            # PONTO CHAVE: Captura o dicionário de serviços exato do nó
            active_services = node_data.get("services", {})

            server_resources[server] = {
                "cpu_capacity": round(cap_cpu, 3),
                "cache_capacity": round(cap_cache, 3),
                "cpu_used": round(used_cpu, 3),
                "cache_used": round(used_cache, 3),
                "cpu_free": round(cap_cpu - used_cpu, 3),
                "cache_free": round(cap_cache - used_cache, 3),
                "position": node_data.get("position", (0, 0)),
                "active_services": active_services,  # Substitui o antigo 'reuse'
            }

        return server_resources

    def backup_algorithm(self, substrate_network, sfc):
        self.servers_used = []
        sfc.get_src_vnf()
        sfc.get_dst_vnf()

        sfs_dict = self.sfc.vnfs_dict

        src = sfs_dict[0]["location"]
        dst = sfs_dict[-1]["location"]

        nodes_resource = self.set_nodes_resources(substrate_network)
        network_links = copy.deepcopy(substrate_network._adj)

        G = self.create_network_graph(network_links)
        # k = 10
        # paths = list(nx.shortest_simple_paths(G, source=src, target=dst, weight='weight'))
        # caminhos = paths[:k]
        route_info = []

        # is_success = False

        services, service_requirements = self.prepare_service_requirements(sfs_dict)
        allocation_results = {"dst": {"allocated_server": dst, "path": [], "cost": 0}}

        backup_sf = service_requirements[services[1]]
        cpu_required = backup_sf["CPU"]
        cache_required = backup_sf["cache"]
        # bw_in = 0 ##
        # bw_out= 0 ##

        location_result = False
        node_choose = 0
        for node in [src, dst]:
            cpu_free = nodes_resource[node]["cpu_free"]
            cache_free = nodes_resource[node]["cpu_free"]
            cpu_capacity = nodes_resource[node]["cpu_capacity"]
            if cpu_required <= cpu_free and cache_required <= cache_free and cpu_capacity > 0:
                node_choose = node
                location_result = True
            break
        if location_result:
            my_paths = []
            if node_choose == src:
                my_paths.append([node_choose])
            else:
                my_paths.append(
                    nx.shortest_path(G, source=src, target=node_choose, weight="weight")
                )

            if node_choose == dst:
                my_paths.append([node_choose])
            else:
                my_paths.append(
                    nx.shortest_path(G, source=node_choose, target=dst, weight="weight")
                )

            allocation_results[services[0]] = {"allocated_server": dst, "path": [dst], "cost": 0}
            allocation_results[services[1]] = {
                "allocated_server": node_choose,
                "path": my_paths[1],
                "cost": 0,
            }
            allocation_results[services[2]] = {
                "allocated_server": src,
                "path": my_paths[0],
                "cost": 0,
            }

            # print(allocation_results)
        else:
            self.fail_reason = "resource"
            return False

        # ultima iteração para o src
        path_to_src = nx.dijkstra_path(G, 0, src, weight="weight")

        route_info = {key: list(value["path"]) for key, value in allocation_results.items()}

        # calculo da latencia antes do src
        total_latency = sum(len(path) - 1 for path in route_info.values() if path)

        route_info["src"] = list(path_to_src)

        # colocando o dst no final
        first_key, first_value = next(iter(route_info.items()))
        del route_info[first_key]
        route_info[first_key] = first_value
        return self.evaluate_result(total_latency, route_info)

    def allocate_sf(
        self,
        G,
        service_requirements,
        server_resources,
        network_links,
        current_location,
        service,
        services,
        restrictions=None,
        solution=None,
        current_session_id=None,
    ):
        if solution is None:
            solution = []
        if restrictions is None:
            restrictions = []
        best_cost, best_candidate, candidates = self.find_candidates_serves_for_sf(
            G,
            service_requirements,
            server_resources,
            current_location,
            service,
            current_session_id,
            restrictions,
            solution,
        )

        if best_cost == float("inf") or best_candidate == float("inf") or candidates is None:
            return False, False, False, False

        server_choose = best_candidate[0]

        if restrictions == []:
            self.servers_used.append(server_choose)
            path_to = best_candidate[1]
            min_cost = best_candidate[2]
            cost_details = best_candidate[3]
        else:
            if service.startswith("dst"):
                server_choose = current_location
                path_to = [current_location]
                min_cost = 0
                cost_details = best_candidate[3]
            else:
                path_to = best_candidate[1]
                min_cost = best_candidate[2]
                cost_details = best_candidate[3]
        return server_choose, path_to, min_cost, cost_details

    def find_candidates_serves_for_sf(
        self,
        G,
        service_requirements,
        server_resources,
        current_location,
        service,
        current_session_id,
        restriction=None,
        solutions=None,
    ):

        if solutions is None:
            solutions = []
        if restriction is None:
            restriction = []
        paths = dict(nx.single_source_shortest_path_length(G, current_location, cutoff=8))
        paths[current_location] = 0  # Custo de 'mover' para o mesmo servidor é 0

        # =========================================================
        # LÓGICA DO GENETIC ALG: Filtragem de nós disponíveis
        # =========================================================
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        allow_md = getattr(self, "allow_md_host", False)

        if allow_md:
            available_nodes = list(self.valid_nodes)
        else:
            available_nodes = [node for node in self.valid_nodes if node != dst]

        # Filtra o dicionário mantendo apenas os nós selecionáveis
        paths = {node: hops for node, hops in paths.items() if node in available_nodes}
        # =========================================================

        candidates = []
        best_cost = float("inf")

        # Função para calcular o custo total
        def calculate_total_cost(cpu_cost, cache_cost, boot_cost, bandwidth_cost, latency_cost):
            boot_cost = 0
            return (
                (cpu_cost * self.cpu_factor)
                + (cache_cost * self.cache_factor)
                + (boot_cost * self.boot_factor)
                + (bandwidth_cost * self.band_factor)
                + latency_cost
            )

        # Função para verificar disponibilidade de largura de banda e recursos do servidor
        def check_resources(
            server, path, bandwidth_requirement, cpu_required, cache_required, restriction
        ):
            if all(G[u][v]["bandwidth"] > bandwidth_requirement for u, v in
                   zip(path, path[1:], strict=False)):
                available_cpu = (
                    server_resources[server]["cpu_capacity"] - server_resources[server]["cpu_used"]
                )
                available_cache = (
                    server_resources[server]["cache_capacity"]
                    - server_resources[server]["cache_used"]
                )
                if restriction == []:
                    if available_cpu > cpu_required and available_cache > cache_required:
                        return True
                else:
                    if available_cpu >= cpu_required and available_cache >= cache_required:
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
                    return float("inf")

            cost = cost * 3.5
            return cost

        bandwidth_requirement = service_requirements[service]["out_bw"]

        for server, _num_hops in paths.items():
            path = nx.shortest_path(G, current_location, server, weight="weight")

            # =========================================================
            # LÓGICA DE REUSO ALINHADA COM O NET2
            # =========================================================
            clean_service_id = service.replace("_b", "")
            reuse = False

            # 1. Verifica se a VNF faz parte dos serviços compartilháveis
            if self.is_shareable(service) or self.is_shareable(clean_service_id):
                # 2. Cria a tupla exata que o Net2 usa como chave
                service_key = (clean_service_id, current_session_id)

                # 3. Verifica se a chave primária está ativa neste servidor
                if service_key in server_resources[server]["active_services"]:
                    reuse = True
            # =========================================================

            node_resource_cost = 0 if reuse else 1

            boot_cost = 0
            cpu_required = 0 if reuse else service_requirements[service]["CPU"]
            cache_required = 0 if reuse else service_requirements[service]["cache"]

            available_cpu = server_resources[server]["cpu_free"]
            if available_cpu > 0:
                if cpu_required == 0:
                    cpu_required = service_requirements[service]["CPU"] * 0.3
                node_resource_cost = cpu_required / available_cpu
            else:
                node_resource_cost = float("inf")

            if check_resources(
                server, path, bandwidth_requirement, cpu_required, cache_required, restriction
            ):
                bandwidth_cost = calculate_bandwidth_cost(path, bandwidth_requirement) * 1000
                cpu_cost = node_resource_cost * 1000
                cache_cost = cpu_cost / 2
                latency_cost = 0

                total_cost = calculate_total_cost(
                    cpu_cost, cache_cost, boot_cost, bandwidth_cost, latency_cost
                )

                if total_cost < best_cost:
                    best_cost = total_cost

                candidates.append(
                    (
                        server,
                        path,
                        round(total_cost, 4),
                        {
                            "cpu_cost": round(cpu_cost, 4),
                            "cache_cost": round(cache_cost, 4),
                            "boot_cost": boot_cost,
                            "bandwidth_cost": round(bandwidth_cost, 4),
                            "latency_cost": latency_cost,
                            "total_cost": round(total_cost, 4),
                            "reuse": reuse,
                        },
                    )
                )

        best_candidate = min(candidates, key=lambda x: x[2], default=(None, None, None, None))
        return best_cost, best_candidate, candidates

    # latency as restriction
    def calculate_latency_cost(self, distance):
        if distance <= self.latency_request:
            return 0
        else:
            return 1000

    def algorithm(self, substrate_network, sfc):
        self.servers_used = []
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()
        sfc.get_substrate_node(src_vnf)
        dst = sfc.get_substrate_node(dst_vnf)

        nodes_resource = self.set_nodes_resources(substrate_network)
        network_links = copy.deepcopy(substrate_network._adj)

        G = self.create_network_graph(network_links)

        sfs_dict = self.sfc.vnfs_dict
        services, service_requirements = self.prepare_service_requirements(sfs_dict)

        # Extrai a sessão da SFC atual
        current_session_id = self._extract_session(sfc.id)

        is_success = False

        # Repassa current_session_id
        route_info, latency = self.find_best_allocation_for_sfc(
            G,
            service_requirements,
            nodes_resource,
            network_links,
            services,
            dst,
            current_session_id,
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
        """
        Constrói o grafo auxiliar de roteamento baseado na topologia real.
        Garante a extração segura de atributos dinâmicos (EAFP/Defensive Programming).
        """
        G = nx.Graph()
        for node, edges in network_topology.items():
            for target, edge_attr in edges.items():
                # Extração segura usando .get() com fallback para 0.0
                bw_capacity = edge_attr.get("bandwidth_capacity", 0.0)
                bw_used = edge_attr.get("bandwidth_used", 0.0)

                # Cálculo da banda livre em tempo de execução
                bw_free = max(0.0, bw_capacity - bw_used)

                G.add_edge(node, target, bandwidth=bw_free, weight=1)
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
        service_requirements["dst"] = {"CPU": 0, "cache": 0, "out_bw": 0, "in_bw": 0, "latency": 0}
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
        self,
        G,
        service_requirements,
        server_resources,
        network_links,
        services,
        dst,
        current_session_id,
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
                current_session_id=current_session_id,
            )

            allocation_results[service] = {
                "allocated_server": best_server,
                "path": best_path,
                "cost": min_cost,
            }

            current_location = best_server

            # Se existe um servidor ótimo
            if best_server:
                # O desconto de recursos só deve ocorrer se NÃO for um reuso (nova instância)
                if not cost_details["reuse"]:
                    server_resources[best_server]["cpu_used"] += service_requirements[service][
                        "CPU"
                    ]
                    server_resources[best_server]["cache_used"] += service_requirements[service][
                        "cache"
                    ]

                    server_resources[best_server]["cpu_free"] -= service_requirements[service][
                        "CPU"
                    ]
                    server_resources[best_server]["cache_free"] -= service_requirements[service][
                        "cache"
                    ]
            else:
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

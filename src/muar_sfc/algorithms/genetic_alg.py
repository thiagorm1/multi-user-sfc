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
import random
import time
from pathlib import Path

import networkx as nx
from deap import algorithms, base, creator, tools

from muar_sfc.algorithms.algorithm import Algorithm
from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
    get_available_shortest_path,
    pre_get_single_source_minimum_latency_path,
)
from muar_sfc.config import ROOT_DIR
from muar_sfc.core.sfc import SFC

SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")

# Configuração de Observabilidade Estruturada
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Modernização Orientada a Objetos Multiplataforma (pathlib)
log_path = Path(ROOT_DIR) / "logs" / "MSF.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

ch = logging.FileHandler(log_path)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class Genetic(Algorithm):
    def __init__(self):
        self.name = "ga"
        self.graph = None
        self.sfc = None
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None

        self.can_alocate_sf_in_node = {}
        self.service_requirements = {}
        self.services = []
        self.crashed_servers = []
        self.shareable_sfs = None
        self.server_resources = None
        self.node_info = {}
        self.latency_request = 0
        self.min_latency = 0
        self.elapsed_time = None
        self.latency_minus_dst = 0
        self.valid_nodes = []
        self.G = 0

        self.cpu_weight = 1
        self.cache_weight = 1
        self.band_weight = 2
        self.latency_weight = 0.1
        self.boot_weight = 1

        # Inicializar os atributos para medir o tempo
        self.evaluation_time = 0
        self.crossover_time = 0
        self.mutation_time = 0

    def clear_all(self):
        # logger.debug('clear all')
        self.graph = None
        self.sfc = None
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.node_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.valid_nodes = []
        self.service_requirements = {}
        self.services = []

        # Inicializar os atributos para medir o tempo
        self.evaluation_time = 0
        self.crossover_time = 0
        self.mutation_time = 0

    def install_substrate_network(self, graph, shareable_sfs=None):
        if shareable_sfs is None:
            shareable_sfs = []
        self.graph = graph
        self.valid_nodes = [
            node for node in self.graph.nodes() if self.graph.nodes[node]["type"] != "router"
        ]
        self.single_source_minimum_latency_path = pre_get_single_source_minimum_latency_path(
            self.graph
        )
        return self.graph

    def install_SFC(self, sfc: SFC):
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        is_backup = sfc.id.split("_")[2] == "backup"

        self.latency_request = sfc.get_latency_request()
        self.min_latency = 0

        service_requirements = {}
        services = []  # Lista para guardar os nomes
        sfs_dict = sfc.vnfs_dict

        for item in sfs_dict:
            nome = item["name"]
            services.append(nome)  # Adiciona o nome à lista de nomes
            service_requirements[nome] = {
                "CPU": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        if not is_backup:
            services.append("dst")
            service_requirements["dst"] = {"CPU": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

        self.service_requirements = service_requirements
        self.services = services

        return self.sfc

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def start_algorithm(self):  # ,is_backup):
        self.algorithm()
        is_success = self.check_solution()
        if is_success:
            try:
                logger.info("Finished algorithm, success")
                return True
            except Exception as e:
                # Proteção: captura estritamente erros da aplicação
                logger.exception(f"Erro ao processar o sucesso do algoritmo: {e}")
                self.handle_failure()
                return False
        else:
            self.handle_failure()
            logger.info("End algorithm, failed")
            return False

    def algorithm(self):
        # Get src and dst vnf
        src_vnf = self.sfc.get_src_vnf()
        dst_vnf = self.sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        self.sfc.get_substrate_node(src_vnf)
        dst = self.sfc.get_substrate_node(dst_vnf)

        service_requirements = self.service_requirements
        services = self.services

        # Criação do grafo representando a rede com capacidade de banda
        sup_graph = copy.deepcopy(self.graph)

        all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(sup_graph, weight="weight"))
        a = time.time()
        route_info, latency = self.genetic_alg(
            sup_graph, service_requirements, services, dst, all_pairs_shortest_path
        )
        b = time.time()
        self.elapsed_time = b - a
        if latency > self.latency_request or not route_info:
            self.latency = None
            self.route_info = False
            return False
        else:
            self.latency = latency
            self.route_info = route_info
            return True

    # Início da função fit_path modificado
    def genetic_alg(self, sup_graph, service_requirements, services, dst, all_pairs_shortest_path):
        a = time.time()
        service_requirements_local = service_requirements

        # Verifica se a classe já existe e, em caso afirmativo, exclui-a
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Recupera a configuração (com fallback para False por segurança)
        allow_md = getattr(self, "allow_md_host", False)

        # --- ALTERAÇÃO 1: FILTRAGEM CONDICIONAL ---
        if allow_md:
            # Mantém todos os nós válidos (o dst já entra aqui porque seu tipo não é 'router')
            available_nodes = list(self.valid_nodes)
        else:
            # Filtra o dst para impedir que ele seja sorteado
            available_nodes = [node for node in self.valid_nodes if node != dst]

        # Verificação de segurança: se não houver nós suficientes (excluindo o mobile) para alocar
        if len(available_nodes) < 4:
            # Retorna falha ou ajusta o tamanho da amostra conforme necessário
            logger.error("Não há nós suficientes para alocar a SFC excluindo o nó mobile.")
            return False, float("inf")
        # -------------------------------------

        # DEAP setup para minimizar o fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        # --- ALTERAÇÃO 2: USO DA LISTA FILTRADA ---
        # Usamos 'available_nodes' em vez de 'self.valid_nodes'
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            lambda: random.sample(available_nodes, 4),
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            # --- ALTERAÇÃO 3: PROTEÇÃO DEFENSIVA ---
            if not allow_md and dst in individual:
                return (float("inf"),)
            # ---------------------------------------

            individual_w_dst = individual + [dst]
            node_cost = 0
            band_cost = 0
            latency_cost = 0
            total_latency = 0

            for service_index, server_id in enumerate(individual_w_dst):
                if service_index >= len(services):
                    break

                service = services[service_index]
                vnf = self.sfc.get_vnf_by_id(service)
                cpu_request = self.service_requirements[service]["CPU"]
                cache_request = self.service_requirements[service]["cache"]
                bw_required = self.service_requirements[service]["out_bw"]

                cpu_capacity = self.graph.nodes[server_id]["cpu_capacity"]
                cache_capacity = self.graph.nodes[server_id]["cache_capacity"]

                # 🔴 NOVA PROTEÇÃO: capacidade zero => custo infinito
                if cpu_capacity == 0 or cache_capacity == 0:
                    return (float("inf"),)

                cpu_available = cpu_capacity - self.graph.nodes[server_id]["cpu_used"]
                cache_available = cache_capacity - self.graph.nodes[server_id]["cache_used"]

                self.graph.nodes[server_id]["reuse"]
                LAMBDA = 0.0001

                if service != "dst":
                    if cpu_request <= cpu_available and cache_request <= cache_available:
                        node_resource_cost = (
                            self.graph.nodes[server_id]["cpu_used"] + cpu_request + LAMBDA
                        ) / cpu_capacity + (
                            self.graph.nodes[server_id]["cache_used"] + cache_request + LAMBDA
                        ) / cache_capacity
                    else:
                        return (float("inf"),)
                else:
                    # É apenas o tráfego final chegando ao usuário (não consome CPU/Cache extra)
                    node_resource_cost = 0.2

                comp_latency = calculate_computational_latency(self.graph, server_id, vnf)

                node_cost += (self.cpu_weight * node_resource_cost) + (
                    self.cache_weight * node_resource_cost
                )

                edge_latency = 0
                link_band_cost = 0

                if service_index < len(individual_w_dst) - 1:
                    next_server_id = individual_w_dst[service_index + 1]
                    path = get_available_shortest_path(
                        self.graph,
                        source=server_id,
                        target=next_server_id,
                        bandwidth_required=bw_required,
                    )

                    if path == []:
                        return (float("inf"),)

                    if len(path) > 1:
                        for u, v in zip(path[:-1], path[1:], strict=False):
                            edge_latency += calculate_latency_betwen_nodes(self.graph, u, v, vnf)

                            bw_used = self.graph[u][v].get("bandwidth_used")
                            bw_capacity = self.graph[u][v].get("bandwidth_capacity")

                            if bw_capacity == 0:
                                return (float("inf"),)

                            bw_cost = (bw_used + bw_required) / bw_capacity
                            link_band_cost += bw_cost

                total_latency = comp_latency + edge_latency

                band_cost += self.band_weight * link_band_cost
                latency_cost += self.latency_weight * total_latency

            total_cost = node_cost + band_cost + latency_cost
            return (total_cost,)

        def custom_mutation(individual):
            start_time = time.time()
            tools.mutShuffleIndexes(individual, indpb=0.05)
            end_time = time.time()
            self.mutation_time += (end_time - start_time) * 1000  # Tempo em milissegundos
            return (individual,)

        def custom_crossover(parent1, parent2):
            """
            Realiza o crossover entre dois pais, garantindo que os filhos não tenham
            elementos repetidos.
            Sobrescreve os pais diretamente.
            """
            start_time = time.time()
            size = len(parent1)
            # Escolhe dois pontos de corte
            cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))

            # Cria os filhos com base nos segmentos dos pais
            child1 = [None] * size
            child2 = [None] * size

            # Copia o segmento do pai 1 para o filho 1
            child1[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
            # Copia o segmento do pai 2 para o filho 2
            child2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2]

            # Preenche os filhos com os elementos restantes dos outros pais
            fill_child(child1, parent2, cxpoint2)
            fill_child(child2, parent1, cxpoint2)

            # Sobrescreve os pais diretamente com os novos filhos
            parent1[:] = child1
            parent2[:] = child2
            end_time = time.time()
            self.crossover_time += (end_time - start_time) * 1000  # Tempo em milissegundos
            return parent1, parent2

        def fill_child(child, parent, start):
            """
            Preenche os elementos faltantes no filho, garantindo que não haja repetição.
            """
            size = len(parent)
            current_pos = start
            for gene in parent:
                if gene not in child:
                    child[current_pos] = gene
                    current_pos = (current_pos + 1) % size

        toolbox.register("mate", custom_crossover)
        toolbox.register("mutate", custom_mutation)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)
        # Registrar o método de paralelização

        # Parâmetros do algoritmo genético
        population_size = 15
        crossover_probability = 0.7
        mutation_probability = 0.2
        number_of_generations = 25

        # Inicialização da população
        pop = toolbox.population(n=population_size)

        # Algoritmo genético
        algorithms.eaSimple(
            pop,
            toolbox,
            crossover_probability,
            mutation_probability,
            ngen=number_of_generations,
            verbose=False,
        )

        def display_paths_and_create_service_dict(best_individual):
            service_to_server_dict = {}
            service_names = list(
                service_requirements_local.keys()
            )  # Assumindo que você tem os nomes dos serviços

            for i, server_id in enumerate(best_individual):
                service_name = service_names[i]
                if i < len(best_individual) - 1:
                    next_server_id = best_individual[i + 1]
                    if server_id == next_server_id:
                        service_to_server_dict[service_name] = [server_id]
                    else:
                        path = nx.shortest_path(
                            sup_graph, source=server_id, target=next_server_id, weight="weight"
                        )
                        service_to_server_dict[service_name] = path
                else:
                    service_to_server_dict[service_name] = [server_id]
            return service_to_server_dict

        # Após a execução do algoritmo genético
        best_ind = tools.selBest(pop, 1)[0]

        if best_ind.fitness.values[0] == float("inf"):
            return False, 100
        else:
            best_ind.append(dst)
            service_to_server_dict = display_paths_and_create_service_dict(best_ind)

            service_to_server_dict.popitem()

            # Inverter a ordem dos itens no dicionário
            route_info = dict(reversed(list(service_to_server_dict.items())))

            src_node = next(reversed(route_info.values()))[0]

            path_to_src = list(reversed(nx.dijkstra_path(sup_graph, src_node, 0, weight="weight")))

            total_latency = sum(len(path) - 1 for path in route_info.values() if path)

            route_info["src"] = path_to_src
            route_info["dst"] = []
            b = time.time()
            elapsed_time_ms = (b - a) * 1000  # Convertendo para milissegundos

            logger.info(f"Tempo total de avaliação: {self.evaluation_time:.2f} ms")
            logger.info(f"Tempo total de crossover: {self.crossover_time:.2f} ms")
            logger.info(f"Tempo total de mutação: {self.mutation_time:.2f} ms")
            logger.info(f"Tempo de execução: {elapsed_time_ms:.2f} ms")

            return route_info, total_latency

    def handle_failure(self):
        self.route_info = False
        self.latency = None

    def check_solution(self):
        if (
            not isinstance(self.latency, (int, float))
            or self.latency < 0
            or self.latency > self.sfc.get_latency_request()
            or not self.route_info
        ):
            return False
        if len(list(self.route_info.keys())) != 6:
            return False

        prev_path_end = None
        prev_sf = None  # Correção: movido para fora do loop (F821 Undefined name)

        for sf, path in self.route_info.items():
            if sf == "dst":
                continue
            if prev_path_end is not None and path[-1] != prev_path_end:
                logger.warning(
                    f"Inconsistência entre {prev_sf} e {sf}: {prev_path_end} != {path[0]}"
                )
                return False  # ou raise Exception se quiser abortar
            prev_path_end = path[0]
            prev_sf = sf

        return True

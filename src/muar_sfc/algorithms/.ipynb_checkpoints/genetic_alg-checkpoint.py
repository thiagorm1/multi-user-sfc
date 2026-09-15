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
import time 
import pandas as pd
from config import ROOT_PATH
import random
import deap
from deap import base, creator, tools, algorithms
import networkx as nx
import numpy
from multiprocessing import Pool
from muar_sfc.algorithms.networkUtils import get_link_bandwidth_free,pre_get_single_source_minimum_latency_path, get_link_latency,get_shortest_path, get_available_shortest_path
from muar_sfc.algorithms.networkUtils import calculate_computational_latency,calculate_latency_betwen_nodes
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')


# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler(ROOT_PATH + './logs/MSF.log')
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

from muar_sfc.algorithms.algorithm import Algorithm


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
        self.latency_minus_dst =  0
        self.valid_nodes =[]
        self.G = 0 
        self.services = 0

        self.cpu_weight   =  1
        self.cache_weight =  1
        self.band_weight  =  2
        self.latency_weight  =  0.1
        self.boot_weight  =  1

        # Inicializar os atributos para medir o tempo
        self.evaluation_time = 0
        self.crossover_time = 0
        self.mutation_time = 0

    def clear_all(self):
        #logger.debug('clear all')
        self.graph = None
        self.sfc = None
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.node_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.valid_nodes =[]
        self.service_requirements = {}
        self.services = []
        
        # Inicializar os atributos para medir o tempo
        self.evaluation_time = 0
        self.crossover_time = 0
        self.mutation_time = 0

    def install_substrate_network(self,graph, shareable_sfs=[]):
        self.graph = graph
        self.valid_nodes = [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] != 'router']
        self.single_source_minimum_latency_path = pre_get_single_source_minimum_latency_path(self.graph)
        return self.graph
    
    def install_SFC(self, sfc):
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        is_backup = True if sfc.id.split("_")[2]=='backup' else False

        self.latency_request = sfc.get_latency_request()
        self.min_latency  = 0 

        service_requirements = {} 
        services = []  # Lista para guardar os nomes
        sfs_dict = sfc.vnfs_dict
        
        for item in sfs_dict:
            nome = item['name']
            services.append(nome)  # Adiciona o nome à lista de nomes
            service_requirements[nome] = {
                'CPU': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw']}

        if not is_backup:
            services.append('dst')
            service_requirements['dst'] = {'CPU': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0}  
        
        self.service_requirements = service_requirements
        self.services = services
        #################################################################################################################################################
        #server_resources = self.graph._node
        #self.edge_computing_servers = list(set(self.edge_computing_servers) - set(crashed_servers))

        # shareable_sfs = {node_id: [] for node_id in server_resources.keys()}
        # # Adicionando o campo 'reuse' no node_table
        # for node_id, node_info in server_resources.items():
        #     #if node_id in crashed_servers:
        #     #    continue
        #     node_info['reuse'] = []
        #     if node_id in shareable_sfs:
        #         for vnf in shareable_sfs[node_id]:
        #             node_info['reuse'].append(vnf.id)  # Acessando o atributo 'id' da VNF

        # self.shareable_sfs = shareable_sfs      
        # self.server_resources = server_resources  

        # for node in self.graph.nodes():
        #     self.node_info[node] = {}
        #     cpu_capacity = server_resources[node]['cpu_capacity']
        #     cache_capacity = server_resources[node]['cache_capacity']

        #     cpu_used = server_resources[node]['cpu_used']
        #     cache_used = server_resources[node]['cache_used']

        #     cpu_free = cpu_capacity - cpu_used
        #     cache_free = cache_capacity - cache_used

        #     reuse_list = server_resources[node]['reuse']  # Obter lista de VNFs reutilizáveis para o nó atual

        #     for vnf_id, vnf in list(self.sfc.vnfs.items()):
        #         if cpu_capacity == 0:
        #             continue
        #         is_reusable = vnf_id in reuse_list
        #         cpu_request = 0 if is_reusable else vnf.get_cpu_request() 
        #         cache_request = 0 if is_reusable else vnf.get_cache_request()
                
        #         # Verificar se os recursos estão disponíveis e se o VNF pode ser reutilizado
        #         resources_sufficient = (cpu_free >= cpu_request) and (cache_free >= cache_request)
                
        #         # Adicionar a informação de reuso na estrutura self.node_info
        #         # if node in crashed_servers:
        #         #     resources_sufficient = False
        #         #     is_reusable = False
        #         self.node_info[node][vnf_id] = {
        #             'resources_sufficient': resources_sufficient,
        #             'is_reusable': is_reusable}
    
        # # Construir o dicionário de alocação de SFs
        # sf_allocation = {vnf_id: [] for vnf_id in self.sfc.vnfs.keys()}
        # for node, vnfs in self.node_info.items():
        #     for vnf_id, info in vnfs.items():
        #         if info['resources_sufficient']:
        #             sf_allocation[vnf_id].append(node)
        # self.can_alocate_sf_in_node = sf_allocation

        return self.sfc

    def get_latency(self):
        return self.latency
    
    def get_route_info(self):
        return self.route_info

    def start_algorithm(self):#,is_backup):
        self.algorithm()
        is_success = self.check_solution()
        if is_success:
            try:
                logger.info("Finished algorithm, success")
                return True  
            except:
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
        src = self.sfc.get_substrate_node(src_vnf)
        dst = self.sfc.get_substrate_node(dst_vnf)
        
        service_requirements = self.service_requirements
        services = self.services

        # Criação do grafo representando a rede com capacidade de banda
        sup_graph = copy.deepcopy(self.graph)

        all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(sup_graph, weight='weight'))
        a = time.time()
        route_info, latency = self.genetic_alg(sup_graph,service_requirements,services,dst,all_pairs_shortest_path)
        b = time.time()
        self.elapsed_time = (b-a)
        if latency > self.latency_request or route_info == False:
            self.latency = None
            self.route_info = False
            return False
        else:
            self.latency = latency
            self.route_info = route_info
            return True

    # Início da função fit_path modificado
    def genetic_alg(self,sup_graph, service_requirements, services, dst, all_pairs_shortest_path):
        a = time.time()
        service_requirements_local = service_requirements
        
        # Verifica se a classe já existe e, em caso afirmativo, exclui-a
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        #self.valid_nodes.remove(dst) # TODO pode deletar essa condicional depois
        if dst in self.valid_nodes: 
            self.valid_nodes.remove(dst)
        
        # DEAP setup para minimizar o fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, lambda: random.sample(self.valid_nodes, 4))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
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
                cpu_request = self.service_requirements[service]['CPU']
                cache_request = self.service_requirements[service]['cache']
                bw_required = self.service_requirements[service]['out_bw']

                cpu_available = self.graph.nodes[server_id]['cpu_capacity'] - self.graph.nodes[server_id]['cpu_used']
                cache_available = self.graph.nodes[server_id]['cache_capacity'] - self.graph.nodes[server_id]['cache_used']
                    
                #TODO can reuse sf functionality missing
                reuse_sfs = self.graph.nodes[server_id]['reuse']
                LAMBDA = 0.0001
                if server_id != dst:
                    if cpu_request < cpu_available and cache_request < cache_available:
                        reuse = service in reuse_sfs
                        node_resource_cost = (self.graph.nodes[server_id]['cpu_used'] + cpu_request + LAMBDA)/ self.graph.nodes[server_id]['cpu_capacity'] + \
                                            (self.graph.nodes[server_id]['cache_used'] + cache_request + LAMBDA)/ self.graph.nodes[server_id]['cpu_capacity']
                    else:
                        return float('inf'),
                else:
                    reuse = True
                    node_resource_cost = 0.2
                comp_latency = calculate_computational_latency(self.graph,server_id,vnf)
                node_cost += (self.cpu_weight * node_resource_cost) + (self.cache_weight * node_resource_cost)
                
                edge_latency = 0
                if service_index < len(individual_w_dst) - 1:
                    next_server_id = individual_w_dst[service_index + 1]
                    path = get_available_shortest_path(self.graph,source=server_id,target=next_server_id,bandwidth_required=bw_required)
                    if path == []:
                        return float('inf'),
                    link_band_cost = 0
                    if len(path) > 1:
                        for u, v in zip(path[:-1], path[1:]):
                            edge_latency += calculate_latency_betwen_nodes(self.graph,u,v,vnf)
                            bw_used = self.graph[u][v].get('bandwidth_used')
                            bw_capacity = self.graph[u][v].get('bandwidth_capacity')  # evitar divisão por zero
                            bw_cost = (bw_used + bw_required) / bw_capacity  # uso percentual após alocação
                            link_band_cost += bw_cost

                total_latency =  comp_latency       +  edge_latency

                band_cost     += self.band_weight      *  link_band_cost
                latency_cost  += self.latency_weight *  total_latency
                
            total_cost =  node_cost + band_cost + latency_cost
            return total_cost,

        def custom_mutation(individual):
            start_time = time.time()
            tools.mutShuffleIndexes(individual, indpb=0.05)
            end_time = time.time()
            self.mutation_time += (end_time - start_time) * 1000  # Tempo em milissegundos
            return individual,

        def custom_crossover(parent1, parent2):
            """
            Realiza o crossover entre dois pais, garantindo que os filhos não tenham elementos repetidos.
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

        toolbox.register("mate",custom_crossover)
        toolbox.register("mutate",custom_mutation)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)
        # Registrar o método de paralelização
    
        # Parâmetros do algoritmo genético
        population_size = 30
        crossover_probability = 0.7
        mutation_probability = 0.2
        number_of_generations = 50

        # Inicialização da população
        pop = toolbox.population(n=population_size)

        # Algoritmo genético
        algorithms.eaSimple(pop, toolbox, crossover_probability, mutation_probability, ngen=number_of_generations,verbose=False)

        def display_paths_and_create_service_dict(best_individual):
            paths = []  # Lista para armazenar os caminhos ótimos
            service_to_server_dict = {}
            service_names = list(service_requirements_local.keys())  # Assumindo que você tem os nomes dos serviços
            
            for i, server_id in enumerate(best_individual):
                service_name = service_names[i]
                if i < len(best_individual) - 1:
                    next_server_id = best_individual[i + 1]
                    if server_id == next_server_id:
                        service_to_server_dict[service_name] = [server_id]
                    else:
                        path = nx.shortest_path(sup_graph, source=server_id, target=next_server_id, weight='weight')
                        service_to_server_dict[service_name] = path
                else:
                    service_to_server_dict[service_name] = [server_id]
            return service_to_server_dict

        # Após a execução do algoritmo genético
        best_ind = tools.selBest(pop, 1)[0]
        
        if best_ind.fitness.values[0] == float('inf'):
            return False, 100
        else:
            best_ind.append(dst)     
            service_to_server_dict = display_paths_and_create_service_dict(best_ind)

            service_to_server_dict.popitem()

            # Inverter a ordem dos itens no dicionário
            route_info = dict(reversed(list(service_to_server_dict.items())))

            src_node = next(reversed(route_info.values()))[0]

            path_to_src = list(reversed(nx.dijkstra_path(sup_graph, src_node, 0, weight='weight')))
        
            total_latency = sum(len(path) - 1 for path in route_info.values() if path) 

            route_info['src'] = path_to_src
            route_info['dst'] = []
            b = time.time()
            elapsed_time_ms = (b - a) * 1000  # Convertendo para milissegundos

            print(f"Tempo total de avaliação: {self.evaluation_time:.2f} ms")
            print(f"Tempo total de crossover: {self.crossover_time:.2f} ms")
            print(f"Tempo total de mutação: {self.mutation_time:.2f} ms")

            print(f"Tempo de execução: {elapsed_time_ms:.2f} ms")
            
            #[2, 5, 6, 8, 9, 14, 18, 23, 25, 28, 33, 34]
            return route_info, total_latency
        
    def handle_failure(self):
        self.route_info = False
        self.latency = None

    def check_solution(self):
        if not isinstance(self.latency, (int, float)) or self.latency < 0 or self.latency > self.sfc.get_latency_request() or not self.route_info:
            return False
        if len(list(self.route_info.keys()))!=6:
            return False
        
        prev_path_end = None
        for sf, path in self.route_info.items():
            prev_sf = None
            if sf == "dst":
                continue
            if prev_path_end is not None:
                if path[-1] != prev_path_end:
                    print(f"Inconsistência entre {prev_sf} e {sf}: {prev_path_end} != {path[0]}")
                    return False  # ou raise Exception se quiser abortar
            prev_path_end = path[0]
            prev_sf = sf
        return True














































    # def cut_topology(self,G_complex, complex_network_topology, server_resources_complex, node_reference, hops_cuff):
    #     self.edge_computing_servers.append(node_reference)
    #     edge_computing_servers = set(self.edge_computing_servers)
    #     sub_graph_complex = G_complex.subgraph(edge_computing_servers)

    #     #nodes_within_hops_complex = nx.single_source_shortest_path_length(G_complex, node_reference, cutoff=hops_cuff)
    #     #sub_graph_complex = G_complex.subgraph(nodes_within_hops_complex.keys())
    #     server_resources = {node: server_resources_complex[node] for node in sub_graph_complex.nodes if node != 0}

    #     new_network_topology = {}
    #     for node in sub_graph_complex.nodes:
    #         if node ==  0:  # Pula o nó '0' se presente
    #             continue
    #         new_network_topology[node] = {}
    #         for neighbor in sub_graph_complex.neighbors(node):
    #             if neighbor == 0:  # Pula o vizinho '0' se presente
    #                 continue
    #             if neighbor in sub_graph_complex.nodes:
    #                 new_network_topology[node][neighbor] = complex_network_topology[node][neighbor]

    #     G_non_complex = nx.Graph()
        
    #     for node, edges in new_network_topology.items():
    #         for target, edge_attr in edges.items():
    #             bandwidth_free = edge_attr['bandwidth_capacity'] - edge_attr['bandwidth_used']
    #             G_non_complex.add_edge(node, target, bandwidth=bandwidth_free, weight=1)

    #     # Removendo a chave '0' após a criação para garantir que não esteja presente
    #     if 0 in G_non_complex:
    #         G_non_complex.remove_node(0)
    #     del server_resources[node_reference]
    #     return G_non_complex, server_resources









            
    # def backup_algorithm(self,substrate_network, sfc, shareable_sfs=None):
    #     vnf_info = sfc.vnfs_dict
    #     vnf_id = vnf_info[1]['name']

    #     # Get substrate network nodes that src and dst are assigned in advanced
    #     src = sfc.get_substrate_node(sfc.get_src_vnf())
    #     dst = sfc.get_substrate_node(sfc.get_dst_vnf())

    #     net_info = substrate_network
    #     network_topology = net_info._adj

    #     service_requirements = self.service_requirements
    #     services = self.services

    #     # Criação do grafo representando a rede com capacidade de banda
    #     A = nx.Graph()
    #     for node, edges in network_topology.items():
    #         for target, edge_attr in edges.items():
    #             bandwidth_free = edge_attr['bandwidth_capacity'] - edge_attr['bandwidth_used']

    #             A.add_edge(node, target, bandwidth=bandwidth_free, weight=1)

    #     G,server_resources = self.cut_topology(G_complex=A,
    #                       complex_network_topology=network_topology,
    #                       server_resources_complex=self.server_resources,
    #                       node_reference=dst,hops_cuff=5)

    #     #server_resources = self.server_resources        
    #     all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(A, weight='weight'))

    #     # Removing the restriction from the structures
    #     restriction = vnf_info[1]['original_loc']
    #     if restriction in list(self.can_alocate_sf_in_node[vnf_id]):
    #         self.can_alocate_sf_in_node[vnf_id].remove(restriction)
    #     if restriction in self.edge_computing_servers:        #del self.node_info[restriction]
    #         self.edge_computing_servers.remove(restriction)
    #     #del self.server_resources[restriction]

    #     a = time.time()
    #     route_info, latency = self.genetic_backup(G,A,service_requirements,services,src,dst,all_pairs_shortest_path)
    #     b = time.time()
    #     self.elapsed_time = (b-a)
        
    #     if latency > self.latency_request or route_info == False:
    #         self.latency = None
    #         self.route_info = False
    #         return False
    #     else:
    #         self.latency = latency
    #         self.route_info = route_info
    #         return True

    # def genetic_backup(self, G,G_old, service_requirements, services,src,dst,all_pairs_shortest_path):
    #     available_servers = self.can_alocate_sf_in_node[services[1]]
    #     def evaluate(individual):
    #         reuse = services[1] in self.server_resources[individual[0]]['reuse']
    #         node_resource_cost = 0.2 if reuse else 1
            
    #         src_to_vnf = (len(all_pairs_shortest_path[src][individual[0]]) - 1) 
    #         vnf_to_dst = (len(all_pairs_shortest_path[individual[0]][dst]) - 1) 
            
    #         total_cost = (self.cpu_weight * node_resource_cost) + (self.cache_weight * node_resource_cost) + (src_to_vnf+vnf_to_dst) * 2
    #         return total_cost,

    #     def custom_mutation(individual):
    #         """ Muda o gene do indivíduo para outro dentro do intervalo permitido. """
    #         new_gene = random.choice(available_servers)
    #         while new_gene == individual[0]:  # Garante que o valor mude
    #             new_gene = random.choice(available_servers)
    #         individual[0] = new_gene
    #         return individual,

    #     def custom_crossover(ind1, ind2):
    #         """ Crossover desativado (probabilidade 0), mas a função existe por compatibilidade. """
    #         pass
        
    #     a = time.time()
    #     service_requirements_local = service_requirements

    #     # Verifica se a classe já existe e, em caso afirmativo, exclui-a
    #     if hasattr(creator, "FitnessMin"):
    #         del creator.FitnessMin
    #     if hasattr(creator, "Individual"):
    #         del creator.Individual

    #     # DEAP setup para minimizar o fitness
    #     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    #     creator.create("Individual", list, fitness=creator.FitnessMin)

    #     toolbox = base.Toolbox()
    #     toolbox.register("individual", tools.initIterate, creator.Individual, lambda: random.sample(available_servers, 1))
    #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #     toolbox.register("mate",custom_crossover)
    #     toolbox.register("mutate",custom_mutation)
    #     toolbox.register("select", tools.selTournament, tournsize=3)
    #     toolbox.register("evaluate", evaluate)
    #     # Registrar o método de paralelização
    
    #     # Parâmetros do algoritmo genético
    #     population_size = 20
    #     crossover_probability = 0
    #     mutation_probability = 0.2
    #     number_of_generations = 20

    #     # Inicialização da população
    #     pop = toolbox.population(n=population_size)

    #     # Algoritmo genético
    #     algorithms.eaSimple(pop, toolbox, crossover_probability, mutation_probability, ngen=number_of_generations,verbose=False)

    #     def display_paths_and_create_service_dict(best_individual):
    #         paths = []  # Lista para armazenar os caminhos ótimos
    #         service_to_server_dict = {}
    #         service_names = list(service_requirements_local.keys())  # Assumindo que você tem os nomes dos serviços
            
    #         for i, server_id in enumerate(best_individual):
    #             service_name = service_names[i]
    #             if i < len(best_individual) - 1:
    #                 next_server_id = best_individual[i + 1]
    #                 if server_id == next_server_id:
    #                     service_to_server_dict[service_name] = [server_id]
    #                 else:
    #                     path = nx.shortest_path(G_old, source=server_id, target=next_server_id, weight='weight')
    #                     service_to_server_dict[service_name] = path
    #             else:
    #                 service_to_server_dict[service_name] = [server_id]
    #         return service_to_server_dict

    #     # Após a execução do algoritmo genético
    #     best_ind = tools.selBest(pop, 1)[0]
        
    #     if best_ind.fitness.values[0] != float('inf'):
    #         best_ind = [src] + best_ind + [dst]
    #         service_to_server_dict = display_paths_and_create_service_dict(best_ind)

    #         #service_to_server_dict.popitem()
    #         # Inverter a ordem dos itens no dicionário
    #         route_info = dict(reversed(list(service_to_server_dict.items())))
    #         #src_node = next(reversed(route_info.values()))[0]

    #         #path_to_src = list(reversed(nx.dijkstra_path(G_old, src_node, 0, weight='weight')))
            
    #         total_latency = sum(len(path) - 1 for path in route_info.values() if path) 

    #         #route_info['src'] = path_to_src
    #         #route_info['dst'] = []
    #         b = time.time()
    #         elapsed_time_ms = (b - a) * 1000  # Convertendo para milissegundos

    #         # print(f"Tempo total de avaliação: {self.evaluation_time:.2f} ms")
    #         # print(f"Tempo total de crossover: {self.crossover_time:.2f} ms")
    #         # print(f"Tempo total de mutação: {self.mutation_time:.2f} ms")
    #         print(f"Tempo de execução: {elapsed_time_ms:.2f} ms")
    #         return route_info, total_latency            
    #     else:
    #         return False, 100


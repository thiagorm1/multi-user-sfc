import re 
import time

from numpy import copy
class Crasher():
    """Simulates network node failures based on specified modes and probabilities."""
    
    def __init__(self,edge_servers=None,edges_vnf=None,time=30,crash_links=False):
        self.time = time
        self.ec_servers = edge_servers
        self.edges_vnf = edges_vnf
        self.crash_links = False
        self.trials = 0
        self.nodes_crashed = []
        self.cluster_to_crash = []
        self.statistics = {}
    
        #self.a_server_was_crashed = 0
        #self.server_to_reroute = None
        #self.links_to_crash = []

    def cluster_method(self,network):
        node_choose = None
        edges = network.sfs_flux_info.keys()
        highest_consume = 0
        for node in self.ec_servers:
            edges_partners = [node]
            for edge in edges:
                if node in edge:
                    # Identifica o servidor parceiro na edge
                    server_par = edge[0] if node_choose != edge[0] else edge[1]
                    if server_par in self.ec_servers and server_par != 0:
                        edges_partners.append(server_par)  # Armazena a edge na lista de edges a serem derrubadas
            
            # Remove duplicatas da lista de nós a serem derrubados
            edges_partners = list(set(edges_partners))
            consumo = 0
            for edge_server in edges_partners:
                # if node != edge_server:
                cpu_used = network.get_node_cpu_used(edge_server)
                consumo = consumo + cpu_used
            if  highest_consume < consumo:
                highest_consume = consumo
                node_choose     = node
        return node_choose

    def most_sf_type(self,network,sf_type='EC'):
        node_choose = None
        max_sfc = -1
        max_cpu_cache_usage = -1
        for node in self.ec_servers:
            unique_sfc_set = set()  # Usar um conjunto para rastrear SFCS únicas no nó
            for sfc_vnf in network.get_node_sfc_vnf_list(node):
                sfc_type = sfc_vnf[0].split("_")[1]
                vnf_id = sfc_vnf[1].id.split("_")[0]
                if (vnf_id == sf_type):
                    unique_sfc_set.add(sfc_vnf[0])  # Adiciona ao conjunto (evita duplicados)
            
            unique_sfc_count = len(unique_sfc_set)  # Conta as SFCs únicas
            cpu_used = network.get_node_cpu_used(node)  # Obtém o uso de CPU do nó
            cache_used = network.get_node_cache_used(node)  # Obtém o uso de cache do nó
            cpu_cache_usage = cpu_used + cache_used  # Soma para considerar o uso total de recursos
            
            # Atualiza o nó com mais SFCs únicas ou desempata com base no uso de CPU e cache
            if (unique_sfc_count > max_sfc): #or (unique_sfc_count == max_sfc and cpu_cache_usage > max_cpu_cache_usage):
                max_sfc = unique_sfc_count
                max_cpu_cache_usage = cpu_cache_usage
                node_choose = node
        return node_choose

    def highest_resource_consumer(self, network):
        """
        Seleciona o nó com o maior consumo de recursos (CPU e cache) na rede.
        
        Args:
            network: Objeto que representa a rede.
            
        Returns:
            node_choose: O nó que consome mais recursos.
        """
        node_choose = None
        max_resource_usage = -1  # Variável para rastrear o maior consumo de recursos
        
        for node in self.ec_servers:
            # Obtém o consumo de CPU e cache para o nó
            cpu_used   = network.get_node_cpu_used(node)  
            cache_used = network.get_node_cache_used(node) 
            total_resource_usage = cpu_used + cache_used  # Soma dos recursos usados
            
            # Atualiza o nó com maior consumo de recursos
            if total_resource_usage > max_resource_usage:
                max_resource_usage = total_resource_usage
                node_choose = node
        return node_choose
    
    def crash_cluster(self, network):
        node_choose = None
        edges = network.sfs_flux_info.keys()
        highest_consume = 0
        if self.cluster_to_crash == []:
            for node in self.ec_servers:
                edges_partners = [node]
                for edge in edges:
                    if node in edge:
                        # Identifica o servidor parceiro na edge
                        server_par = edge[0] if node_choose != edge[0] else edge[1]
                        if server_par in self.ec_servers and server_par != 0:
                            edges_partners.append(server_par)  # Armazena a edge na lista de edges a serem derrubadas

                # Remove duplicatas da lista de nós a serem derrubados
                edges_partners = list(set(edges_partners))
                consumo = 0
                for edge_server in edges_partners:
                    # if node != edge_server:
                    cpu_used = network.get_node_cpu_used(edge_server)
                    consumo = consumo + cpu_used
                if  highest_consume < consumo:
                    highest_consume = consumo
                    node_choose     = node

            edges_from_cluster = []
            for edge in edges:
                if node in edge:
                    # Identifica o servidor parceiro na edge
                    server_par = edge[0] if node_choose != edge[0] else edge[1]
                    if server_par in self.ec_servers and server_par != 0:
                        edges_from_cluster.append(server_par)  # Armazena a edge na lista de edges a serem derrubadas

            edges_from_cluster = list(set(edges_from_cluster))
            self.cluster_to_crash = edges_partners
        
        node_choose = None
        for node in self.cluster_to_crash:
            node_choose = self.cluster_to_crash.pop(0) 
        return node_choose

    def activate_crasher(self, network,sfc_manager,alg_name):
        """Activates the crasher on the given network, excluding specific nodes."""
        nodes_info = {}
        
        node_choose = None
        h_rel = 0

        # lowest_rel = 100
        # for node in self.ec_servers:
        #     rel = network.get_node_reliability(node)
        #     cpu_used = network.get_node_cpu_used(node)
        #     cache_used = network.get_node_cpu_used(node)

        #     if rel < lowest_rel and (cpu_used >0) and (cache_used>=0):
        #         lowest_rel = rel
        #         node_choose = node
        # print(lowest_rel)
        # print(node_choose)


        # #node_choose = 5
        #     #print(rel)
        node_choose = None
        if alg_name != 'vegeta' :
            #node_choose = self.cluster_method(network=network)
            #node_choose = self.most_sf_type(network,sf_type='RE')
            #node_choose = self.highest_resource_consumer(network)
            node_choose = self.crash_cluster(network)
            
            # lowest_rel = 100
            # for node in self.ec_servers:
            #     rel = network.get_node_reliability(node)
            #     cpu_used = network.get_node_cpu_used(node)
            #     cache_used = network.get_node_cpu_used(node)

            #     if rel < lowest_rel and (cpu_used >0) and (cache_used>=0):
            #         lowest_rel = rel
            #         node_choose = node
            # print(lowest_rel)


            # node_choose = None
            # h_rel = 0
            # for node in self.ec_servers:
            #     cpu_used = network.get_node_cpu_used(node)
            #     cache_used = network.get_node_cpu_used(node)
            #     #     if cpu_used >= 50 and cache_used >= 50:
            #     total = cpu_used + cache_used
            #     if total > h_rel:
            #         h_rel = total
            #         node_choose = node
            # node_choose = None
            # max_sfc = -1
            # max_cpu_cache_usage = -1

            # for node in self.ec_servers:
            #     unique_sfc_set = set()  # Usar um conjunto para rastrear SFCS únicas no nó
                
            #     for sfc_vnf in network.get_node_sfc_vnf_list(node):
            #         sfc_type = sfc_vnf[0].split("_")[1]
            #         vnf_id = sfc_vnf[1].id.split("_")[0]
            #         if sfc_type == 'unique' :#and (vnf_id == 'UNI' or vnf_id == 'RE'):
            #             unique_sfc_set.add(sfc_vnf[0])  # Adiciona ao conjunto (evita duplicados)
                
            #     unique_sfc_count = len(unique_sfc_set)  # Conta as SFCs únicas
            #     cpu_used = network.get_node_cpu_used(node)  # Obtém o uso de CPU do nó
            #     cache_used = network.get_node_cache_used(node)  # Obtém o uso de cache do nó
            #     cpu_cache_usage = cpu_used + cache_used  # Soma para considerar o uso total de recursos
                
            #     # Atualiza o nó com mais SFCs únicas ou desempata com base no uso de CPU e cache
            #     if (unique_sfc_count > max_sfc) or \
            #     (unique_sfc_count == max_sfc and cpu_cache_usage > max_cpu_cache_usage):
            #         max_sfc = unique_sfc_count
            #         max_cpu_cache_usage = cpu_cache_usage
            #         node_choose = node


# Após o loop, `node_more_sfc` terá o nó com mais SFCs únicas

            
        # if node_choose == None:
        #     return False
        if alg_name == 'vegeta':
    ###############################################################
            node_choose = None
            h_rel = 0
            
            nodes_rel = network.nodes_reliability.copy()
            for node,rel in nodes_rel.items():
                if rel > h_rel:
                    h_rel = rel
                    node_choose = node
    ###############################################################

                
        # sfcs_with_backup = list(sfc_manager.sfs_backup.keys())
        # server_backup_count = {}

        # # Itera sobre os servidores
        # node_more_cpu = None  # Nenhum servidor inicial selecionado
        # max_cpu_used = -1  # Valor inicial menor que qualquer possível uso de CPU

        # # Itera sobre os servidores
        # for node in self.ec_servers:
        #     cpu_used = network.get_node_cpu_used(node)
        #     # Verifica se este servidor tem mais CPU usada que o máximo atual
        #     if cpu_used > max_cpu_used:
        #         max_cpu_used = cpu_used
        #         node_more_cpu = node
        # # Define o servidor escolhido
        # node_choose = node_more_cpu

        # node_more_sfc = None  
        # max_sfc = -1  
        # for node in self.ec_servers:
        #     count_unique = []  # Lista para rastrear SFCS únicas no nó
        #     for sfc_vnf in network.get_node_sfc_vnf_list(node):
        #         sfc_type = sfc_vnf[0].split("_")[1]
        #         if sfc_type == 'unique':
        #             count_unique.append(sfc_vnf[0])  # Adiciona a SFC se não estiver na lista
            
            
        #sfcs_in_node = len(count_unique)  # Calcula o número de SFCS únicas no nó

        #     # Verifica se este nó tem mais SFCS únicas que o máximo atual
        #     if sfcs_in_node > max_sfc:
        #         max_sfc = sfcs_in_node
        #         node_more_sfc = node

        # Define o servidor escolhido
        # node_choose = node_more_sfc


        #reuse_quantity 
        if self.crash_links:
            nodes_to_crash = [node_choose]
            
            # Coleta todas as edges (conexões) da rede
            edges = network.sfs_flux_info.keys()

            # Lista para armazenar as edges do servidor escolhido
            edges_to_crash = []
            
            for edge in edges:
                # Verifica se o node escolhido (node_choose) está na edge
                if node_choose in edge:
                    # Identifica o servidor parceiro na edge
                    server_par = edge[0] if node_choose != edge[0] else edge[1]
                    
                    # Se o servidor parceiro não estiver na lista de servidores de edges ou for um valor específico, adicione-o
                    if server_par not in self.ec_servers and server_par != 0 and server_par != 34:
                        nodes_to_crash.append(server_par)

                    edges_to_crash.append(edge)  # Armazena a edge na lista de edges a serem derrubadas
            # Remove duplicatas da lista de nós a serem derrubados
            nodes_to_crash = list(set(nodes_to_crash))
            self.nodes_crashed = nodes_to_crash
            return nodes_to_crash
        else:
            self.nodes_crashed.append(node_choose)
            return [node_choose]

    def recover_from_crash(self, network):
        nodes_crashed = list(self.nodes_crashed)
        if len(self.nodes_crashed) != 0: 
            print(f"Recuperando servidor: {self.nodes_crashed}")
            for server in nodes_crashed:
                # Definir capacidades negativas para simular o crash
                network.set_node_cache_capacity(server, 100)
                network.set_node_cpu_capacity(server, 100)
        self.nodes_crashed = []

    def implement_crash(self, nodes_crashed, network):
        sfcs_crashed = {}
        if len(nodes_crashed) != 0: 
            print(f"Servidores Crashados: {nodes_crashed}")
            self.sfcs_crashed = {}
            sfc_ids = []
            sfcs_to_crash = []

            for server in nodes_crashed:
                server_info = network.get_node_sfc_vnf_list(server)
                #filtered_edges = {key: value for key, value in self.edges_vnf.items() if server in key}
                
                # Definir capacidades negativas para simular o crash
                network.set_node_cache_capacity(server, -0.0000001)
                network.set_node_cpu_capacity(server, -0.0000001)
                
                # if self.crash_links:
                #     for link, sfc_vnf in filtered_edges.items():
                #         network.set_link_bandwidth_capacity(link[0], link[1], -0.0000001)
                #         network.set_link_latency(link[0], link[1], 10000)
                    
                if server_info != []:
                    # Extrai os sfc_ids
                    sfc_ids = list(set([sfc[0] for sfc in server_info]))
                    # users_crashed = []
                    # pattern = re.compile(r'p\d+_\d+')

                    # for sfc_id in sfc_ids:
                    #     # Popula o dicionário sfcs_crashed com usuários afetados
                    #     match = pattern.search(sfc_id)
                    #     if match:
                    #         users_crashed.append(match.group())
                    # users_crashed = list(set(users_crashed))

                    # Para cada sfc_id crashada, extrair as VNFs afetadas
                    for sfc_id in sfc_ids:
                        if sfc_id not in network.sfc_dict:
                            continue
                        
                        sfc = network.get_sfc_by_id(sfc_id)
                        sfc_rf = network.sfc_route_info[sfc_id]

                        latency_sfc= sum((len(value) - 1) for key, value in sfc_rf.items() if key not in ('src', 'dst'))
                        
                        # Obter os IDs das VNFs crashadas que pertencem a esta sfc_id
                        crashed_vnf_ids = [vnf.id for sid, vnf in server_info if sid == sfc_id]

                        sfcs_crashed[sfc_id] = {
                            'fall_time': time.time(),
                            'old_latency': latency_sfc,
                            'vnf_ids': crashed_vnf_ids
                        }
        return sfcs_crashed

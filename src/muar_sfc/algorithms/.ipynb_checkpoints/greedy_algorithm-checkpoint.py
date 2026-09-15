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
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
from config import ROOT_PATH
ch = logging.FileHandler(ROOT_PATH + './logs/GreedyAlgorithm.log')
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class GreedyAlgorithm():
    '''Greedy Algorithm.
    This algorithm starts from the substrate network node which hosts src of an SFC, checks its neighbor nodes,
    finds the neighbor node with a shortest latency edge, and use the node to host the vnf.
    The algorithm greedily finds all nodes for hosting vnf.
    Finally, the algorithm finds a shortest path from the substrate node who hosts the last vnf in the SFC
    to the substrate node who hosts dst of the SFC.

    Deploy VNF one by one, with a shortest path from the node to the previous substrate node.
    '''
    def __init__(self):
        self.name = "Greedy Algorithm"
        self.substrate_network = None
        self.sfc = None
        self.node_info = None
        self.route_info = None
        self.latency = None
        self.mono = False
        self.old_greedy = True

    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.node_info = None
        self.route_info = None
        self.latency = None

    def install_substrate_network(self, substrate_network):
        self.substrate_network = substrate_network
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        return self.sfc

    def start_algorithm(self):
        substrate_network = self.substrate_network
        sfc = self.sfc
        logger.info("Start algorithm")
        if self.algorithm(substrate_network, sfc):
            logger.info("End algorithm, success")
            return True
        logger.info("End algorithm, failed")
        return False

    def get_latency(self):
        return self.latency
    
    def get_route_info(self):
        return self.route_info

    def algorithm(self, substrate_network, sfc):
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advance
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        route_info = {}
        bandwidth_usage_info = {}

        latency = 0
        used_node = [dst_substrate_node, src_substrate_node]

        number_of_vnfs = sfc.get_number_of_vnfs()
        current_vnf = dst_vnf
        current_substrate_node = dst_substrate_node

        net_info = substrate_network 
        old_server_resources = net_info._node
        servers = list(old_server_resources.keys())
        
        # Inicializa o dicionário de recursos dos servidores
        server_resources = {
            server: {
                'cpu_capacity': substrate_network.get_node_cpu_capacity(server),
                'cache_capacity':substrate_network.get_node_cache_capacity(server),
                'cpu_used': substrate_network.get_node_cpu_used(server),
                'cache_used': substrate_network.get_node_cache_used(server),
                'cpu_free': substrate_network.get_node_cpu_free(server),
                'cache_free': substrate_network.get_node_cache_free(server),
                'position': old_server_resources[server]['position'],
                'reuse': []
            } for server in servers}

        nodes_used = []
        for i in range(number_of_vnfs - 1, -1, -1):
            edges = list(substrate_network.edges(current_substrate_node))
            edges = list(set([x[1] for x in edges]))

            #edges.append((current_substrate_node, current_substrate_node))
            prev_vnf = current_vnf.get_previous_vnf()

            cpu_request = sfc.get_vnf_cpu_request(prev_vnf)
            cache_request = sfc.get_vnf_cache_request(prev_vnf)
            bandwidth_request = sfc.get_link_bandwidth_request(prev_vnf.id, current_vnf.id)

            min_latency = float("inf")
            node = None

            if self.old_greedy:
                nodes_to_check = edges
            else:
                nodes_to_check = servers
                
            for node_a in nodes_to_check:
                if not self.mono:
                    if node_a == current_substrate_node or node_a in nodes_used:
                        continue
                # Agora buscamos valores diretamente em server_resources:
                cpu_available = server_resources[node_a]['cpu_free']
                cache_available = server_resources[node_a]['cache_free']

                cpu_used = server_resources[node_a]['cpu_used']
                cache_used = server_resources[node_a]['cache_used']

                cpu_cap = server_resources[node_a]['cpu_capacity']
                cache_cap = server_resources[node_a]['cache_capacity']

                
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
                    logger.debug("Node %s não tem CACHE suficiente para %s",node_a, cache_request)
                    continue

                # Verificando link (apenas se não for laço no mesmo nó)
                if node_a == node:
                    edge_latency = 0
                else:
                    # bandwidth_available = substrate_network.get_link_bandwidth_free(e[0], e[1])
                    # if bandwidth_request > bandwidth_available:
                    #     logger.debug("Aresta (%s, %s) sem banda suficiente", e[0], e[1])
                    #     continue
                    edge_latency = substrate_network.single_source_minimum_latency_path[current_substrate_node][0][node_a]

                # Verifica se a latência desse caminho é a menor
                if edge_latency < min_latency:
                    min_latency = edge_latency
                    node = node_a

            # Se encontrou um nó para alocar
            if node is not None:
                nodes_used.append(node)
                # Atualizamos o dicionário de recursos
                server_resources[node]['cpu_used'] += cpu_request
                server_resources[node]['cpu_free'] -= cpu_request

                server_resources[node]['cache_used'] += cache_request
                server_resources[node]['cache_free'] -= cache_request
                
                # Se for usar a banda, você também decrementa a banda do enlace
                # se node != current_substrate_node, por exemplo
                # Ajuste do route_info e soma de latência
                if node == current_substrate_node:
                    route_info[prev_vnf.id] = [node]
                else:
                    route_info[prev_vnf.id] = substrate_network.single_source_minimum_latency_path[node][1][current_substrate_node]
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
            path = substrate_network.get_shortest_path(src_substrate_node, node)
            path_latency = substrate_network.get_shortest_path_length(src_substrate_node, node)
        except:
            logger.warning('Não há caminho entre src e primeira VNF: %s - %s',
                        src_substrate_node, node)
            self.route_info = {}
            self.latency = None
            return False

        # Adicionamos esse path como 'src' no route_info
        route_info['src'] = path
        route_info['dst'] = []
        latency += path_latency

        # Define route_info e latency no objeto
        self.route_info = route_info
        self.latency = latency

        # Se você precisa fazer algum ajuste de latência baseado em edges do path:
        path = self.route_info['src']
        for i in range(len(path) - 1):
            edge_latency = substrate_network.get_link_latency(path[i], path[i + 1])
            self.latency = self.latency - edge_latency

        # Algumas checagens que já existiam
        if len(self.route_info.keys()) != 6:
            self.latency = None
            self.route_info = {}
            return False

        if self.latency > sfc.get_latency_request():
            self.route_info = {}
            return False
        return True


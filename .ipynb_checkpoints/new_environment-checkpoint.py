import copy
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces

# Supondo que essas funções existem em um módulo `algorithms.networkUtils`
from muar_sfc.algorithms.networkUtils import get_available_shortest_path_optimized, calculate_computational_latency, calculate_latency_betwen_nodes

# --- CONSTANTES ---
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

# --- FUNÇÕES AUXILIARES EXTERNAS ---
def latency_rounded(u, v, data):
    """Função de peso para o NetworkX, usada para arredondar a latência."""
    return int(round(data.get('latency', 0), 3) * 1000)

def equalizar_listas(lista1, lista2):
    """
    Ajusta o tamanho de duas listas para que fiquem do mesmo comprimento.
    Remove os últimos elementos da lista maior até igualar ao tamanho da menor.
    """
    while len(lista1) > len(lista2):
        lista1.pop()
    while len(lista2) > len(lista1):
        lista2.pop()
    return lista1, lista2

# --- CLASSE DO AMBIENTE ---
class NetworkEnv(gym.Env):
    """
    Ambiente customizado do Gymnasium para simular a alocação de Service Function Chains (SFCs)
    em uma rede de substrato. O objetivo de um agente de RL neste ambiente é aprender a
    alocar uma cadeia de serviços sequencialmente, minimizando um custo combinado de
    recursos computacionais (CPU, cache), latência e largura de banda.
    """
    def __init__(self, list_graph: list, list_sfc: list,valid_nodes: list, pesos, is_training=True): # MODIFICADO
        super().__init__()

        list_graph, list_sfc = equalizar_listas(list_graph, list_sfc)

        # --- Validação das listas ---
        if not list_graph or not list_sfc:
            raise ValueError("As listas de grafos e SFCs não podem ser vazias.")
        if len(list_graph) != len(list_sfc):
            raise ValueError("A lista de grafos e a lista de SFCs devem ter o mesmo tamanho.")

        # --- Topologia e Configuração da Rede ---
        self.list_graph = list_graph
        self.list_sfc = list_sfc
        self.G = None  # Será definido a cada reset
        self.initial_resource_snapshot = {}
        self.cached_paths = {}

        # --- Parâmetros da SFC e Estado Atual ---
        self.sfc = None
        self.services = None
        self.service_requirements = None
        self.latency_request = None
        self.dst_node = None
        self.current_location = None
        self.service = None
        self.session_number = None
        self.valid_nodes = valid_nodes  # Lista de nós válidos para alocação

        # --- Métricas de Desempenho e Controle ---
        self.latency_used = 0
        self.is_training = is_training
        self.servers_used = []
        self.allocation_results = {}
        self.success = False
        self.reuse = False
        
        # --- Fatores de Custo para a Recompensa ---
        self.cpu_factor = pesos.get("cpu", 1)
        self.cache_factor = pesos.get("cache", 1)
        self.band_factor = pesos.get("band", 1)
        self.latency_factor = pesos.get("latency", 1)
        self._initialize_snapshots()
        # --- Espaços de Ação e Observação ---
        self.action_space = spaces.Discrete(len(self.valid_nodes))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(valid_nodes) * 6,),  # 5 métricas por nó válido
            dtype=np.float32
        )


    # --------------------------------------------------------------------------
    # --- MÉTODOS PRINCIPAIS DO GYMNASIUM (CORE API) ---
    # --------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = np.random.randint(len(self.list_graph))
        
        # Define self.G como uma referência ao grafo da lista (que será modificado)
        self.G = self.list_graph[idx]
        sfc_for_episode = self.list_sfc[idx]

        # Busca o snapshot PRÉ-CALCULADO e limpo do grafo correspondente
        snapshot_nodes = self.initial_resource_snapshot[idx]['nodes']
        for node_id, initial_state in snapshot_nodes.items():
            if node_id in self.G.nodes:
                self.G.nodes[node_id]['cpu_used'] = initial_state['cpu_used']
                self.G.nodes[node_id]['cache_used'] = initial_state['cache_used']

        snapshot_edges = self.initial_resource_snapshot[idx]['edges']
        for (u, v), initial_state in snapshot_edges.items():
            if self.G.has_edge(u, v):
                self.G.edges[u, v]['bandwidth_used'] = initial_state['bandwidth_used']

        # Agora que o grafo está resetado, configure a SFC para o episódio
        self.set_sfc(sfc_for_episode)

        # Reinicializa o resto do estado do episódio
        self.latency_used = 0
        self.servers_used.clear()
        self.cached_paths.clear()
        self.reward = 0
        self.total_reward = 0
        self.total_cost = 0
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        
        return self.get_normalized_state(), {}
    def step(self, action):
        """
        Executa um passo no ambiente a partir de uma ação do agente.
        A ação corresponde a escolher um nó para alocar o serviço atual da SFC.
        """
        if action == len(self.valid_nodes) - 1:
            # Se for, o servidor escolhido é o dst_node da SFC atual
            chosen_server = self.sfc.dst_node
        else:
            # Caso contrário, o mapeamento é direto da lista
            chosen_server = self.valid_nodes[action]

        # Agora, o resto da função usa 'chosen_server' que já foi traduzido corretamente
        self.server = chosen_server

        self.reuse = self.is_reusable_at_node(self.server, self.service, self.sfc.id.split("_")[-1])

        if not self.allocate_resources_on_node(self.server, self.reuse):
            return self._fail_step('resource')

        self.path = get_available_shortest_path_optimized(
            self.G, self.current_location, chosen_server, 
            self.bandwidth_required, rounded=True
        )
        if not self.path:
            return self._fail_step('bandwidth')

        # 3. Alocar banda e calcular latência do caminho
        success_band, path_latency = self.allocate_bandwidth_along_path(self.path, self.bandwidth_required, self.service)
        
        if not success_band:
            # Esta verificação é uma dupla segurança, o fallback já deveria ter resolvido.
            return self._fail_step('bandwidth')

        self.latency_used += path_latency
        
        
        
        # 4. Verificar restrição de latência
        if self.latency_used > self.latency_request:
            return self._fail_step('latency')

        # 5. Calcular custo e recompensa
        self.servers_used.append(self.server)
        self.total_cost = self.calculate_total_cost(self.server, self.path)
        if True:
            print(f"Custos envolvendo nó movel: {self.total_cost}")
        self.reward = -self.total_cost
        self.total_reward += self.reward

        # 6. Atualizar estado para o próximo passo
        
        self.current_location = self.server

        if not self.is_training:
            self.allocation_results[self.service] = {'allocated_server': self.server, 'path': self.path, 'cost': self.total_cost}

        # 7. Verificar se o episódio terminou
        done = False
        if self.service == self.services[-1]:
            done = True
            self.success = True

        else:
            current_index = self.services.index(self.service)
            self.service = self.services[current_index + 1]
            self.update_bandwidth_required()
        
        return self.get_normalized_state(), self.reward, done, False, {}
    
    # --------------------------------------------------------------------------
    # --- MÉTODOS DE CONFIGURAÇÃO DO AMBIENTE ---
    # --------------------------------------------------------------------------

    def set_graph(self, graph):
        """Define e faz backup do grafo da rede."""
        # self.G_backup = copy.deepcopy(graph)
        self.G = graph
        

    def set_list_graph(self, list_graph):
        """Define a lista de grafos para o ambiente."""
        if not isinstance(list_graph, list):
            raise ValueError("list_graph deve ser uma lista de grafos.")
        self.list_graph = list_graph

    def set_list_sfcs(self, list_sfc):
        """Define a lista de SFCs para o ambiente."""
        if not isinstance(list_sfc, list):
            raise ValueError("list_sfc deve ser uma lista de SFCs.")
        self.services = list_sfc

    def set_sfc(self, sfc):
        """Define a SFC atual e captura suas propriedades."""
        if not sfc:
            raise ValueError("SFC não pode ser None.")
        self.sfc = sfc
        # print(f"Trocou dst_node para: {self.sfc.dst_node}")
        self.services, self.service_requirements = self.prepare_service_requirements(self.sfc.vnfs_dict)
        self.latency_request = 10
        self.dst_node = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        self.current_location = self.dst_node
        self.service = self.services[0]
        self.update_bandwidth_required()

    def set_dst_node(self, dst_node):
        """Define o nó de destino inicial da SFC."""
        if not dst_node:
            raise ValueError("Nó de destino (dst_node) não pode ser None.")
        self.dst_node = dst_node
        self.current_location = dst_node

    

    # --------------------------------------------------------------------------
    # --- MÉTODOS AUXILIARES DE LÓGICA INTERNA ---
    # --------------------------------------------------------------------------

    def _fail_step(self, reason):
        """Padroniza a finalização de um episódio por falha."""
        self.fail_reason = reason
        self.success = False
        self.reward = -2000  # Penalidade fixa e alta por falha
        done = True

        return self.get_normalized_state(), self.reward, done, False, {}

    def _get_path_with_fallback(self, source, target):
        """Tenta obter o caminho do cache, senão busca o caminho mais curto com banda disponível."""
        # Tenta o caminho mais rápido (Dijkstra puro)

        if (source, target) not in self.cached_paths:
            path = get_available_shortest_path_optimized(self.G, source, target, self.bandwidth_required, rounded=True)
            self.cached_paths[(source, target)] = path
        else:
            path = self.cached_paths[(source, target)]
        
        # Verifica se este caminho rápido tem banda
        if self.get_critical_link_bandwidth(path) >= self.bandwidth_required:
            return path
        
        # Fallback: Se não tem banda, busca um caminho viável (pode ser mais lento)
        path = get_available_shortest_path_optimized(self.G, source, target, self.bandwidth_required, rounded=True)
        if not path:
            return self.cached_paths[(source, target)]
        else:
            self.cached_paths[(source, target)] = path
            return path
        

    # environment.py

    def allocate_resources_on_node(self, node_id, is_reusable):
        """Aloca CPU e Cache em um nó, considerando a possibilidade de reuso."""
        
        vnf = self.sfc.get_vnf_by_id(self.service)
        
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        can_reuse = self.is_reusable_at_node(node_id,vnf.id,self.sfc.id.split("_")[-1])
        # Recurso necessário é zero se for reutilizável
        effective_cpu_req = 0 if can_reuse else cpu_req
        effective_cache_req = 0 if can_reuse else cache_req

        # Esta verificação de segurança agora usa o valor efetivo (corrigindo o Erro 2)
        if (self.G.nodes[node_id]['cpu_used'] + cpu_req > self.G.nodes[node_id]['cpu_capacity']) or \
            (self.G.nodes[node_id]['cache_used'] + cache_req) > self.G.nodes[node_id]['cache_capacity']:
            return False
        
        # Aloca os recursos efetivos
        self.G.nodes[node_id]['cpu_used'] += effective_cpu_req
        self.G.nodes[node_id]['cache_used'] += effective_cache_req


        return True
    
    def allocate_bandwidth_along_path(self, path, bandwidth_required, ms_name):
        """
        Aloca banda ao longo de um caminho usando uma abordagem de duas passagens (verificar, depois alocar).
        Retorna (True, latência_total) em sucesso, ou (False, 0.0) em falha.
        """
        if not path or len(path) < 2:
            return True, 0.0

        # --- 1ª Passagem: VERIFICAÇÃO ---
        # Percorre todo o caminho para garantir que cada enlace tem capacidade suficiente.
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                # Se qualquer enlace no caminho falhar na verificação, a operação inteira é abortada.
                return False, 0.0

        # --- 2ª Passagem: ALOCAÇÃO (COMMIT) ---
        # Se o código chegou a este ponto, o caminho inteiro é válido.
        # Agora, percorremos o caminho novamente para efetivamente alocar os recursos.
        total_latency = 0.0
        vnf = self.sfc.get_vnf_by_id(ms_name)
        for u, v in zip(path[:-1], path[1:]):
            # A chamada para _commit_bandwidth_on_link agora é garantida de não exceder a capacidade.
            latency = self._commit_bandwidth_on_link(u, v, vnf, bandwidth_required)
            total_latency += latency

        return True, total_latency
        
    def _commit_bandwidth_on_link(self, u, v, vnf, bandwidth_required):
        """Aloca banda e calcula latência para um único enlace (u, v)."""
        edge = self.G.edges[u, v]
        edge['bandwidth_used'] += bandwidth_required
        latency = calculate_latency_betwen_nodes(self.G, u, v, vnf)
        return latency

    # new_environment.py

    def calculate_total_cost(self, server_id, path):
        """Calcula o custo total da alocação de um serviço."""
        node = self.G.nodes[server_id]

        # Custo de CPU e Cache (zero se houver reuso)
        cpu_capacity = node["cpu_capacity"] or 1
        cpu_cost = (node["cpu_used"] / cpu_capacity + 1) ** self.cpu_factor if not self.reuse else 0
        
        cache_capacity = node["cache_capacity"] or 1
        cache_cost = (node["cache_used"] / cache_capacity + 1) ** self.cache_factor if not self.reuse else 0
        
        # Custo de Rede (Latência e Banda)
        self.latency_cost = 0
        bandwidth_cost = 0
        if len(path) >= 2:
            latency_request = self.latency_request or 1
            self.latency_cost = ((self.latency_used / latency_request) + 1) ** self.latency_factor
            
            # Custo de banda baseado no link crítico do caminho (LINHAS DESCOMENTADAS)
            cap_band, used_band = self.get_critical_link_info(path)
            cap_band = cap_band or 1  # Evita divisão por zero
            bandwidth_cost = ((used_band / cap_band) + 1) ** self.band_factor
 
        return sum([cpu_cost, cache_cost, self.latency_cost, bandwidth_cost])
    # --------------------------------------------------------------------------
    # --- MÉTODOS DE GERAÇÃO DE ESTADO ---
    # --------------------------------------------------------------------------

    def get_normalized_state(self):
        """Gera o vetor de estado normalizado para o agente de RL."""
        state_vectors = []

        for i, node_id_or_placeholder in enumerate(self.valid_nodes):
            if i == len(self.valid_nodes) - 1:
                node_id = self.sfc.dst_node
            else:
                node_id = node_id_or_placeholder

            node = self.G.nodes[node_id]

            # 1. Custos de Recursos (CPU & Cache)
            cpu_capacity = node["cpu_capacity"] or 1
            cache_capacity = node["cache_capacity"] or 1
            is_reusable = int(self.is_reusable_at_node(node_id, self.service, self.sfc.id.split("_")[-1]))
            
            cpu_req = 0 if is_reusable else self.service_requirements[self.service]["cpu"]
            cache_req = 0 if is_reusable else self.service_requirements[self.service]["cache"]

            # Calcula custos projetados de CPU e Cache
            if (self.service_requirements[self.service]["cpu"] + node["cpu_used"]) > node["cpu_capacity"] or \
            (self.service_requirements[self.service]["cache"] + node["cache_used"]) > node["cache_capacity"]:
                proj_cpu_cost = 1.0
                proj_cache_cost = 1.0
            else:
                proj_cpu_cost = (node["cpu_used"] + cpu_req) / cpu_capacity
                proj_cache_cost = (node["cache_used"] + cache_req) / cache_capacity
            
            # 2. Custos de Rede (Latência e Banda)
            path = get_available_shortest_path_optimized(
                self.G, self.current_location, node_id, 
                self.bandwidth_required, rounded=True
            )

            # Lógica de custo de banda projetado (CORRIGIDA)
            proj_bandwidth_cost = 1.0
            if path:
                cap_band, used_band = self.get_critical_link_info(path)
                cap_band = cap_band or 1
                proj_bandwidth_cost = (used_band + self.bandwidth_required) / cap_band
            
            # Custo de latência projetado
            path_latency = self.calculate_path_latency(path, self.service)
            latency_request = self.latency_request or 1
            proj_latency_cost = (self.latency_used + path_latency) / latency_request

            # 3. Flag de "Não pode alocar"
            cant_allocate = 1.0 if (proj_cpu_cost > 1.0 or proj_cache_cost > 1.0 or not path or proj_latency_cost > 1.0 or proj_bandwidth_cost > 1.0) else 0.0

            # 4. Montagem do Vetor de Estado (MELHORADO)
            node_state = [
                min(proj_cpu_cost, 1.0),
                min(proj_cache_cost, 1.0),
                min(proj_latency_cost, 1.0),
                min(proj_bandwidth_cost, 1.0), # <- NOVA MÉTRICA ADICIONADA
                float(is_reusable),
                cant_allocate
            ]
            state_vectors.extend(node_state)

        return np.array(state_vectors, dtype=np.float32)

    # --------------------------------------------------------------------------
    # --- MÉTODOS UTILITÁRIOS E DE CONSULTA ---
    # --------------------------------------------------------------------------

    def update_bandwidth_required(self):
        """Atualiza a necessidade de banda para o serviço atual na SFC."""
        if not self.service:
            raise ValueError("Serviço atual não definido.")
        self.bandwidth_required = self.sfc.get_vnf_by_id(self.service).get_outcome_interface_bandwidth()

    def calculate_path_latency(self, path, ms_name):
        """Calcula a latência total de um caminho sem modificar o grafo."""
        if not path or len(path) < 2:
            return 0.0
        
        total_latency = 0.0
        vnf = self.sfc.get_vnf_by_id(ms_name)
        for u, v in zip(path[:-1], path[1:]):
            total_latency += calculate_latency_betwen_nodes(self.G, u, v, vnf)
        return total_latency

    def get_critical_link_bandwidth(self, path):
        """Retorna a menor banda disponível em um caminho."""
        if not path or len(path) < 2:
            return float('inf')
        
        min_available = float('inf')
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            available = edge.get('bandwidth_capacity', 0) - edge.get('bandwidth_used', 0)
            min_available = min(min_available, available)
        return min_available

    def get_critical_link_info(self, path):
        """Retorna a capacidade e o uso do link mais crítico (menor banda livre) em um caminho."""
        if not path or len(path) < 2:
            return 0, 0

        min_available = float('inf')
        crit_cap, crit_used = 0, 0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            cap = edge.get('bandwidth_capacity', 0)
            used = edge.get('bandwidth_used', 0)
            available = cap - used
            if available < min_available:
                min_available = available
                crit_cap, crit_used = cap, used
        return crit_cap, crit_used
    
    def is_shareable(self,service_name):
        # TODO Mudar para a informação de compartilháveis estar em uma variável separável.
        #if self.shareable_node:
        if True:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False

    # environment.py

    def is_reusable_at_node(self, node_id, service_name, session_id):

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False
        service_key = (service_name,session_id)
        if service_key in self.G.nodes[node_id]['services']:
                return True
        return False

    def _initialize_snapshots(self):
        """
        Cria um snapshot do estado inicial de todos os grafos na lista, uma única vez.
        Isso garante que temos uma cópia "limpa" de referência para o reset.
        """
        self.initial_resource_snapshot = {}
        for idx, graph in enumerate(self.list_graph):
            snapshot_nodes = {}
            for node_id, data in graph.nodes(data=True):
                snapshot_nodes[node_id] = {
                    'cpu_used': data.get('cpu_used', 0),
                    'cache_used': data.get('cache_used', 0),
                }
            
            snapshot_edges = {}
            for u, v, data in graph.edges(data=True):
                snapshot_edges[(u, v)] = {
                    'bandwidth_used': data.get('bandwidth_used', 0),
                }
            
            self.initial_resource_snapshot[idx] = {
                'nodes': snapshot_nodes,
                'edges': snapshot_edges
            }
        

    def prepare_service_requirements(self, sfs_dict):
        service_requirements = {}
        services = [item['name'] for item in sfs_dict]
        for item in sfs_dict:
            name = item['name']
            service_requirements[name] = {
                'cpu': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw'],
                'latency': item['latency']
            }
        service_requirements['dst'] = {'CPU': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0, 'latency': 0}
        return list(reversed(services)), service_requirements
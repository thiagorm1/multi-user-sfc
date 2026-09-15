import gymnasium
from gymnasium import spaces
import numpy as np
from networkx import Graph
from typing import Union, List, Dict
from muar_sfc.core.sfc import SFC, VNF
from muar_sfc.algorithms.networkUtils import get_available_shortest_path, calculate_computational_latency, calculate_latency_betwen_nodes, get_available_shortest_path_fast
from muar_sfc.utils.network_utils import calcular_percentual_cpu_total

import math
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')
PUNICAO_POR_NAO_REUSO = 10
RECOMPENSA_POR_USO_DE_MOVEL = 0


class SFC_AllocationEnv(gymnasium.Env):
    """
    Ambiente do Gymnasium para o problema de alocação de Service Function Chains (SFCs).
    
    Este ambiente simula a alocação de Virtual Network Functions (VNFs) de uma SFC
    em nós de uma infraestrutura de rede, considerando restrições de CPU, cache,
    latência e largura de banda.
    """

    # =================================================================================
    # 1. Métodos Principais da Interface do Gymnasium
    # =================================================================================

    def __init__(self,
                 valid_nodes: List[Union[int, str]],
                 list_graph: List[Graph],
                 list_sfc: List[SFC],
                 pesos_fatores: Dict[str, float] = None,
                 reward_config: Dict[str, float] = None,
                 is_training = True):
        """
        Inicializa o ambiente de alocação de SFC.
        """
        super().__init__()

        if len(list_graph) != len(list_sfc):
            raise ValueError("A lista de grafos deve ter o mesmo tamanho da lista de SFCs.")

        # --- Parâmetros de Configuração ---
        self.valid_nodes = valid_nodes
        self.list_graph = list_graph
        self.list_sfc = list_sfc
        self.pesos_fatores = pesos_fatores if pesos_fatores is not None else \
                             {"cpu": 1, "cache": 1, "lat": 1, "band": 3}
        
        self.is_training = is_training
        if self.is_training:
            self.initial_resource_snapshot=self._initialize_snapshots(self.list_graph)
        
        if reward_config is None:
            self.reward_config = {"success_bonus": 100.0, "failure_penalty": -100.0}
        else:
            self.reward_config = reward_config
        
        # --- Estado do Episódio ---
        self.graph: Graph = None
        self.current_sfc: SFC = None
        self.current_vnf: VNF = None
        self.current_location: Union[int, str] = None
        self.latency_request = None
        self.features = None
        self.ratio_cpu_used = 0
        
        
        self.cache_path = {}

        # --- Espaços de Ação e Observação ---
        num_nodes = len(valid_nodes)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict({ 
        #0 se cache e 1 se unique
        "tipo_sfc": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "recursos_nos_validos": spaces.Box(low=0, high=1, shape=(num_nodes, 6), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para o início de um novo episódio.
        Seleciona aleatoriamente um grafo e uma sfc de um caso real,
        restaura o estado inicial dos recursos e prepara a primeira SFC para alocação.
        """
        super().reset(seed=seed)
        self.latency_used = 0
        idx = 0

        if self.is_training:
            idx = np.random.randint(len(self.list_graph))
            self.graph = self.list_graph[idx]

            snapshot_nodes = self.initial_resource_snapshot[idx]['nodes']
            for node_id, initial_state in snapshot_nodes.items():
                if node_id in self.graph.nodes:
                    self.graph.nodes[node_id]['cpu_used'] = initial_state['cpu_used']
                    self.graph.nodes[node_id]['cache_used'] = initial_state['cache_used']

            snapshot_edges = self.initial_resource_snapshot[idx]['edges']
            for (u, v), initial_state in snapshot_edges.items():
                if self.graph.has_edge(u, v):
                    self.graph.edges[u, v]['bandwidth_used'] = initial_state['bandwidth_used']

        else:
            self.graph = self.list_graph[idx]
        sfc_sorteada = self.list_sfc[idx]
        
        self.set_current_sfc(sfc_sorteada)
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
    
        vnf = self.current_vnf
        bw_req = self.service_requirements[vnf.id]["out_bw"]
        current_node = self.current_location
        
        self.ratio_cpu_used = calcular_percentual_cpu_total(self.graph)
        
        print("VALOR DE CPU USADO NO CODIGO  ", self.ratio_cpu_used)
        self.features = self._get_nodes_features(vnf, bw_req, current_node)

        obs = self._get_obs()
        
        return obs, {}

    def step(self, action: int):
        """
        Executa um passo no ambiente a partir de uma ação do agente.
        A ação corresponde à escolha de um nó para alocar a VNF atual.
        """
        # 1. Traduzir a ação para um nó do grafo
        if action == len(self.valid_nodes) - 1:
            chosen_server = self.current_sfc.dst_node
        else: chosen_server = self.valid_nodes[action]

        vnf = self.current_vnf
        band_req = self.service_requirements[vnf.id]['out_bw']
        current_location = self.current_location
        path = get_available_shortest_path_fast(self.graph, current_location, chosen_server, band_req)
        total_cost = self.calculate_total_cost(self.current_sfc, vnf, chosen_server, band_req, path )
        
        # 2. Tentar alocar recursos (CPU/cache) no nó escolhido
        if not self.allocate_resources_on_node(chosen_server, self.current_vnf):
            return self._fail_step('resource')
        
        if not path or not self.allocate_bandwidth_along_path(path, band_req):
            return self._fail_step('bandwidth')

        self.servers_used.append(chosen_server)
        
        reward = -total_cost

         # ======================= ADICIONE ESTA VERIFICAÇÃO =======================
        if math.isnan(reward) or math.isinf(reward):
            print(f"--- DEBUG: Recompensa inválida detectada! Valor: {reward} ---")
            print(f"Custo total calculado: {total_cost}")
            assert not (math.isnan(reward) or math.isinf(reward))
        # =======================================================================


        # 6. Atualizar estado para o próximo passo
        self.current_location = chosen_server
        if not self.is_training:
            self.allocation_results[self.current_vnf.id] = {'allocated_server': chosen_server, 'path': path, 'cost': total_cost}

        self.latency_used += calculate_total_latency(self.graph, path, vnf)

        # 7. Verificar conclusão e avançar para a próxima VNF/SFC
        done = False
        if self.current_vnf == self.reverse_vnf_list[-1]:
            done = True
            self.success = True
            reward += self.reward_config['success_bonus']
            self.current_vnf = None

        else:
            idx=self.reverse_vnf_list.index(self.current_vnf)
            self.current_vnf = self.reverse_vnf_list[idx + 1]

        bw_required = self.service_requirements[self.current_vnf.id]['out_bw'] if self.current_vnf else 0
        self.features=self._get_nodes_features(self.current_vnf, bw_required, self.current_location)
        obs = self._get_obs()

        
        # 🚀 CORREÇÃO: Retorne todos os valores, incluindo o dicionário 'info'.
        return obs, reward, done, False, {}


    # =================================================================================
    # 2. Lógica Central da Simulação e Estado
    # =================================================================================





    def _get_nodes_features(self, vnf: VNF, bw_required, current_location: any) -> np.ndarray:
        """
        Calcula o vetor de features para cada nó candidato.
        Inclui lógicas especiais com a seguinte prioridade:
        1. Prioriza "nós dourados" (reuso com baixo custo), se existirem.
        2. Se não houver "nó dourado", aplica regras para VNF de cache ou "unique".
        """

        # Features: [0:cpu_used, 1:cache_used, 2:reusable, 3:N/A, 4:band_cost, 5:latency_cost, 6:is_invalid, 7:is_dst]
        num_valid_nodes = len(self.valid_nodes)
        features = np.zeros((num_valid_nodes, 8))

        if not vnf:
            # Se não houver VNF para alocar, retorna features zeradas, marcando todos como inválidos
            features[:, 6] = 1 
            return features

        # --- 1. Loop principal para calcular as features de cada nó (sem alterações) ---
        for i, node_id in enumerate(self.valid_nodes):
            # ... (código do loop, sem alterações) ...
            if i == num_valid_nodes - 1:
                node_id = self.current_sfc.dst_node
                features[i, 7] = 1
            node_data = self.graph.nodes[node_id]
            is_reusable = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
            features[i, 2] = float(is_reusable)
            cpu_req = vnf.get_cpu_request()
            cache_req = vnf.get_cache_request()
            if is_reusable:
                cpu_req, cache_req = 0, 0
            features[i, 0] = (node_data["cpu_used"] + cpu_req) / node_data["cpu_capacity"]
            features[i, 1] = (node_data["cache_used"] + cache_req) / node_data["cache_capacity"]
            if (node_data["cpu_used"] + cpu_req) >= node_data["cpu_capacity"] or \
               (node_data["cache_used"] + cache_req) >= node_data["cache_capacity"]:
                features[i, 6] = 1
            path = get_available_shortest_path_fast(self.graph, current_location, node_id, bw_required)
            if not path:
                features[i, 4] = 1.0
                features[i, 5] = 1.0
                features[i, 6] = 1
            else:
                bd_cost, latency_cost = self.calculate_bw_lat_cost(vnf, node_id, path, bw_required)
                features[i, 4] = bd_cost
                features[i, 5] = latency_cost

#         # --- 2. LÓGICA: Priorizar "Nós Dourados" ---
#         golden_nodes_mask = (
#             (features[:, 2] == 1) &      # É reutilizável
#             (features[:, 5] < 10) &       # Custo de latência é baixo
#             (features[:, 4] < 4)         # Custo de banda é baixo
#         )
#         # MODIFICAÇÃO: Guarda o resultado da checagem em uma variável
#         has_golden_node = np.any(golden_nodes_mask)

#         if has_golden_node:
#             # Se um nó dourado existe, ele se torna a única opção
#             nodes_to_invalidate_mask = ~golden_nodes_mask
#             features[nodes_to_invalidate_mask, 6] = 1

        # --- 3. Lógica especial para VNF de cache (sem alterações) ---
        first_vnf = self.current_sfc.get_previous_vnf(self.current_sfc.get_dst_vnf())
        second_vnf = first_vnf.get_previous_vnf() if first_vnf else None

        if first_vnf == self.current_vnf or (second_vnf and second_vnf == self.current_vnf):
            if "cache" in self.current_sfc.id and not features[-1, 6]:
                features[:-1, 6] = 1

        # --- 4. LÓGICA MODIFICADA: Regra para a primeira VNF "unique" ---
        is_1_or_2_vnf = first_vnf == self.current_vnf or second_vnf == self.current_vnf
        unique_in_id =  "unique" in self.current_sfc.id
        valid_node = not features[-1, 6]
        aceitable_latency = bool(features[-1, 5] < 2.58)

        # MODIFICAÇÃO: Adicionada a condição "not has_golden_node"
        # Esta regra só é ativada se um nó dourado NÃO foi encontrado na etapa 2
        if (self.ratio_cpu_used >40 and
            is_1_or_2_vnf and
            unique_in_id and
            valid_node ):

            # Invalida todos os outros nós, forçando a escolha do destino.
            features[:-1, 6] = 1

        return features


    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Monta a observação do ambiente de forma estruturada e eficiente usando NumPy.
        """
        valid_nodes = self.valid_nodes
        num_valid_nodes = len(valid_nodes)

        # --- 1. Determinação do Último Nó Escolhido ---
        primeira_sf = np.zeros(1, dtype=np.float32)
        current_loc = self.current_location if not isinstance(self.current_location, str) else 'M'
        
        # O destino é tratado como o último índice
        idx_loc = valid_nodes.index(current_loc) if current_loc != 'M' else num_valid_nodes - 1
        primeira_sf[0] = 1.0 if self.current_sfc.get_previous_vnf(self.current_sfc.get_dst_vnf()) == self.current_vnf else 0.0

        # --- 2. Coleta de Features dos Nós Válidos (Versão Otimizada) ---
        # self.features é um array NumPy com as colunas:
        # [0:cpu, 1:cache, 2:reusable, 3:path(não usado aqui), 4:band, 5:latency, 6:is_invalid]

        # Define as colunas que queremos selecionar do array self.features
        # para formar nossa observação.
        indices_das_features = [0, 1, 2, 4, 5, 7]

        # Usa o fatiamento avançado do NumPy para selecionar todas as linhas
        # e apenas as colunas desejadas de uma só vez.
        # Isso elimina a necessidade de um loop em Python, sendo muito mais rápido.
        recursos_nodes = self.features[:, indices_das_features].astype(np.float32)

        # Normaliza a coluna de latência (que agora é a coluna de índice 4 no novo array)
        # A operação é feita em toda a coluna de uma vez.
        recursos_nodes[:, 4] /= 30

        if not self.current_vnf or  "cache" in self.current_sfc.id:
            tipo_sfc = np.array([0.0], dtype=np.float32)
        else:
            tipo_sfc = tipo_sfc = np.array([1.0], dtype=np.float32)
        obs = {
            "tipo_sfc": tipo_sfc ,
            "recursos_nos_validos": recursos_nodes,
        }


        for key, value in obs.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                print(f"--- DEBUG: NaN ou Inf detectado na observação final (chave: {key})! ---")
                print(value)
                assert not (np.any(np.isnan(value)) or np.any(np.isinf(value)))

        return obs



    def action_masks(self) -> np.ndarray:
        """
        Cria uma máscara de ações válidas para a decisão atual de forma declarativa.

        Returns:
            np.ndarray: Um array binário (máscara) de ações válidas [1, 0, 0, 1, ...].
        """
        # Se não houver VNF para alocar, nenhuma ação é possível.
        
        if self.current_vnf is None:
            return np.zeros(len(self.valid_nodes), dtype=np.int8)
        
        if self.features is None:
            vnf=self.current_vnf
            bw_req = self.service_requirements[vnf.id]["out_bw"]
            current_node = self.current_location
            self.features = self._get_nodes_features(vnf, bw_req, current_node)
        
        mask = [
            1 if self.features[i, 6] == 0 else 0
            for i, _ in enumerate(self.valid_nodes)
        ]
        if np.nan in mask:
            epa = 1
        return np.array(mask, dtype=np.int8)

    
    def allocate_resources_on_node(self, node_id: Union[int, str], vnf: VNF) -> bool:
        """
        Aloca CPU e Cache em um nó, considerando o reuso de serviços.
        Retorna True se a alocação for bem-sucedida, False caso contrário.
        """
        node = self.graph.nodes[node_id]
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        
        # Se o serviço for reutilizável, o custo efetivo de recursos é zero
        can_reuse = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
        effective_cpu_req = 0 if can_reuse else cpu_req
        effective_cache_req = 0 if can_reuse else cache_req
        
        # Verifica se há capacidade disponível para a alocação
        if (node['cpu_used'] + effective_cpu_req > node['cpu_capacity']) or \
           (node['cache_used'] + effective_cache_req > node['cache_capacity']):
            return False

        # Aloca os recursos e atualiza os metadados do serviço
        node['cpu_used'] += effective_cpu_req
        node['cache_used'] += effective_cache_req
            
        return True

    def allocate_bandwidth_along_path(self, path: List, bandwidth_required: float) -> bool:
        """
        Aloca largura de banda ao longo de um caminho de forma atômica.
        Verifica todos os links primeiro e, se todos tiverem capacidade, aloca a banda.
        Retorna True em caso de sucesso, False caso contrário.
        """
        # 1. Verificar se todos os links no caminho têm capacidade suficiente
        for u, v in zip(path[:-1], path[1:]):
            edge = self.graph.edges[u, v]
            available_bw = edge.get('bandwidth_capacity', 0) - edge.get('bandwidth_used', 0)
            if available_bw < bandwidth_required + 1e-9: # Tolerância para ponto flutuante
                return False

        # 2. Se a verificação passou, alocar a banda em todos os links
        for u, v in zip(path[:-1], path[1:]):
            self.graph.edges[u, v]['bandwidth_used'] += bandwidth_required
        
        return True
    

    def _set_list_graph_sfcs(self, list_graph: List[Graph] ,list_sfc: List[SFC]):
        if len(list_graph) != len(list_sfc):
            raise Exception("O tamanho da lista de grafos deve ser igual ao de SFCs para correspondência")
        else: 
            self.list_graph = list_graph
            self.list_sfc = list_sfc
            self.reset()


    def _fail_step(self, reason: str):
        """
        Finaliza um episódio com falha, aplicando uma penalidade alta.
        """
        self.fail_reason = reason
        # print(f"Causa da falha: {reason}")
        self.success = False
        
        # --- USA A PENALIDADE CONFIGURADA ---
        reward = self.reward_config['failure_penalty']
        
        done = True
        # 🚀 CORREÇÃO: Garanta que mesmo em falha, a última observação e info sejam retornados.
        bw_required = self.service_requirements[self.current_vnf.id]['out_bw']
        self.features = self._get_nodes_features(self.current_vnf, bw_required, self.current_location)
        obs = self._get_obs()
        
        return obs, reward, done, False, {}
    
    def _initialize_snapshots(self, list_graph: List[Graph] = None):
        """
        Cria um snapshot do estado inicial dos recursos de todos os grafos
        para garantir um reset consistente dos episódios.
        """
        initial_resource_snapshot = {}
        for idx, graph in enumerate(list_graph):
            nodes = {n_id: {'cpu_used': data.get('cpu_used', 0),
                            'cache_used': data.get('cache_used', 0)
                            }
                     for n_id, data in graph.nodes(data=True)}
            edges = {(u, v): {'bandwidth_used': data.get('bandwidth_used', 0)}
                     for u, v, data in graph.edges(data=True)}
            initial_resource_snapshot[idx] = {'nodes': nodes, 'edges': edges}
        return initial_resource_snapshot
    
    def set_current_sfc(self, sfc: SFC):
        """Define a SFC atual para alocação e inicializa seus parâmetros."""
        if not sfc:
            raise ValueError("SFC não pode ser None.")
        self.current_sfc = sfc
        self.reverse_vnf_list = self.define_reverse_vnf_list(sfc)
        self.current_vnf = self.reverse_vnf_list[0]
        self.current_location = self.current_sfc.dst_node
        self.servers_used = []

        service_requirements = {} 
        services = []  # Lista para guardar os nomes
        sfs_dict = sfc.vnfs_dict
        
        for item in sfs_dict:
            nome = item['name']
            services.append(nome)  # Adiciona o nome à lista de nomes
            service_requirements[nome] = {
                'cpu': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw']}

        if True:
            services.append('dst')
            service_requirements['dst'] = {'cpu': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0}  
        
        self.service_requirements = service_requirements

    def define_reverse_vnf_list(self, sfc: SFC) -> List[VNF]:
        """Retorna a lista de VNFs da SFC em ordem reversa (do destino para a origem)."""
        vnf_list = []
        dst_vnf = sfc.get_dst_vnf()
        current_vnf = sfc.get_previous_vnf(dst_vnf)
        while True:
            vnf_list.append(current_vnf)
            if current_vnf.previous_vnf is None or current_vnf.previous_vnf.id == 'src':
                break
            current_vnf = sfc.get_previous_vnf(current_vnf)
        return vnf_list

    def is_reusable_at_node(self,sfc: SFC, graph: Graph, node_id: Union[int, str], vnf: VNF) -> bool:
        """Verifica se uma VNF compartilhável já está alocada em um nó."""
        if not vnf:
            return False
        service_name = vnf.id

        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()

        node = graph.nodes[node_id]

        cpu_used, cpu_cap = node["cpu_used"], node["cpu_capacity"]
        cache_used, cache_cap = node["cache_used"], node["cache_capacity"]

        if cpu_used+cpu_req >= cpu_cap or cache_used+cache_req >= cache_cap:
            return False

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False
        
        session_id = sfc.id.split("_")[-1]
        service_key = (service_name, session_id)
        result = service_key in graph.nodes[node_id].get('services', {})

        return result

    def calculate_bw_lat_cost(self, vnf: VNF, server_id, path: List, bw_required: float):
        # Latência computacional
        latency_cost = calculate_computational_latency(self.graph, server_id, vnf)

        if not path or len(path) < 2:
            return 0, latency_cost

        bw_cost = 0
        for u, v in zip(path[:-1], path[1:]):
            # Acesso à aresta da rede
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get('bandwidth_capacity', None)
            bd_used = edge.get('bandwidth_used', 0)

            # Calcula latência do enlace
            latency_cost += calculate_latency_betwen_nodes(self.graph, u, v, vnf)

            # Se a capacidade de banda for insuficiente, retorna custo infinito
            if bd_capacity is None or bd_capacity == 0 or bw_required + bd_used > bd_capacity:
                return float(999), latency_cost

            # Cálculo do custo de banda
            link_cost = (bw_required + bd_used) / bd_capacity
            bw_cost += link_cost

        return bw_cost, latency_cost

    
    def calculate_total_cost(self,sfc ,vnf: VNF, server_id, bw_required,path):
        """Calcula o custo total da alocação de um serviço."""

        node = self.graph.nodes[server_id]
        reusable = self.is_reusable_at_node(sfc, self.graph, server_id,vnf)
        cpu_capacity = node["cpu_capacity"] or 1
        cache_capacity = node["cache_capacity"] or 1
        vnf_id = vnf.id
        cpu_request = self.service_requirements[vnf_id]['cpu']
        cache_request = self.service_requirements[vnf_id]['cache']

        
        cpu_cost = ((node["cpu_used"] + cpu_request) / cpu_capacity) 
        cache_cost = ((node["cache_used"] + cache_request) / cache_capacity) 

        if not reusable:
            cpu_cost+= PUNICAO_POR_NAO_REUSO
            cache_cost+= PUNICAO_POR_NAO_REUSO

        mobile_device_cost = 0
        if server_id == self.current_sfc.dst_node:
            mobile_device_cost -= RECOMPENSA_POR_USO_DE_MOVEL
        
        bw_cost, lat_cost = self.calculate_bw_lat_cost(vnf, server_id, path, bw_required)
        
        resource_cost = cpu_cost * self.pesos_fatores['cpu'] + cache_cost * self.pesos_fatores['cache']

        band_cost=bw_cost * self.pesos_fatores['band']
        return resource_cost + band_cost + lat_cost * self.pesos_fatores['lat'] + mobile_device_cost
         
            
def calculate_total_latency(graph: Graph, path: List, vnf: VNF):
    """
    Calcula a latência total de um caminho dado e de uma VNF.
    
    A latência total é composta pela latência computacional no último nó 
    (onde a VNF é alocada) e pela latência de rede entre os nós ao longo do caminho.
    
    :param graph: O grafo que representa a rede, com informações sobre os links e servidores.
    :param path: Lista de nós representando o caminho de alocação do serviço.
    :param vnf: O VNF (função de rede virtual) que está sendo alocado.
    :return: A latência total (latência computacional + latência de rede).
    """
    total_latency = 0

    # Latência de rede (entre os nós do caminho)
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        # Cálculo da latência de rede entre os nós u e v
        edge_latency = calculate_latency_betwen_nodes(graph, u, v, vnf)
        total_latency += edge_latency

    # Latência computacional (apenas no último nó, onde a VNF é alocada)
    last_server = path[-1]  # Último nó do caminho
    comp_latency = calculate_computational_latency(graph, last_server, vnf)
    total_latency += comp_latency

    return total_latency

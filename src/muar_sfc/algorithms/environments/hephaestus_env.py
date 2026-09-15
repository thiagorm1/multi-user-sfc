
import gymnasium
import numpy as np
from gymnasium import spaces
from networkx import Graph

from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
    get_available_shortest_path_fast,
)
from muar_sfc.core.sfc import SFC, VNF

# ADICIONADO:


SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")
LATENCY_REQ = 13
MOBILE_DEVICE_USAGE_REWARD = 0


class SFC_AllocationEnv_hephaestus(gymnasium.Env):
    """
    Ambiente do Gymnasium para o problema de alocação de Service Function Chains (SFCs).

    Este ambiente simula a alocação de Virtual Network Functions (VNFs) de uma SFC
    em nós de uma infraestrutura de rede, considerando restrições de CPU, cache,
    latência e largura de banda.
    """

    # =================================================================================
    # 1. Métodos Principais da Interface do Gymnasium
    # =================================================================================

    def __init__(
        self,
        valid_nodes: list[int | str],
        list_graph: list[Graph],
        list_sfc: list[SFC],
        pesos_fatores: dict[str, float] = None,
        is_training=True,
    ):
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
        self.pesos_fatores = (
            pesos_fatores
            if pesos_fatores is not None
            else {
                "w_cost": 0.1,  # Prioridade baixa
                "w_latency": 0.3,  # Prioridade média
                "w_inequality": 0.3,  # Prioridade baixa
                "w_bandwidth": 0.1,  # Prioridade MÁXIMA E INEQUÍVOCA
            }
        )

        self.is_training = is_training
        if self.is_training:
            self.initial_resource_snapshot = self._initialize_snapshots(self.list_graph)

        # --- Estado do Episódio ---
        self.graph: Graph = None
        self.current_sfc: SFC = None
        self.current_vnf: VNF = None
        self.current_location: int | str = None
        self.latency_request = None
        self.features = None
        self.ratio_cpu_used = 0
        self.ratio_cache_used = 0

        self.cache_path = {}

        # --- Espaços de Ação e Observação ---
        num_nodes = len(valid_nodes)
        self.action_space = spaces.Discrete(num_nodes)

        # NOVA ESTRUTURA DO ESPAÇO DE OBSERVAÇÃO
        # Inspirado na Tabela 3 do artigo HephaestusForge
        num_app_features = 3  # 1. CPU req, 2. Cache req, 3. Latency req
        num_cluster_features = (
            6  # 1. CPU total, 2. Cache total, 3. CPU usado, 4. Cache usado, 5. Latência, 6. Banda
        )

        self.observation_space = spaces.Dict(
            {
                # Métricas da requisição atual (análogo ao "App" do artigo)
                "app_metrics": spaces.Box(
                    low=0, high=1, shape=(num_app_features,), dtype=np.float32
                ),
                # Métricas da infraestrutura (análogo ao "Cluster" do artigo)
                "cluster_metrics": spaces.Box(
                    low=0, high=1, shape=(num_nodes, num_cluster_features), dtype=np.float32
                ),
            }
        )

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para o início de um novo episódio.
        Seleciona aleatoriamente um grafo e uma sfc de um caso real,
        restaura o estado inicial dos recursos e prepara a primeira SFC para alocação.
        """
        super().reset(seed=seed)
        self.latency_used = 0
        self.total_bw_cost_episode = 0.0
        idx = 0

        if self.is_training:
            idx = np.random.randint(len(self.list_graph))
            self.graph = self.list_graph[idx]

            snapshot_nodes = self.initial_resource_snapshot[idx]["nodes"]
            for node_id, initial_state in snapshot_nodes.items():
                if node_id in self.graph.nodes:
                    self.graph.nodes[node_id]["cpu_used"] = initial_state["cpu_used"]
                    self.graph.nodes[node_id]["cache_used"] = initial_state["cache_used"]

            snapshot_edges = self.initial_resource_snapshot[idx]["edges"]
            for (u, v), initial_state in snapshot_edges.items():
                if self.graph.has_edge(u, v):
                    self.graph.edges[u, v]["bandwidth_used"] = initial_state["bandwidth_used"]

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

        self.features = self._get_nodes_features(vnf, bw_req, current_node)

        obs = self._get_obs()

        return obs, {}

    def step(self, action: int):
        """
        Executa um passo no ambiente com feedback imediato para todos os custos,
        incluindo a melhora no Coeficiente de Gini.
        """
        # 1. Traduzir a ação para um nó do grafo e obter requisitos
        if action == len(self.valid_nodes) - 1:
            chosen_server = self.current_sfc.dst_node
        else:
            chosen_server = self.valid_nodes[action]

        vnf = self.current_vnf
        band_req = self.service_requirements[vnf.id]["out_bw"]
        current_location = self.current_location
        path = get_available_shortest_path_fast(
            self.graph, current_location, chosen_server, band_req
        )

        # 2. Calcular os CUSTOS IMEDIATOS e o ESTADO DO GINI ANTES DA AÇÃO
        step_costs = {}
        gini_before = self._calculate_gini_coefficient()  # <<-- NOVO: Gini ANTES

        if path:
            bw_cost_step, latency_cost_step = self.calculate_bw_lat_cost(
                vnf, chosen_server, path, band_req
            )
            step_costs["bw_cost"] = bw_cost_step
            step_costs["latency_cost"] = latency_cost_step

            max_cpu_cap = max(
                d["cpu_capacity"] for _, d in self.graph.nodes(data=True) if d.get("cpu_capacity")
            )
            cpu_cap_chosen = self.graph.nodes[chosen_server]["cpu_capacity"]
            step_costs["deployment_cost"] = 1 + 9 * (cpu_cap_chosen / max_cpu_cap)
        else:
            # Custos máximos se não houver caminho levarão à falha
            step_costs["bw_cost"] = 10.0
            step_costs["latency_cost"] = self.current_sfc.get_latency_request()
            step_costs["deployment_cost"] = 10.0

        # 3. Tentar alocar recursos
        if not self.allocate_resources_on_node(chosen_server, self.current_vnf):
            return self._fail_step("resource")

        if not path or not self.allocate_bandwidth_along_path(path, band_req):
            # A alocação de banda falhar, precisamos reverter a alocação de CPU/Cache
            return self._fail_step("bandwidth")

        # 4. Ação bem-sucedida. CALCULAR GINI DEPOIS E A MELHORA
        gini_after = self._calculate_gini_coefficient()  # <<-- NOVO: Gini DEPOIS
        step_costs["gini_improvement"] = (
            gini_before - gini_after
        )  # <<-- NOVO: A "melhora" é a redução do Gini

        # 5. Calcular a RECOMPENSA IMEDIATA com base nos custos E na melhora do Gini
        reward = self._calculate_step_reward(step_costs)
        self.servers_used.append(chosen_server)

        # 6. Atualizar estado para o próximo passo
        self.current_location = chosen_server
        self.latency_used += step_costs.get("latency_cost", 0)

        # 7. Verificar conclusão
        done = False
        if not self.is_training:
            self.allocation_results[self.current_vnf.id] = {
                "allocated_server": chosen_server,
                "path": path,
                "cost": 0,
            }
        if self.current_vnf == self.reverse_vnf_list[-1]:
            done = True
            self.success = True
            if self.latency_used <= self.current_sfc.get_latency_request():
                reward += 10
            else:
                reward -= 10
            self.current_vnf = None
        else:
            idx = self.reverse_vnf_list.index(self.current_vnf)
            self.current_vnf = self.reverse_vnf_list[idx + 1]

        bw_required = (
            self.service_requirements[self.current_vnf.id]["out_bw"] if self.current_vnf else 0
        )
        self.features = self._get_nodes_features(
            self.current_vnf, bw_required, self.current_location
        )
        obs = self._get_obs()

        return obs, reward, done, False, {}

    # =================================================================================
    # 2. Lógica Central da Simulação e Estado
    # =================================================================================

    def _get_nodes_features(self, vnf: VNF, bw_required, current_location: any) -> np.ndarray:
        """
        Calcula o vetor de features para cada nó candidato.
        """

        # Features Mapping:
        # 0: cpu_utilization (normalized)
        # 1: cache_utilization (normalized)
        # 2: is_reusable (binary)
        # 3: bandwidth_cost (ratio)
        # 4: latency_cost (raw value)
        # 5: is_invalid (binary mask)

        num_valid_nodes = len(self.valid_nodes)
        features = np.zeros((num_valid_nodes, 6))

        if not vnf:
            features[:, 5] = 1
            return features

        for i, node_id in enumerate(self.valid_nodes):
            # Mapeia o último nó da lista como o nó de destino (Mobile/User)
            if i == num_valid_nodes - 1:
                node_id = self.current_sfc.dst_node

            # --- NOVA LÓGICA: BLOQUEAR MOBILE ---
            # Se o nó for o destino (dispositivo do usuário), marca como inválido
            if node_id == self.current_sfc.dst_node:
                features[i, 5] = 1.0  # Máscara de inválido
                continue  # Pula o resto dos cálculos para economizar processamento
            # ------------------------------------

            node_data = self.graph.nodes[node_id]
            is_reusable = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
            features[i, 2] = float(is_reusable)

            cpu_req = vnf.get_cpu_request()
            cache_req = vnf.get_cache_request()
            if is_reusable:
                cpu_req, cache_req = 0, 0

            # --- PROTEÇÃO CONTRA DIVISÃO POR ZERO (CPU) ---
            if node_data["cpu_capacity"] > 0:
                features[i, 0] = (node_data["cpu_used"] + cpu_req) / node_data["cpu_capacity"]
            else:
                features[i, 0] = 0.0
                features[i, 5] = 1.0  # Marca inválido imediatamente

            # --- PROTEÇÃO CONTRA DIVISÃO POR ZERO (CACHE) ---
            if node_data["cache_capacity"] > 0:
                features[i, 1] = (node_data["cache_used"] + cache_req) / node_data[
                    "cache_capacity"
                ]
            else:
                features[i, 1] = 0.0
                features[i, 5] = 1.0  # Marca inválido imediatamente

            # Verifica Sobrecarga (Apenas se o nó ainda for válido)
            if features[i, 5] == 0 and (
                (node_data["cpu_used"] + cpu_req) > node_data["cpu_capacity"] or
                (node_data["cache_used"] + cache_req) > node_data["cache_capacity"]
            ):
                features[i, 5] = 1.0

            # Verifica Caminho e Banda
            # (Só calcula se o nó ainda for considerado válido para economizar tempo)
            if features[i, 5] == 0:
                path = get_available_shortest_path_fast(
                    self.graph, current_location, node_id, bw_required
                )
                if not path:
                    features[i, 5] = 1.0
                    features[i, 3] = 1.0
                    features[i, 4] = 100.0
                else:
                    bd_cost, latency_cost = self.calculate_bw_lat_cost(
                        vnf, node_id, path, bw_required
                    )
                    features[i, 3] = bd_cost
                    features[i, 4] = latency_cost

                    if bd_cost >= 999:
                        features[i, 5] = 1.0

        return features

    # Em environment.py, SUBSTITUA a sua função _get_obs por esta

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        Monta a observação do ambiente seguindo a estrutura do trabalho HephaestusForge,
        separando métricas da aplicação e da infraestrutura.
        Todos os valores são normalizados para o intervalo [0, 1].
        """
        # --- 1. Preparar Métricas da Aplicação ("app_metrics") ---
        app_metrics = np.zeros(self.observation_space["app_metrics"].shape, dtype=np.float32)

        # Preenche com os requisitos da VNF atual, se houver
        max_latency_sfc = 24
        if self.current_vnf:
            # Normalização: dividir pelo máximo possível para manter no intervalo [0, 1]
            # Assumido valores máximos razoáveis. Ajuste se necessário.
            max_cpu_req = 100.0
            max_cache_req = 100.0

            cpu_req = self.current_vnf.get_cpu_request()
            cache_req = self.current_vnf.get_cache_request()
            latency_req = LATENCY_REQ

            app_metrics[0] = cpu_req / max_cpu_req
            app_metrics[1] = cache_req / max_cache_req
            app_metrics[2] = latency_req / max_latency_sfc

        # --- 2. Preparar Métricas dos Clusters ("cluster_metrics") ---
        len(self.valid_nodes)
        cluster_metrics = np.zeros(
            self.observation_space["cluster_metrics"].shape, dtype=np.float32
        )

        # Para normalizar a capacidade, encontramos o máximo na rede
        max_total_cpu = max(
            data["cpu_capacity"]
            for _, data in self.graph.nodes(data=True)
            if data.get("cpu_capacity")
        )
        max_total_cache = max(
            data["cache_capacity"]
            for _, data in self.graph.nodes(data=True)
            if data.get("cache_capacity")
        )
        if max_total_cpu == 0:
            max_total_cpu = 1
        if max_total_cache == 0:
            max_total_cache = 1

        for i, node_id in enumerate(self.valid_nodes):
            if node_id == "M":
                node_id = self.current_sfc.dst_node
            node_data = self.graph.nodes[node_id]

            # Capacidade Total (Π_cpu, Π_mem do artigo)
            total_cpu = node_data.get("cpu_capacity", 0)
            total_cache = node_data.get("cache_capacity", 0)

            # Recursos Alocados (Θ_cpu, Θ_mem do artigo)
            used_cpu = node_data.get("cpu_used", 0)
            used_cache = node_data.get("cache_used", 0)

            # Latência do nó (δc do artigo) - usamos o custo de latência já calculado
            node_latency = self.features[i, 4] if self.features is not None else 0
            node_bw_cost = self.features[i, 3] if self.features is not None else 0  # ADICIONADO

            # Normalização e preenchimento
            cluster_metrics[i, 0] = total_cpu / max_total_cpu
            cluster_metrics[i, 1] = total_cache / max_total_cache
            cluster_metrics[i, 2] = used_cpu / total_cpu if total_cpu > 0 else 0
            cluster_metrics[i, 3] = used_cache / total_cache if total_cache > 0 else 0
            cluster_metrics[i, 4] = min(node_latency / max_latency_sfc, 1.0)
            cluster_metrics[i, 5] = node_bw_cost  # ADICIONADO (o custo já é uma proporção)

        # --- 3. Montar e retornar a observação final ---
        obs = {"app_metrics": app_metrics, "cluster_metrics": cluster_metrics}

        # Verificação de segurança para evitar valores inválidos
        for key, value in obs.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                print(f"--- DEBUG: NaN ou Inf detectado na observação (chave: {key})! ---")
                raise AssertionError("Observação inválida gerada.")

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
            vnf = self.current_vnf
            bw_req = self.service_requirements[vnf.id]["out_bw"]
            current_node = self.current_location
            self.features = self._get_nodes_features(vnf, bw_req, current_node)

        mask = [1 if self.features[i, 5] == 0 else 0 for i, _ in enumerate(self.valid_nodes)]
        if np.nan in mask:
            pass
        return np.array(mask, dtype=np.int8)

    def allocate_resources_on_node(self, node_id: int | str, vnf: VNF) -> bool:
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
        if (node["cpu_used"] + effective_cpu_req > node["cpu_capacity"]) or (
            node["cache_used"] + effective_cache_req > node["cache_capacity"]
        ):
            return False

        # Aloca os recursos e atualiza os metadados do serviço
        node["cpu_used"] += effective_cpu_req
        node["cache_used"] += effective_cache_req

        return True

    def allocate_bandwidth_along_path(self, path: list, bandwidth_required: float) -> bool:
        """
        Aloca largura de banda ao longo de um caminho de forma atômica.
        Verifica todos os links primeiro e, se todos tiverem capacidade, aloca a banda.
        Retorna True em caso de sucesso, False caso contrário.
        """
        # 1. Verificar se todos os links no caminho têm capacidade suficiente
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges[u, v]
            available_bw = edge.get("bandwidth_capacity", 0) - edge.get("bandwidth_used", 0)
            if available_bw < bandwidth_required + 1e-9:  # Tolerância para ponto flutuante
                return False

        # 2. Se a verificação passou, alocar a banda em todos os links
        for u, v in zip(path[:-1], path[1:], strict=False):
            self.graph.edges[u, v]["bandwidth_used"] += bandwidth_required

        return True

    def _set_list_graph_sfcs(self, list_graph: list[Graph], list_sfc: list[SFC]):
        if len(list_graph) != len(list_sfc):
            raise Exception(
                "O tamanho da lista de grafos deve ser igual ao de SFCs para correspondência"
            )
        else:
            self.list_graph = list_graph
            self.list_sfc = list_sfc
            self.reset()
            # print("RESET EM environment no _set_list_graph_sfcs")

    def _fail_step(self, reason: str):
        """
        Finaliza um episódio com falha, aplicando uma penalidade alta.
        """
        self.fail_reason = reason
        # print(f"Causa da falha: {reason}")
        self.success = False

        # --- USA A PENALIDADE CONFIGURADA ---
        reward = -100

        done = True
        # 🚀 CORREÇÃO: Garanta que mesmo em falha, a última observação e info sejam retornados.
        bw_required = self.service_requirements[self.current_vnf.id]["out_bw"]
        self.features = self._get_nodes_features(
            self.current_vnf, bw_required, self.current_location
        )
        obs = self._get_obs()

        return obs, reward, done, False, {}

    def _initialize_snapshots(self, list_graph: list[Graph] = None):
        """
        Cria um snapshot do estado inicial dos recursos de todos os grafos
        para garantir um reset consistente dos episódios.
        """
        initial_resource_snapshot = {}
        for idx, graph in enumerate(list_graph):
            nodes = {
                n_id: {
                    "cpu_used": data.get("cpu_used", 0),
                    "cache_used": data.get("cache_used", 0),
                }
                for n_id, data in graph.nodes(data=True)
            }
            edges = {
                (u, v): {"bandwidth_used": data.get("bandwidth_used", 0)}
                for u, v, data in graph.edges(data=True)
            }
            initial_resource_snapshot[idx] = {"nodes": nodes, "edges": edges}
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
            nome = item["name"]
            services.append(nome)  # Adiciona o nome à lista de nomes
            service_requirements[nome] = {
                "cpu": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        if True:
            services.append("dst")
            service_requirements["dst"] = {"cpu": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

        self.service_requirements = service_requirements

    def define_reverse_vnf_list(self, sfc: SFC) -> list[VNF]:
        """Retorna a lista de VNFs da SFC em ordem reversa (do destino para a origem)."""
        vnf_list = []
        dst_vnf = sfc.get_dst_vnf()
        current_vnf = sfc.get_previous_vnf(dst_vnf)
        while True:
            vnf_list.append(current_vnf)
            if current_vnf.previous_vnf is None or current_vnf.previous_vnf.id == "src":
                break
            current_vnf = sfc.get_previous_vnf(current_vnf)
        return vnf_list

    def is_reusable_at_node(
        self, sfc: SFC, graph: Graph, node_id: int | str, vnf: VNF
    ) -> bool:
        """Verifica se uma VNF compartilhável já está alocada em um nó."""
        if not vnf:
            return False
        service_name = vnf.id

        cpu_req = vnf.get_cpu_request()
        vnf.get_cache_request()

        node = graph.nodes[node_id]

        cpu_used, cpu_cap = node["cpu_used"], node["cpu_capacity"]
        _cache_used, _cache_cap = node["cache_used"], node["cache_capacity"]

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False

        session_id = sfc.id.split("_")[-1]
        service_key = (service_name, session_id)
        result = service_key in graph.nodes[node_id].get("services", {})

        if result and cpu_used + cpu_req >= cpu_cap:
            return False

        return result

    def calculate_bw_lat_cost(self, vnf: VNF, server_id, path: list, bw_required: float):
        # Latência computacional
        latency_cost = calculate_computational_latency(self.graph, server_id, vnf)

        if not path or len(path) < 2:
            return 0, latency_cost

        bw_cost = 0
        for u, v in zip(path[:-1], path[1:], strict=False):
            # Acesso à aresta da rede
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get("bandwidth_capacity", None)
            bd_used = edge.get("bandwidth_used", 0)

            # Calcula latência do enlace
            latency_cost += calculate_latency_betwen_nodes(self.graph, u, v, vnf)

            # Se a capacidade de banda for insuficiente, retorna custo infinito
            if bd_capacity is None or bd_capacity == 0 or bw_required + bd_used > bd_capacity:
                return float(999), latency_cost

            # Cálculo do custo de banda
            link_cost = (bw_required + bd_used) / bd_capacity
            bw_cost += link_cost

        return bw_cost, latency_cost

    # Em environment.py, adicione este método à classe SFC_AllocationEnv
    def _calculate_deployment_cost(self) -> float:
        """
        Calcula o custo de implantação normalizado.
        CORRIGIDO: Proteção contra max_cpu_capacity igual a zero.
        """
        if not self.servers_used:
            return 1.0

        aux = []
        for n in self.valid_nodes:
            if n == "M":
                n = self.current_sfc.dst_node
            # Garante que lê 0 se a chave não existir
            aux.append(self.graph.nodes[n].get("cpu_capacity", 0))

        # Proteção se a lista estiver vazia ou só tiver zeros
        max_cpu_capacity = 1.0 if not aux else max(aux)

        # Evita divisão por zero se todos os nós tiverem capacidade 0
        if max_cpu_capacity == 0:
            max_cpu_capacity = 1.0

        total_cost = 0
        for server_id in self.servers_used:
            cpu_cap = self.graph.nodes[server_id].get("cpu_capacity", 0)

            # Custo proporcional à capacidade
            node_cost = 1 + 9 * (cpu_cap / max_cpu_capacity)
            total_cost += node_cost

        avg_cost = total_cost / len(self.servers_used)

        normalized_cost = avg_cost / 10.0

        return min(normalized_cost, 1.0)

    # Em environment.py, adicione este método à classe SFC_AllocationEnv
    def _calculate_latency_metric(self) -> float:
        """
        Calcula a métrica de latência normalizada.
        """
        latency_request = self.current_sfc.get_latency_request()
        if not latency_request or latency_request == 0:
            return 1.0  # Custo máximo se não houver requisição

        # A métrica é a latência usada como uma fração da latência permitida
        normalized_latency = self.latency_used / latency_request

        return min(normalized_latency, 1.0)  # Garante que não passe de 1.0

    # Em environment.py, adicione este método à classe SFC_AllocationEnv
    def _calculate_gini_coefficient(self) -> float:
        """
        Calcula o Coeficiente de Gini para a utilização de CPU nos nós.
        Um valor de 0 significa igualdade perfeita, e 1 significa desigualdade perfeita.
        Baseado na fórmula do artigo[cite: 383, 385].
        """
        # Pega a utilização de CPU de todos os nós válidos
        aux = []
        for n in self.valid_nodes:
            if n == "M":
                n = self.current_sfc.dst_node
            aux.append(self.graph.nodes[n]["cpu_used"])
        cpu_usages = np.array(aux)

        # O Coeficiente de Gini só é significativo se houver alguma carga
        if np.sum(cpu_usages) == 0:
            return 0.0  # Igualdade perfeita se não há carga

        # Ordena os valores para o cálculo
        sorted_usages = np.sort(cpu_usages)
        n = len(sorted_usages)
        cum_usages = np.cumsum(sorted_usages, dtype=float)

        # Fórmula de Gini: (Sum( (i+1) * x_i ) / Sum(x_i)) * (2/n) - (n+1)/n
        # Simplificação comum é baseada na área sob a curva de Lorenz
        lorenz_curve_area = cum_usages.sum() / cum_usages[-1]
        gini = (n + 1 - 2 * lorenz_curve_area) / n

        return gini

    def _calculate_bandwidth_metric(self) -> float:
        """
        Calcula a métrica de custo de banda normalizada.
        Um valor mais alto significa um custo maior (pior).
        """
        if not self.servers_used:
            return 1.0  # Custo máximo se nenhuma VNF foi alocada

        # O custo de banda por passo já é uma proporção (uso/capacidade).
        # A média desses custos nos dá uma métrica razoável para o episódio.
        average_bw_cost = self.total_bw_cost_episode / len(self.servers_used)

        # Garante que o valor fique entre 0 e 1
        return min(average_bw_cost, 1.0)

    # Em environment.py, adicione este método à classe SFC_AllocationEnv

    # Em hephaestus_env.py, método _calculate_hephaestus_reward()
    # Em hephaestus_env.py
    # SUBSTITUA a função _calculate_step_reward pela versão abaixo

    def _calculate_step_reward(self, step_costs: dict[str, float]) -> float:
        """
        Calcula a recompensa para um único passo, incluindo a melhora no Gini.
        """
        # 1. Obter os pesos da configuração
        w_cost = self.pesos_fatores.get("w_cost", 0.25)
        w_latency = self.pesos_fatores.get("w_latency", 0.25)
        w_inequality = self.pesos_fatores.get("w_inequality", 0.25)
        w_bandwidth = self.pesos_fatores.get("w_bandwidth", 0.25)

        # 2. Calcular componentes da recompensa baseados em CUSTO (onde 1 - custo é bom)
        cost_metric = min(step_costs.get("deployment_cost", 0) / 10.0, 1.0)
        latency_metric = min(
            step_costs.get("latency_cost", 0) / self.current_sfc.get_latency_request(), 1.0
        )
        bandwidth_metric = min(step_costs.get("bw_cost", 0), 1.0)

        r_cost = 1.0 - cost_metric
        r_latency = 1.0 - latency_metric
        r_bandwidth = 1.0 - bandwidth_metric

        # 3. O componente de recompensa do GINI é a MELHORA DIRETA
        # Se gini_improvement > 0, a rede ficou mais igualitária (recompensa)
        # Se gini_improvement < 0, a rede ficou mais desigual (penalidade)
        r_inequality = step_costs.get("gini_improvement", 0)

        # 4. Calcular recompensa final ponderada para o PASSO
        step_reward = (
            w_cost * r_cost
            + w_latency * r_latency
            + w_bandwidth * r_bandwidth
            + w_inequality * r_inequality
        )  # A melhora do Gini é somada diretamente

        return step_reward

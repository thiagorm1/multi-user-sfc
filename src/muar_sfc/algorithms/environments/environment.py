import math

import gymnasium
import numpy as np
from gymnasium import spaces
from networkx import Graph

from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
    get_available_shortest_path_fast,
)

# Módulos Locais
from muar_sfc.core.sfc import SFC, VNF
from muar_sfc.utils.network_utils import (
    calcular_percentual_banda_total,
    calcular_percentual_cache_total,
    get_graph_processing_utilization_simplified,
)

# --- Constantes Globais ---
SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")
NON_REUSABLE_PENALTY = 4


class SFC_AllocationEnv(gymnasium.Env):
    """
    Ambiente do Gymnasium para o problema de alocação de Service Function Chains (SFCs).

    Este ambiente simula a alocação de Virtual Network Functions (VNFs) de uma SFC
    em nós de uma infraestrutura de rede, considerando restrições de CPU, cache,
    latência e largura de banda.
    """

    # =================================================================================
    # 1. Inicialização e Setup
    # =================================================================================

    def __init__(
        self,
        valid_nodes: list[int | str],
        list_graph: list[Graph],
        list_sfc: list[SFC],
        pesos_fatores: dict[str, float] = None,
        reward_config: dict[str, float] = None,
        is_training: bool = True,
    ):
        """
        Inicializa o ambiente de alocação de SFC.
        """
        super().__init__()

        if len(list_graph) != len(list_sfc):
            raise ValueError("A lista de grafos deve ter o mesmo tamanho da lista de SFCs.")

        # --- Configurações Básicas ---
        self.valid_nodes = valid_nodes
        self.list_graph = list_graph
        self.list_sfc = list_sfc
        self.is_training = is_training

        # --- Configuração de Pesos e Recompensas ---
        self.pesos_fatores = (
            pesos_fatores
            if pesos_fatores is not None
            else {"cpu": 1, "cache": 1, "lat": 5, "band": 5, "mobile": 0}
        )

        self.reward_config = (
            reward_config
            if reward_config is not None
            else {"success_bonus": 100.0, "failure_penalty": -100.0}
        )

        # --- Inicialização de Snapshots (Training) ---
        if self.is_training:
            self.initial_resource_snapshot = self._initialize_snapshots(self.list_graph)

        # --- Estado do Episódio ---
        self.graph: Graph | None = None
        self.current_sfc: SFC | None = None
        self.current_vnf: VNF | None = None
        self.current_location: int | str = None
        self.latency_request = None
        self.features = None
        self.forbidden_nodes = []

        # Métricas de uso
        self.ratio_cpu_used = 0
        self.ratio_cache_used = 0
        self.ratio_banda_used = 0
        self.latency_used = 0

        self.cache_path = {}
        self.servers_used = []
        self.allocation_results = {}
        self.success = False
        self.fail_reason = None

        # --- Definição dos Espaços (Gym) ---
        num_nodes = len(valid_nodes)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict(
            {
                # 0 se cache e 1 se unique
                "tipo_sfc": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "usos_rede": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
                "recursos_nos_validos": spaces.Box(
                    low=0, high=1, shape=(num_nodes, 6), dtype=np.float32
                ),
            }
        )

    # =================================================================================
    # 2. Interface Principal do Gymnasium (Reset, Step, Mask)
    # =================================================================================

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para o início de um novo episódio.
        """
        super().reset(seed=seed)
        self.latency_used = 0
        idx = 0

        # Seleção do Cenário (Grafo e SFC)
        if self.is_training:
            idx = np.random.randint(len(self.list_graph))
            self.graph = self.list_graph[idx]
            self._restore_graph_resources(idx)
        else:
            self.graph = self.list_graph[idx]

        sfc_sorteada = self.list_sfc[idx]
        self.set_current_sfc(sfc_sorteada)

        # Reset de Variáveis de Controle
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}

        vnf = self.current_vnf
        bw_req = self.service_requirements[vnf.id]["out_bw"]
        current_node = self.current_location

        # Atualização de métricas de rede
        self.ratio_cpu_used = get_graph_processing_utilization_simplified(self.graph)
        self.ratio_cache_used = calcular_percentual_cache_total(self.graph)
        self.ratio_banda_used = calcular_percentual_banda_total(self.graph)

        self.features = self._get_nodes_features(vnf, bw_req, current_node)
        obs = self._get_obs()

        return obs, {}

    def step(self, action: int):
        """
        Executa um passo no ambiente a partir de uma ação do agente.
        """
        # 1. Traduzir a ação para um nó do grafo
        if action == len(self.valid_nodes) - 1:
            chosen_server = self.current_sfc.dst_node
        else:
            chosen_server = self.valid_nodes[action]

        vnf = self.current_vnf
        band_req = self.service_requirements[vnf.id]["out_bw"]
        current_location = self.current_location

        # Calcular caminho e métricas
        path = get_available_shortest_path_fast(
            self.graph, current_location, chosen_server, band_req
        )
        self.ratio_cpu_used = get_graph_processing_utilization_simplified(self.graph)
        self.ratio_cache_used = calcular_percentual_cache_total(self.graph)
        self.ratio_banda_used = calcular_percentual_banda_total(self.graph)

        total_cost = self._compute_allocation_cost(vnf, chosen_server, path, band_req)

        # 2. Tentar alocar recursos (CPU/cache) no nó escolhido
        if not self.allocate_resources_on_node(chosen_server, self.current_vnf):
            return self._fail_step("resource")

        # 3. Tentar alocar banda no caminho
        if not path or not self.allocate_bandwidth_along_path(path, band_req):
            return self._fail_step("bandwidth")

        self.servers_used.append(chosen_server)

        # 4. Calcular Recompensa
        reward = -total_cost
        if math.isnan(reward) or math.isinf(reward):
            print(f"--- DEBUG: Recompensa inválida detectada! Valor: {reward} ---")
            print(f"Custo total calculado: {total_cost}")
            assert not (math.isnan(reward) or math.isinf(reward))

        # 5. Atualizar estado para o próximo passo
        self.current_location = chosen_server
        if not self.is_training:
            self.allocation_results[self.current_vnf.id] = {
                "allocated_server": chosen_server,
                "path": path,
                "cost": total_cost,
            }

        self.latency_used += calculate_total_latency(self.graph, path, vnf)

        # 6. Verificar conclusão e avançar para a próxima VNF/SFC
        done = False
        if self.current_vnf == self.reverse_vnf_list[-1]:
            done = True
            self.success = True
            reward += self.reward_config["success_bonus"]
            self.current_vnf = None
        else:
            idx = self.reverse_vnf_list.index(self.current_vnf)
            self.current_vnf = self.reverse_vnf_list[idx + 1]

        # 7. Preparar nova observação
        bw_required = (
            self.service_requirements[self.current_vnf.id]["out_bw"] if self.current_vnf else 0
        )
        self.features = self._get_nodes_features(
            self.current_vnf, bw_required, self.current_location
        )
        obs = self._get_obs()

        return obs, reward, done, False, {}

    def action_masks(self) -> np.ndarray:
        """
        Cria uma máscara de ações válidas para a decisão atual.
        """
        if self.current_vnf is None:
            return np.zeros(len(self.valid_nodes), dtype=np.int8)

        if self.features is None:
            vnf = self.current_vnf
            bw_req = self.service_requirements[vnf.id]["out_bw"]
            current_node = self.current_location
            self.features = self._get_nodes_features(vnf, bw_req, current_node)

        mask = []
        for i, node_id in enumerate(self.valid_nodes):
            # 1. Verificação básica: Validade de recurso
            is_valid_resource = 1 if self.features[i, 6] == 0 else 0

            # 2. Verificação de redundância: Nós proibidos
            is_allowed_node = 1 if node_id not in self.forbidden_nodes else 0

            mask.append(is_valid_resource * is_allowed_node)

        return np.array(mask, dtype=np.int8)

    # =================================================================================
    # 3. Observações e Features
    # =================================================================================

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        Monta a observação do ambiente de forma estruturada e eficiente usando NumPy.
        """

        # --- 1. Determinação do Último Nó Escolhido ---
        # (Lógica original: 'primeira_sf' e 'current_loc' parecem não ser
        # utilizados diretamente na obs final, mas mantidos)
        primeira_sf = np.zeros(1, dtype=np.float32)
        self.current_location if not isinstance(self.current_location, str) else "M"

        # O destino é tratado como o último índice
        primeira_sf[0] = (
            1.0
            if self.current_sfc.get_previous_vnf(self.current_sfc.get_dst_vnf())
            == self.current_vnf
            else 0.0
        )

        # --- 2. Seleção de Features dos Nós ---
        # Indices: [0:cpu, 1:cache, 2:reusable, 4:band_cost, 5:lat_cost, 7:is_dst]
        indices_das_features = [0, 1, 2, 4, 5, 7]
        recursos_nodes = self.features[:, indices_das_features].astype(np.float32)

        # Normaliza a coluna de latência (índice 4 no novo array, original índice 5)
        recursos_nodes[:, 4] /= 100

        # --- 3. Métricas Globais da Rede ---
        usos_rede = np.array(
            [
                calcular_percentual_banda_total(self.graph),
                get_graph_processing_utilization_simplified(self.graph),
                self.latency_used / 100,
            ],
            dtype=np.float32,
        )

        # --- 4. Tipo da SFC ---
        if not self.current_vnf or "cache" in self.current_sfc.id:
            tipo_sfc = np.array([0.0], dtype=np.float32)
        else:
            tipo_sfc = np.array([1.0], dtype=np.float32)

        obs = {
            "tipo_sfc": tipo_sfc,
            "usos_rede": usos_rede,
            "recursos_nos_validos": recursos_nodes,
        }

        # Verificação de integridade
        for key, value in obs.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                print(f"--- DEBUG: NaN ou Inf detectado na observação final (chave: {key})! ---")
                print(value)
                assert not (np.any(np.isnan(value)) or np.any(np.isinf(value)))

        return obs

    def _get_nodes_features(self, vnf: VNF, bw_required, current_location: any) -> np.ndarray:
        """
        Calcula o vetor de features para cada nó candidato.
        """
        # Features mapping:
        # [0:cpu_used, 1:cache_used, 2:reusable, 3:N/A,
        #  4:band_cost, 5:latency_cost, 6:is_invalid, 7:is_dst]
        num_valid_nodes = len(self.valid_nodes)
        features = np.zeros((num_valid_nodes, 8))

        if not vnf:
            features[:, 6] = 1
            return features

        # --- 1. Loop principal: Features básicas de cada nó ---
        for i, node_id in enumerate(self.valid_nodes):
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

            features[i, 0] = (
                (node_data["cpu_used"] + cpu_req) / node_data["cpu_capacity"]
                if node_data["cpu_capacity"] > 0
                else 0
            )
            features[i, 1] = (
                (node_data["cache_used"] + cache_req) / node_data["cache_capacity"]
                if node_data["cache_capacity"] > 0
                else 0
            )

            # Validação de Capacidade
            if (node_data["cpu_used"] + cpu_req) >= node_data["cpu_capacity"] or (
                node_data["cache_used"] + cache_req
            ) >= node_data["cache_capacity"]:
                features[i, 6] = 1

            # Validação de Caminho e Banda
            path = get_available_shortest_path_fast(
                self.graph, current_location, node_id, bw_required
            )
            if not path:
                features[i, 4] = 1.0
                features[i, 5] = 1.0
                features[i, 6] = 1
            else:
                bd_cost, latency_cost = self.calculate_bw_lat_cost(vnf, node_id, path, bw_required)
                features[i, 4] = bd_cost
                features[i, 5] = latency_cost

        # --- 2. Lógica Específica para VNFs "Unique" e "Cache" ---
        first_vnf = self.current_sfc.get_previous_vnf(self.current_sfc.get_dst_vnf())
        second_vnf = first_vnf.get_previous_vnf() if first_vnf else None

        is_1_vnf = first_vnf == self.current_vnf
        is_2_vnf = second_vnf == self.current_vnf
        unique_in_id = "unique" in self.current_sfc.id
        cache_in_id = "cache" in self.current_sfc.id
        valid_node = not features[-1, 6]

        # Regra para Cache com alta utilização de CPU
        if (is_1_vnf or is_2_vnf) and valid_node and cache_in_id and self.ratio_cpu_used > 50:
            u = self.current_sfc.dst_node
            v = self.current_sfc.closer_router
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get("bandwidth_capacity", None)
            if bd_capacity / bw_required > 4:
                features[:-1, 6] = 1

        # Regra para Unique com altíssima utilização de CPU
        if (is_1_vnf) and valid_node and unique_in_id and self.ratio_cpu_used >= 60:
            u = self.current_sfc.dst_node
            v = self.current_sfc.closer_router
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get("bandwidth_capacity", None)
            if bd_capacity / bw_required > 4:
                features[:-1, 6] = 1

        if not (is_1_vnf or is_2_vnf):
            features[-1, 6] = 1

        return features

    # =================================================================================
    # 4. Gerenciamento de Recursos (Allocations & Checks)
    # =================================================================================

    def allocate_resources_on_node(self, node_id: int | str, vnf: VNF) -> bool:
        """
        Aloca CPU e Cache em um nó, considerando o reuso de serviços.
        """
        node = self.graph.nodes[node_id]
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()

        # Se o serviço for reutilizável, o custo efetivo de recursos é zero
        can_reuse = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
        effective_cpu_req = 0 if can_reuse else cpu_req
        effective_cache_req = 0 if can_reuse else cache_req

        # Verifica capacidade disponível
        if (node["cpu_used"] + effective_cpu_req > node["cpu_capacity"]) or (
            node["cache_used"] + effective_cache_req > node["cache_capacity"]
        ):
            if not self.is_training:
                print(f"Não recurso o suficiente no nó {node_id}")
            return False

        # Efetiva a alocação
        node["cpu_used"] += effective_cpu_req
        node["cache_used"] += effective_cache_req

        return True

    def allocate_bandwidth_along_path(self, path: list, bandwidth_required: float) -> bool:
        """
        Aloca largura de banda ao longo de um caminho de forma atômica.
        """
        # 1. Verificar se todos os links no caminho têm capacidade suficiente
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges[u, v]
            available_bw = edge.get("bandwidth_capacity", 0) - edge.get("bandwidth_used", 0)
            if available_bw < bandwidth_required:
                if not self.is_training:
                    print(f"Não houve banda o suficiente no link {u} e {v}")
                return False

        # 2. Se a verificação passou, alocar a banda em todos os links
        for u, v in zip(path[:-1], path[1:], strict=False):
            self.graph.edges[u, v]["bandwidth_used"] += bandwidth_required

        return True

    def is_reusable_at_node(
        self, sfc: SFC, graph: Graph, node_id: int | str, vnf: VNF
    ) -> bool:
        """
        Verifica se uma VNF compartilhável já está alocada em um nó.
        """
        if not vnf:
            return False
        service_name = vnf.id

        cpu_req = vnf.get_cpu_request()

        node = graph.nodes[node_id]
        cpu_used, cpu_cap = node["cpu_used"], node["cpu_capacity"]

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False

        session_id = sfc.id.split("_")[-1]
        service_key = (service_name, session_id)
        result = service_key in graph.nodes[node_id].get("services", {})

        if result and cpu_used + cpu_req >= cpu_cap:
            return False

        return result

    # =================================================================================
    # 5. Cálculo de Custos e Recompensas
    # =================================================================================

    def _compute_allocation_cost(
        self, vnf: VNF, server_id: int | str, path: list[int | str], bw_required: float
    ) -> float:
        """
        Calcula o custo total da alocação (base para a recompensa).
        """
        node_data = self.graph.nodes[server_id]
        factor_weights = self.pesos_fatores

        # --- 1. Custo de Recursos do Nó (CPU & Cache) ---
        is_reusable = self.is_reusable_at_node(self.current_sfc, self.graph, server_id, vnf)

        cpu_capacity = node_data.get("cpu_capacity", 1.0) or 1.0
        cache_capacity = node_data.get("cache_capacity", 1.0) or 1.0

        cpu_request = vnf.get_cpu_request()
        cache_request = vnf.get_cache_request()

        # Custo base de utilização
        cpu_cost = (node_data["cpu_used"] + cpu_request) / cpu_capacity
        cache_cost = (node_data["cache_used"] + cache_request) / cache_capacity

        # Penalidade se não reutilizar
        if not is_reusable:
            cpu_cost += NON_REUSABLE_PENALTY
            cache_cost += NON_REUSABLE_PENALTY

        weighted_node_cost = (
            cpu_cost * factor_weights["cpu"] + cache_cost * factor_weights["cache"]
        )

        # --- 2. Custo de Rede (Banda & Latência) ---
        bw_cost, lat_cost = self.calculate_bw_lat_cost(vnf, server_id, path, bw_required)

        base_band_weight = factor_weights["band"]
        weighted_network_cost = bw_cost * base_band_weight + lat_cost * factor_weights["lat"]

        # --- 3. Incentivos Estratégicos ---
        incentive_cost = 0.0
        if server_id == self.current_sfc.dst_node:
            network_stress_ratio = max(self.ratio_cpu_used, self.ratio_cache_used) / 100.0
            stress_factor = network_stress_ratio * factor_weights["mobile"]
            dynamic_reward = stress_factor**3
            incentive_cost = -dynamic_reward

        # --- 4. Custo Total ---
        total_cost = weighted_node_cost + weighted_network_cost + incentive_cost

        return total_cost

    def calculate_bw_lat_cost(self, vnf: VNF, server_id, path: list, bw_required: float):
        """
        Calcula custo de banda e latência para um caminho específico.
        """
        latency_cost = calculate_computational_latency(self.graph, server_id, vnf)

        if not path or len(path) < 2:
            return 0, latency_cost

        bw_cost = 0
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get("bandwidth_capacity", None)
            bd_used = edge.get("bandwidth_used", 0)

            latency_cost += calculate_latency_betwen_nodes(self.graph, u, v, vnf)

            # Retorna custo infinito se não houver banda
            if bd_capacity is None or bd_capacity == 0 or bw_required + bd_used > bd_capacity:
                return float(999), latency_cost

            # Custo do Link
            projected_usage_ratio = (bw_required + bd_used) / bd_capacity
            epsilon = 1e-6
            link_cost = 1.0 / (1.0 - projected_usage_ratio + epsilon)
            bw_cost += link_cost

        return bw_cost, latency_cost

    def _fail_step(self, reason: str):
        """
        Finaliza um episódio com falha e penalidade.
        """
        self.fail_reason = reason
        self.success = False

        reward = self.reward_config["failure_penalty"]
        done = True

        bw_required = self.service_requirements[self.current_vnf.id]["out_bw"]
        self.features = self._get_nodes_features(
            self.current_vnf, bw_required, self.current_location
        )
        obs = self._get_obs()

        return obs, reward, done, False, {}

    # =================================================================================
    # 6. Helpers de Configuração (SFC e Snapshots)
    # =================================================================================

    def set_forbidden_nodes(self, nodes: list[str | int]):
        self.forbidden_nodes = nodes

    def set_current_sfc(self, sfc: SFC):
        """
        Define a SFC atual para alocação e inicializa seus parâmetros.
        """
        if not sfc:
            raise ValueError("SFC não pode ser None.")
        self.current_sfc = sfc
        self.reverse_vnf_list = self.define_reverse_vnf_list(sfc)
        self.current_vnf = self.reverse_vnf_list[0]
        self.current_location = self.current_sfc.dst_node
        self.servers_used = []

        service_requirements = {}
        services = []
        sfs_dict = sfc.vnfs_dict

        for item in sfs_dict:
            nome = item["name"]
            services.append(nome)
            service_requirements[nome] = {
                "cpu": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        # Adiciona dummy requirements para o nó destino
        services.append("dst")
        service_requirements["dst"] = {"cpu": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

        self.service_requirements = service_requirements

    def define_reverse_vnf_list(self, sfc: SFC) -> list[VNF]:
        """
        Retorna a lista de VNFs da SFC em ordem reversa.
        """
        vnf_list = []
        dst_vnf = sfc.get_dst_vnf()
        current_vnf = sfc.get_previous_vnf(dst_vnf)
        while True:
            vnf_list.append(current_vnf)
            if current_vnf.previous_vnf is None or current_vnf.previous_vnf.id == "src":
                break
            current_vnf = sfc.get_previous_vnf(current_vnf)
        return vnf_list

    def _initialize_snapshots(self, list_graph: list[Graph] = None):
        """
        Cria snapshots iniciais dos recursos para reset eficiente.
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

    def _restore_graph_resources(self, idx):
        """
        Restaura os recursos do grafo usando o snapshot.
        """
        snapshot_nodes = self.initial_resource_snapshot[idx]["nodes"]
        for node_id, initial_state in snapshot_nodes.items():
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id]["cpu_used"] = initial_state["cpu_used"]
                self.graph.nodes[node_id]["cache_used"] = initial_state["cache_used"]

        snapshot_edges = self.initial_resource_snapshot[idx]["edges"]
        for (u, v), initial_state in snapshot_edges.items():
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v]["bandwidth_used"] = initial_state["bandwidth_used"]

    def _set_list_graph_sfcs(self, list_graph: list[Graph], list_sfc: list[SFC]):
        if len(list_graph) != len(list_sfc):
            raise Exception(
                "O tamanho da lista de grafos deve ser igual ao de SFCs para correspondência"
            )
        else:
            self.list_graph = list_graph
            self.list_sfc = list_sfc
            self.reset()


# =================================================================================
# Funções Auxiliares (Standalone)
# =================================================================================


def calculate_total_latency(graph: Graph, path: list, vnf: VNF):
    """
    Calcula a latência total (rede + computacional) de um caminho dado e de uma VNF.
    """
    total_latency = 0

    # Latência de rede
    edge_latency = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_latency += calculate_latency_betwen_nodes(graph, u, v, vnf)
    total_latency += edge_latency

    # Latência computacional (apenas no último nó)
    last_server = path[-1]
    comp_latency = calculate_computational_latency(graph, last_server, vnf)
    total_latency += comp_latency

    return total_latency

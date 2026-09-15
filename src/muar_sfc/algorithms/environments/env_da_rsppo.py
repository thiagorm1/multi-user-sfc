import gymnasium
import numpy as np
from gymnasium import spaces
from networkx import Graph

from muar_sfc.algorithms.environments.env_utils.utils import (
    calculate_comunication_latency,
    calculate_total_latency,
    create_route_info_from_allocation_results,
)
from muar_sfc.algorithms.networkUtils import (
    calculate_latency_betwen_nodes,
    get_available_shortest_path_fast,
)
from muar_sfc.core.sfc import SFC, VNF
from muar_sfc.utils.network_utils import get_sfc_latency_from_route

SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")


class SFC_AllocationEnv_DARSPPO(gymnasium.Env):
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
        super().__init__()

        if len(list_graph) != len(list_sfc):
            raise ValueError("A lista de grafos deve ter o mesmo tamanho da lista de SFCs.")

        # --- Parâmetros de Configuração ---
        self.valid_nodes = valid_nodes
        self.list_graph = list_graph
        self.list_sfc = list_sfc
        self.pesos_fatores = pesos_fatores or {"eta1": 1, "eta2": 1, "bw": 1}
        self.is_training = is_training

        if self.is_training:
            self.initial_resource_snapshot = self._initialize_snapshots(self.list_graph)

        # --- Estado do Episódio ---
        self.graph: Graph = None
        self.current_sfc: SFC = None
        self.current_vnf: VNF = None
        self.current_location: int | str = None
        self.features = None
        self.initial_delay = 0.0

        # --- Espaços de Ação e Observação ---
        num_nodes = len(valid_nodes)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict(
            {
                "Se": spaces.Box(low=0, high=1, shape=(num_nodes, 1), dtype=np.float32),
                "Bw": spaces.Box(low=0, high=1, shape=(num_nodes, 1), dtype=np.float32),
                "Sm": spaces.Box(low=0, high=1, shape=(num_nodes, 1), dtype=np.float32),
                "Sr": spaces.Box(low=0, high=1, shape=(num_nodes, 2), dtype=np.float32),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.latency_used = 0
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
        self.allocation_results["dst"] = {
            "allocated_server": sfc_sorteada.dst_node,
            "path": [],
            "cost": 0,
        }
        self._deploy_initial_solution()
        self.features = self._get_nodes_features(vnf, bw_req, current_node)

        return self._get_obs(), {}

    def step(self, action: int):
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

        # 2. Tentar alocar recursos (CPU/cache) no nó escolhido
        if not self.allocate_resources_on_node(chosen_server, self.current_vnf):
            return self._fail_step("resource")

        if not path or not self.allocate_bandwidth_along_path(path, band_req):
            return self._fail_step("bandwidth")

        self.servers_used.append(chosen_server)
        self.latency_used += calculate_total_latency(self.graph, path, vnf)

        dst = self.current_sfc.get_substrate_node(self.current_sfc.get_dst_vnf())
        past_route_info = create_route_info_from_allocation_results(
            dst, self.graph, self.allocation_results
        )
        past_delay = get_sfc_latency_from_route(self.graph, self.current_sfc, past_route_info)
        self.allocation_results[vnf.id]["allocated_server"] = chosen_server
        self.allocation_results[vnf.id]["path"] = path

        route_info = create_route_info_from_allocation_results(
            dst, self.graph, self.allocation_results
        )
        current_delay = get_sfc_latency_from_route(self.graph, self.current_sfc, route_info)

        # 6. Atualizar estado para o próximo passo
        self.current_location = chosen_server

        # 7. Verificar conclusão e avançar para a próxima VNF/SFC
        done = False
        if self.current_vnf == self.reverse_vnf_list[-1]:
            bw_cost = self.features[action, 1]
            reward = self._calculate_reward(
                current_delay, past_delay, self.initial_delay, True, bw_cost
            )
            done = True
            self.success = True
            self.current_vnf = None
        else:
            bw_cost = self.features[action, 1]
            reward = self._calculate_reward(
                current_delay, past_delay, self.initial_delay, False, bw_cost
            )
            idx = self.reverse_vnf_list.index(self.current_vnf)
            self.current_vnf = self.reverse_vnf_list[idx + 1]

        bw_required = (
            self.service_requirements[self.current_vnf.id]["out_bw"] if self.current_vnf else 0
        )
        self.features = self._get_nodes_features(
            self.current_vnf, bw_required, self.current_location
        )

        return self._get_obs(), reward, done, False, {}

    # =================================================================================
    # 2. Lógica Central da Simulação e Estado
    # =================================================================================

    def _get_nodes_features(self, vnf: VNF, bw_required, current_location: any) -> np.ndarray:
        # Features: [latency_cost, bw_cost, is_reusable, cpu_norm, cache_norm, is_invalid]
        num_valid_nodes = len(self.valid_nodes)
        features = np.zeros((num_valid_nodes, 6))

        if not vnf:
            features[:, 5] = 1
            return features

        for i, node_id in enumerate(self.valid_nodes):
            if i == num_valid_nodes - 1:
                node_id = self.current_sfc.dst_node

            if node_id == self.current_sfc.dst_node:
                features[i, 5] = 1.0
                continue

            node_data = self.graph.nodes[node_id]
            is_reusable = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
            features[i, 2] = float(is_reusable)

            cpu_req = vnf.get_cpu_request() if not is_reusable else 0
            cache_req = vnf.get_cache_request() if not is_reusable else 0

            # CPU
            if node_data["cpu_capacity"] > 0:
                features[i, 3] = (
                    node_data["cpu_capacity"] - node_data["cpu_used"]
                ) / node_data["cpu_capacity"]
            else:
                features[i, 3] = 0.0
                features[i, 5] = 1.0

            # CACHE
            if node_data["cache_capacity"] > 0:
                features[i, 4] = (
                    node_data["cache_capacity"] - node_data["cache_used"]
                ) / node_data["cache_capacity"]
            else:
                features[i, 4] = 0.0
                features[i, 5] = 1.0

            # Verificação de Capacidade (Sobrecarga) combinada (SIM102 resolvido)
            if features[i, 5] == 0 and (
                (node_data["cpu_used"] + cpu_req > node_data["cpu_capacity"]) or
                (node_data["cache_used"] + cache_req > node_data["cache_capacity"])
            ):
                features[i, 5] = 1.0

            # Verificação de Caminho e Banda
            # Mesmo inválido, calculamos o caminho para preencher features[0] e [1]
            path = get_available_shortest_path_fast(
                self.graph, current_location, node_id, bw_required
            )

            if not path:
                features[i, 0] = 1.0
                features[i, 1] = 1.0
                features[i, 5] = 1.0
            else:
                bw_cost, latency_cost = self.calculate_bw_lat_cost(
                    vnf, node_id, path, bw_required
                )
                features[i, 0] = latency_cost
                features[i, 1] = bw_cost

                if bw_cost >= 999:
                    features[i, 5] = 1.0

        return features

    def _get_obs(self) -> dict[str, np.ndarray]:
        # idx_Se = [0], idx_bw = [1], idx_Sm = [2], idx_Sr = [3, 4]
        Se = self.features[:, [0]].astype(np.float32)
        Bw = self.features[:, [1]].astype(np.float32) / 10
        Sm = self.features[:, [2]].astype(np.float32)
        Sr = self.features[:, [3, 4]].astype(np.float32)

        Se[:, 0] /= 100

        return {"Se": Se, "Bw": Bw, "Sm": Sm, "Sr": Sr}

    def action_masks(self) -> np.ndarray:
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
        node = self.graph.nodes[node_id]
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()

        can_reuse = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
        effective_cpu_req = 0 if can_reuse else cpu_req
        effective_cache_req = 0 if can_reuse else cache_req

        if (node["cpu_used"] + effective_cpu_req > node["cpu_capacity"]) or (
            node["cache_used"] + effective_cache_req > node["cache_capacity"]
        ):
            return False

        node["cpu_used"] += effective_cpu_req
        node["cache_used"] += effective_cache_req

        return True

    def allocate_bandwidth_along_path(self, path: list, bandwidth_required: float) -> bool:
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges[u, v]
            available_bw = edge.get("bandwidth_capacity", 0) - edge.get("bandwidth_used", 0)
            if available_bw < bandwidth_required + 1e-9:
                return False

        for u, v in zip(path[:-1], path[1:], strict=False):
            self.graph.edges[u, v]["bandwidth_used"] += bandwidth_required

        return True

    def _set_list_graph_sfcs(self, list_graph: list[Graph], list_sfc: list[SFC]):
        if len(list_graph) != len(list_sfc):
            raise Exception(
                "O tamanho da lista de grafos deve ser igual ao de SFCs para correspondência"
            )
        self.list_graph = list_graph
        self.list_sfc = list_sfc
        self.reset()

    def _fail_step(self, reason: str):
        self.fail_reason = reason
        self.success = False

        reward = -100
        done = True

        bw_required = self.service_requirements[self.current_vnf.id]["out_bw"]
        self.features = self._get_nodes_features(
            self.current_vnf, bw_required, self.current_location
        )

        return self._get_obs(), reward, done, False, {}

    def _initialize_snapshots(self, list_graph: list[Graph] = None):
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

        if True:
            services.append("dst")
            service_requirements["dst"] = {"cpu": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

        self.service_requirements = service_requirements

    def define_reverse_vnf_list(self, sfc: SFC) -> list[VNF]:
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
        if not vnf:
            return False
        service_name = vnf.id

        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()

        node = graph.nodes[node_id]

        cpu_used, cpu_cap = node["cpu_used"], node["cpu_capacity"]
        cache_used, cache_cap = node["cache_used"], node["cache_capacity"]

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False

        session_id = sfc.id.split("_")[-1]
        service_key = (service_name, session_id)
        result = service_key in graph.nodes[node_id].get("services", {})

        if result and cpu_used + cpu_req >= cpu_cap and cache_used + cache_req >= cache_cap:
            return False

        return result

    def calculate_bw_lat_cost(self, vnf: VNF, server_id, path: list, bw_required: float):
        if not path or len(path) < 2:
            return 0, 0
        latency_cost = 0
        bw_cost = 0
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get("bandwidth_capacity", None)
            bd_used = edge.get("bandwidth_used", 0)

            latency_cost += calculate_latency_betwen_nodes(self.graph, u, v, vnf)

            if bd_capacity is None or bd_capacity == 0 or bw_required + bd_used > bd_capacity:
                return float(999), latency_cost

            link_cost = (bw_required + bd_used) / bd_capacity
            bw_cost += link_cost

        return bw_cost, latency_cost

    def _calculate_reward(
        self,
        current_delay: float,
        past_delay: float,
        initial_delay: float,
        is_last_step: bool,
        bw_cost=0,
    ) -> float:
        eta1 = self.pesos_fatores["eta1"]
        eta2 = self.pesos_fatores["eta2"]
        bw_factor = self.pesos_fatores["bw"]

        current_action_reward = past_delay - current_delay
        reward = eta1 * current_action_reward

        if is_last_step:
            last_step_reward = initial_delay - current_delay
            reward += eta2 * last_step_reward

        reward -= bw_cost * bw_factor
        return reward

    def _deploy_initial_solution(self):
        self.initial_delay = 0.0
        vnf_list = self.reverse_vnf_list
        dst = self.current_sfc.dst_node
        previous_node_location = dst

        for vnf in vnf_list:
            best_node_for_vnf = None
            min_latency_found = float("inf")
            best_path_segment = None

            bw_required = self.service_requirements[vnf.id]["out_bw"]

            for candidate_node in self.valid_nodes:
                if candidate_node == "M":
                    candidate_node = self.current_sfc.dst_node
                node_data = self.graph.nodes[candidate_node]
                reusable = self.is_reusable_at_node(
                    self.current_sfc, self.graph, candidate_node, vnf
                )
                cpu_req = vnf.get_cpu_request() if not reusable else 0
                cache_req = vnf.get_cache_request() if not reusable else 0

                if (node_data["cpu_used"] + cpu_req > node_data["cpu_capacity"]) or (
                    node_data["cache_used"] + cache_req > node_data["cache_capacity"]
                ):
                    continue

                # O caminho é do nó candidato ATÉ o local da VNF anterior
                path_segment = get_available_shortest_path_fast(
                    self.graph, candidate_node, previous_node_location, bw_required
                )

                if path_segment:
                    communication_latency = calculate_comunication_latency(
                        self.graph, path_segment, vnf
                    )

                    if communication_latency < min_latency_found:
                        min_latency_found = communication_latency
                        best_node_for_vnf = candidate_node
                        best_path_segment = path_segment

            if best_node_for_vnf is not None:
                previous_node_location = best_node_for_vnf
                self.allocation_results[vnf.id] = {
                    "allocated_server": best_node_for_vnf,
                    "path": best_path_segment,
                    "cost": 0,
                }
            else:
                return False

        G = self.graph
        route_info = create_route_info_from_allocation_results(dst, G, self.allocation_results)
        self.initial_delay = get_sfc_latency_from_route(G, self.current_sfc, route_info)
        return True

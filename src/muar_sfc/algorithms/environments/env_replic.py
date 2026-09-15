import math
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
import networkx as nx
from networkx import Graph
from muar_sfc.utils.network_utils import check_vnf_reusability

from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
    get_available_shortest_path_fast,
)
from muar_sfc.core.sfc import SFC, VNF
from muar_sfc.utils.network_utils import (
    calcular_percentual_banda_total,
    calcular_percentual_cache_total,
    get_graph_processing_utilization_simplified,
)

SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")
NON_REUSABLE_PENALTY = 4

LAT_MAX = 50.0
BW_MAX = 12.5
CPU_PENALTY_NORM = 0.3
CACHE_PENALTY_NORM = 0.3
EPSILON = 1e-5

class SFC_AllocationEnv(gymnasium.Env):
    """
    Ambiente do Gymnasium blindado contra Double Charging e erros de estado flutuante.
    """
    def __init__(
        self,
        valid_nodes: list[int | str],
        list_graph: list[Graph],
        list_sfc: list[SFC],
        pesos_fatores: dict[str, float] = None,
        reward_config: dict[str, float] = None,
        reliability_config: dict[str, Any] = None,
        is_training: bool = True,
    ):
        super().__init__()

        if len(list_graph) != len(list_sfc):
            raise ValueError("A lista de grafos deve ter o mesmo tamanho da lista de SFCs.")

        self.valid_nodes = valid_nodes
        self.is_training = is_training

        self.pesos_fatores = pesos_fatores or {
            "cpu": 1, "cache": 1, "band": 3, "rel": 6, "lat": 3, "mobile": 0.0,
        }

        self.reward_config = reward_config or {
            "success_bonus": 40.0, "step_reward": 0, "invalid_action_penalty": -10.0,
            "failure_penalty": -40.0, "severe_failure_penalty": -50.0,
        }

        self.reliability_config = reliability_config or {
            "tiers": {"default": 0.98, "a": 0.95, "b": 0.98, "c": 0.999},
            "stress": {"default": 0.08, "a": 0.15, "b": 0.08, "c": 0.02},
        }

        self.list_graph = []
        self.list_sfc = []
        # Usa o set interno para forçar isolamento e evitar o Double Charging
        self._set_list_graph_sfcs(list_graph, list_sfc)

        if self.is_training:
            self.initial_resource_snapshot = self._initialize_snapshots(self.list_graph)

        self.graph: Graph | None = None
        self.current_sfc: SFC | None = None
        self.current_vnf: VNF | None = None
        self.current_location: int | str = None
        self.latency_request = None
        self.features = None
        self.forbidden_nodes = set()

        self.ratio_cpu_used = 0
        self.ratio_cache_used = 0
        self.ratio_banda_used = 0
        self.latency_used = 0

        self.servers_used = []
        self.allocation_results = {}
        self.success = False
        self.fail_reason = None

        num_nodes = len(valid_nodes)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict(
            {
                "tipo_sfc": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "usos_rede": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
                "recursos_nos_validos": spaces.Box(low=0, high=1, shape=(num_nodes, 7), dtype=np.float32),
            }
        )

    def _set_list_graph_sfcs(self, list_graph: list[Graph], list_sfc: list[SFC]):
        """Cria cópias isoladas para a IA brincar, protegendo o grafo real do Instanciador."""
        safe_graphs = []
        for g in list_graph:
            sg = nx.Graph()
            for n, d in g.nodes(data=True): sg.add_node(n, **d.copy())
            for u, v, d in g.edges(data=True): sg.add_edge(u, v, **d.copy())
            safe_graphs.append(sg)
            
        self.list_graph = safe_graphs
        self.list_sfc = list_sfc
        
        # Refatoração EAFP Estrito
        try:
            _ = self.action_space
            self.reset()
        except AttributeError:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.latency_used = 0
        idx = 0

        if self.is_training:
            idx = np.random.randint(len(self.list_graph))
            self.graph = self.list_graph[idx]
            self._restore_graph_resources(idx)
        else:
            self.graph = self.list_graph[idx]

        self.set_current_sfc(self.list_sfc[idx])

        self.success = False
        self.fail_reason = None
        self.allocation_results = {}

        vnf = self.current_vnf
        bw_req = self.service_requirements[vnf.id]["out_bw"]
        current_node = self.current_location

        self.ratio_cpu_used = get_graph_processing_utilization_simplified(self.graph)
        self.ratio_cache_used = calcular_percentual_cache_total(self.graph)
        self.ratio_banda_used = calcular_percentual_banda_total(self.graph)

        self.features = self._get_nodes_features(vnf, bw_req, current_node)
        return self._get_obs(), {}

    def step(self, action: int):
        self.action_masks()

        # EAFP Limpo para resgate da VNF atual
        try:
            chosen_server = self.current_vnf.location
            if chosen_server is None: raise AttributeError
        except AttributeError:
            if self.current_vnf.id == "src_virt":
                vnf_data = next((v for v in self.current_sfc.vnfs_dict if v["name"] == "src_virt"), None)
                chosen_server = vnf_data["location"] if vnf_data else self.valid_nodes[action]
            else:
                if action == len(self.valid_nodes) - 1:
                    try:
                        chosen_server = self.current_sfc.dst_node
                    except AttributeError:
                        chosen_server = self.valid_nodes[action]
                else:
                    chosen_server = self.valid_nodes[action]

        vnf = self.current_vnf
        band_req = self.service_requirements[vnf.id]["out_bw"]
        current_location = self.current_location

        path = get_available_shortest_path_fast(self.graph, current_location, chosen_server, band_req)
        
        self.ratio_cpu_used = get_graph_processing_utilization_simplified(self.graph)
        self.ratio_cache_used = calcular_percentual_cache_total(self.graph)
        self.ratio_banda_used = calcular_percentual_banda_total(self.graph)

        total_cost = self._compute_allocation_cost(vnf, chosen_server, path, band_req)

        if not self.allocate_resources_on_node(chosen_server, self.current_vnf):
            return self._fail_step("resource")

        if not path or not self.allocate_bandwidth_along_path(path, band_req):
            return self._fail_step("bandwidth")

        self.servers_used.append(chosen_server)
        node_data = self.graph.nodes[chosen_server]
        reliability = self.get_dynamic_reliability(node_data)

        reward = -(total_cost / 5.0) + self.reward_config["step_reward"] + (4.0 * reliability)

        self.current_location = chosen_server
        if not self.is_training:
            self.allocation_results[self.current_vnf.id] = {
                "allocated_server": chosen_server, "path": path, "cost": total_cost,
            }

        self.latency_used += calculate_total_latency(self.graph, path, vnf)

        done = False
        if self.current_vnf == self.reverse_vnf_list[-1]:
            done = True
            self.success = True
            reward += self.reward_config["success_bonus"]
            self.current_vnf = None
        else:
            idx = self.reverse_vnf_list.index(self.current_vnf)
            self.current_vnf = self.reverse_vnf_list[idx + 1]

        bw_required = self.service_requirements[self.current_vnf.id]["out_bw"] if self.current_vnf else 0
        self.features = self._get_nodes_features(self.current_vnf, bw_required, self.current_location)
        
        return self._get_obs(), reward, done, False, {}

    def action_masks(self) -> np.ndarray:
        if self.current_vnf is None:
            return np.zeros(len(self.valid_nodes), dtype=np.int8)

        if self.features is None:
            vnf = self.current_vnf
            bw_req = self.service_requirements[vnf.id]["out_bw"]
            self.features = self._get_nodes_features(vnf, bw_req, self.current_location)

        mask = []
        for i, node_id in enumerate(self.valid_nodes):
            is_valid_resource = 1 if self.features[i, 6] == 0 else 0
            is_allowed_node = 1 if node_id not in self.forbidden_nodes else 0
            mask.append(is_valid_resource * is_allowed_node)

        return np.array(mask, dtype=np.int8)

    def _get_obs(self) -> dict[str, np.ndarray]:
        primeira_sf = np.zeros(1, dtype=np.float32)
        primeira_sf[0] = 1.0 if self.current_sfc.get_previous_vnf(self.current_sfc.get_dst_vnf()) == self.current_vnf else 0.0

        indices_das_features = [0, 1, 2, 3, 4, 5, 7]
        recursos_nodes = self.features[:, indices_das_features].astype(np.float32)
        recursos_nodes[:, 4] /= 100

        usos_rede = np.array([
            calcular_percentual_banda_total(self.graph),
            get_graph_processing_utilization_simplified(self.graph),
            self.latency_used / 100,
        ], dtype=np.float32)

        try:
            tipo_sfc = np.array([0.0], dtype=np.float32) if not self.current_vnf or "cache" in self.current_sfc.id else np.array([1.0], dtype=np.float32)
        except AttributeError:
            tipo_sfc = np.array([0.0], dtype=np.float32)

        return {
            "tipo_sfc": tipo_sfc,
            "usos_rede": usos_rede,
            "recursos_nos_validos": recursos_nodes,
        }

    def _get_nodes_features(self, vnf: VNF, bw_required, current_location: any) -> np.ndarray:
        num_valid_nodes = len(self.valid_nodes)
        features = np.zeros((num_valid_nodes, 8))

        if not vnf:
            features[:, 6] = 1
            return features

        for i, node_id in enumerate(self.valid_nodes):
            if i == num_valid_nodes - 1:
                try:
                    node_id = self.current_sfc.dst_node
                except AttributeError:
                    pass
                features[i, 7] = 1

            node_data = self.graph.nodes[node_id]
            is_reusable = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)

            features[i, 2] = float(is_reusable)
            cpu_req = 0 if is_reusable else vnf.get_cpu_request()
            cache_req = 0 if is_reusable else vnf.get_cache_request()

            c_cap = node_data.get("cpu_capacity", 0.0)
            ca_cap = node_data.get("cache_capacity", 0.0)

            features[i, 0] = (node_data.get("cpu_used", 0) + cpu_req) / c_cap if c_cap > 0 else 0
            features[i, 1] = (node_data.get("cache_used", 0) + cache_req) / ca_cap if ca_cap > 0 else 0

            # CORREÇÃO: Tolerância de + EPSILON protege a IA contra rejeições estritas de float
            if (node_data.get("cpu_used", 0) + cpu_req) > (c_cap + EPSILON) or (
                node_data.get("cache_used", 0) + cache_req
            ) > (ca_cap + EPSILON):
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

            features[i, 3] = self.get_dynamic_reliability(node_data)
            if node_id in self.forbidden_nodes:
                features[i, 6] = 1

        features[-1, 6] = 1
        return features

    def allocate_resources_on_node(self, node_id: int | str, vnf: VNF) -> bool:
        node = self.graph.nodes[node_id]
        can_reuse = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
        
        effective_cpu_req = 0 if can_reuse else vnf.get_cpu_request()
        effective_cache_req = 0 if can_reuse else vnf.get_cache_request()

        if (node.get("cpu_used", 0) + effective_cpu_req > node.get("cpu_capacity", 0)) or (
            node.get("cache_used", 0) + effective_cache_req > node.get("cache_capacity", 0)
        ):
            return False

        node["cpu_used"] = node.get("cpu_used", 0) + effective_cpu_req
        node["cache_used"] = node.get("cache_used", 0) + effective_cache_req
        return True
    
    def allocate_bandwidth_along_path(self, path: list, bandwidth_required: float) -> bool:
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges[u, v]
            # Devolvendo a margem de erro da arquitetura de ponto flutuante
            if (edge.get("bandwidth_capacity", 0) - edge.get("bandwidth_used", 0)) < (bandwidth_required - EPSILON):
                return False

        for u, v in zip(path[:-1], path[1:], strict=False):
            self.graph.edges[u, v]["bandwidth_used"] += bandwidth_required
        return True

    def is_reusable_at_node(self, sfc: SFC, graph: Graph, node_id: int | str, vnf: VNF) -> bool:
        try:
            service_name = vnf.id
        except AttributeError:
            return False

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False

        try:
            session_id = sfc.session_id
        except AttributeError:
            session_id = str(sfc.id).split("_")[-1]

        # Garantimos a extração limpa do ID
        clean_id = service_name.replace("_b", "")

        try:
            services_dict = graph.nodes[node_id].get("services", {})
            # Delegação SRP pura:
            return check_vnf_reusability(services_dict, clean_id, session_id)
        except KeyError:
            return False

    def _compute_allocation_cost(self, vnf, server_id, path, bw_required) -> float:
        node_data = self.graph.nodes[server_id]
        is_reusable = self.is_reusable_at_node(self.current_sfc, self.graph, server_id, vnf)

        c_cap = node_data.get("cpu_capacity", 1.0) or 1.0
        ca_cap = node_data.get("cache_capacity", 1.0) or 1.0

        cpu_cost = (node_data.get("cpu_used", 0) + vnf.get_cpu_request()) / c_cap
        cache_cost = (node_data.get("cache_used", 0) + vnf.get_cache_request()) / ca_cap

        if not is_reusable:
            cpu_cost = min(cpu_cost + CPU_PENALTY_NORM, 1.0)
            cache_cost = min(cache_cost + CACHE_PENALTY_NORM, 1.0)

        bw_cost_raw, lat_cost_raw = self.calculate_bw_lat_cost(vnf, server_id, path, bw_required)

        lat_cost = min(lat_cost_raw / LAT_MAX, 1.0)
        bw_cost = min(bw_cost_raw / BW_MAX, 1.0)
        reliability_cost = 1.0 - self.get_dynamic_reliability(node_data)

        incentive_cost = 0.0
        try:
            dst_node = self.current_sfc.dst_node
        except AttributeError:
            dst_node = None

        if server_id == dst_node:
            stress_factor = (max(self.ratio_cpu_used, self.ratio_cache_used) / 100.0) * self.pesos_fatores["mobile"]
            incentive_cost = -(stress_factor**3)

        return (
            self.pesos_fatores["cpu"] * cpu_cost
            + self.pesos_fatores["cache"] * cache_cost
            + self.pesos_fatores["band"] * bw_cost
            + self.pesos_fatores["lat"] * lat_cost
            + self.pesos_fatores["rel"] * reliability_cost
            + incentive_cost
        )

    def calculate_bw_lat_cost(self, vnf: VNF, server_id, path: list, bw_required: float):
        latency_cost = calculate_computational_latency(self.graph, server_id, vnf)
        if not path or len(path) < 2:
            return 0, latency_cost

        bw_cost = 0
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get("bandwidth_capacity", None)
            bd_used = edge.get("bandwidth_used", 0)

            latency_cost += calculate_latency_betwen_nodes(self.graph, u, v, vnf)

            if bd_capacity is None or bd_capacity == 0 or bw_required + bd_used > bd_capacity:
                return float(999), latency_cost

            link_cost = 1.0 / (1.0 - ((bw_required + bd_used) / bd_capacity) + 1e-6)
            bw_cost += link_cost

        return bw_cost, latency_cost

    def _fail_step(self, reason: str):
        self.fail_reason = reason
        self.success = False

        if reason == "invalid_action": reward = self.reward_config.get("invalid_action_penalty", -20.0)
        elif reason in ["resource", "bandwidth"]: reward = self.reward_config.get("failure_penalty", -80.0)
        else: reward = self.reward_config.get("severe_failure_penalty", -100.0)

        bw_req = self.service_requirements[self.current_vnf.id]["out_bw"]
        self.features = self._get_nodes_features(self.current_vnf, bw_req, self.current_location)
        
        return self._get_obs(), reward, True, False, {}

    def set_forbidden_nodes(self, nodes: list[str | int]):
        self.forbidden_nodes = set(nodes)

    def set_current_sfc(self, sfc: SFC):
        if not sfc: raise ValueError("SFC não pode ser None.")
        self.current_sfc = sfc
        self.reverse_vnf_list = self.define_reverse_vnf_list(sfc)
        self.current_vnf = self.reverse_vnf_list[0]
        
        try: self.current_location = self.current_sfc.dst_node
        except AttributeError: self.current_location = None

        self.servers_used = []
        self.services = []
        self.service_requirements = {}

        for item in sfc.vnfs_dict:
            nome = item["name"]
            self.services.append(nome)
            self.service_requirements[nome] = {
                "cpu": item["CPU"], "cache": item["cache"],
                "out_bw": item["out_bw"], "in_bw": item["in_bw"],
            }

        self.services.append("dst")
        self.service_requirements["dst"] = {"cpu": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

    def define_reverse_vnf_list(self, sfc: SFC) -> list[VNF]:
        vnf_list = []
        try:
            is_bkp = sfc.is_backup
        except AttributeError:
            is_bkp = "backup" in sfc.id

        if not is_bkp:
            current_vnf = sfc.get_previous_vnf(sfc.get_dst_vnf())
        else:
            current_vnf = sfc.get_previous_vnf(sfc.vnfs[sfc.vnfs_dict[-1]["name"]])

        while True:
            vnf_list.append(current_vnf)
            if not current_vnf.previous_vnf or current_vnf.previous_vnf.id == "src" or "virt" in current_vnf.id:
                break
            current_vnf = sfc.get_previous_vnf(current_vnf)
            
        return vnf_list

    def _initialize_snapshots(self, list_graph: list[Graph] = None):
        return {
            idx: {
                "nodes": {n: {"cpu_used": d.get("cpu_used", 0), "cache_used": d.get("cache_used", 0)} for n, d in g.nodes(data=True)},
                "edges": {(u, v): {"bandwidth_used": d.get("bandwidth_used", 0)} for u, v, d in g.edges(data=True)}
            } for idx, g in enumerate(list_graph)
        }

    def _restore_graph_resources(self, idx):
        snap = self.initial_resource_snapshot[idx]
        for n_id, st in snap["nodes"].items():
            if n_id in self.graph.nodes:
                self.graph.nodes[n_id].update(st)
        for (u, v), st in snap["edges"].items():
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v].update(st)

    def get_dynamic_reliability(self, node_data: dict) -> float:
        cpu_cap = node_data.get("cpu_capacity", 1.0) or node_data.get("original_cpu_capacity", 1.0) or 1.0
        server_level = str(node_data.get("level_server", "default")).lower()

        base_r = self.reliability_config["tiers"].get(server_level, self.reliability_config["tiers"]["default"])
        alpha = self.reliability_config["stress"].get(server_level, self.reliability_config["stress"]["default"])

        return max(0.0, base_r - (min(node_data.get("cpu_used", 0.0) / cpu_cap, 1.0) * alpha))

def calculate_total_latency(graph: Graph, path: list, vnf: VNF):
    lat = sum(calculate_latency_betwen_nodes(graph, path[i], path[i+1], vnf) for i in range(len(path)-1))
    return lat + calculate_computational_latency(graph, path[-1], vnf) if path else lat
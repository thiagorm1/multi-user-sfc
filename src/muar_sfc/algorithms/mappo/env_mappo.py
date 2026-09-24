from typing import Any

import networkx as nx
import numpy as np

from muar_sfc.algorithms.mappo.partitioner import RegionalTopologyPartitioner
from muar_sfc.algorithms.networkUtils import (
    calculate_latency_betwen_nodes,
    get_available_shortest_path_fast,
)
from muar_sfc.core.sfc import SFC, VNF


def safe_comp_latency(graph: nx.Graph, node: Any, vnf: VNF) -> float:
    """Calcula latência computacional com fallback seguro para qualquer topologia."""
    if node == 0:
        return 0.0
    try:
        ips = graph.nodes[node].get("ips")
        if ips:
            packet = vnf.get_income_interface_bandwidth() / 60 * 1e6
            return float(packet * 10 * 1000 / ips)
        mips = graph.nodes[node].get("servers_mips")
        if mips:
            packet = vnf.get_income_interface_bandwidth() / 60 * 1e6
            return float(packet * 10 * 1000 / (mips * 1e6 if mips < 1e6 else mips))
    except Exception:
        pass
    return 1.0


class MAPPO_SFC_Env:
    """
    Ambiente Multi-Agente para MAPPO (CTDE) no contexto de SFC em redes Cloud-Edge 6G.

    Suporta:
    - K Agentes Regionais de Borda + Tier de Nuvem.
    - Observações descentralizadas (o_t^k) com visão do cluster regional.
    - Estado global centralizado (s_t) para o Crítico Central com Load Imbalance (LB).
    - Máscaras de ação (Action Masking) garantindo restrições rígidas de CPU/Cache/Banda.
    - Cadeias Lineares e Grafos Acíclicos Dirigidos (DAG) com Sincronização Inter-Modal.
    """

    def __init__(
        self,
        graph: nx.Graph,
        sfc: SFC | None = None,
        n_regions: int = 4,
        cloud_node: int | str = 0,
        sync_tolerance: float = 5.0,
        is_training: bool = False,
        cognitive_guidance: Any = None,
    ):
        if isinstance(graph, nx.Graph):
            self.base_graph = graph
        elif hasattr(graph, "graph"):
            self.base_graph = graph.graph
        else:
            self.base_graph = graph
        self.graph = self.base_graph.copy()
        self.sfc = sfc
        self.n_regions = n_regions
        self.cloud_node = cloud_node
        self.sync_tolerance = sync_tolerance
        self.is_training = is_training
        self.cognitive_guidance = cognitive_guidance

        self.partitioner = RegionalTopologyPartitioner(
            self.graph, n_regions=n_regions, cloud_node=cloud_node
        )
        self.cloud_node = self.partitioner.cloud_node

        # Determina o número máximo de nós por região para padronizar o action space
        self.region_nodes = {
            r: self.partitioner.get_nodes_in_region(r) for r in range(self.n_regions)
        }
        has_nodes = self.region_nodes and any(self.region_nodes.values())
        self.max_nodes_per_region = (
            max(len(n) for n in self.region_nodes.values()) if has_nodes else 1
        )
        # Ação: nó local na região (0 .. max_nodes - 1) ou ação Nuvem (max_nodes)
        self.action_dim_per_agent = self.max_nodes_per_region + 1

        # Dimensão da observação por agente
        # Por nó da região: [cpu_util, cache_util, avg_bw_free, path_latency_norm]
        self.node_feat_dim = 4
        # Requisitos da VNF: [cpu_norm, cache_norm, bw_norm, is_fork, is_join, dist_to_dst]
        self.vnf_feat_dim = 6
        self.obs_dim = (self.max_nodes_per_region * self.node_feat_dim) + self.vnf_feat_dim

        # Dimensão do estado global: K * obs_dim + estatísticas globais
        self.global_stat_dim = 4
        self.state_dim = (self.n_regions * self.obs_dim) + self.global_stat_dim

        # Pesos da função de recompensa (artigo 6G + prompt)
        self.w_latency = 1.0
        self.w_cpu = 0.5
        self.w_lb = 1.5
        self.lat_max = 50.0

        # Estado da simulação
        self.current_location = 0
        self.current_vnf_idx = 0
        self.vnf_sequence: list[VNF] = []
        self.allocated_nodes: dict[str, Any] = {}
        self.route_info: dict[str, list[Any]] = {}
        self.total_latency = 0.0
        self.success = False
        self.fail_reason = None

        if self.sfc:
            self.reset_sfc(self.sfc)

    def reset_sfc(self, sfc: SFC):
        """Reinicia o ambiente para uma nova SFC."""
        self.sfc = sfc
        self.graph = self.base_graph.copy()
        self.current_vnf_idx = 0
        self.allocated_nodes = {}
        self.route_info = {}
        self.total_latency = 0.0
        self.success = False
        self.fail_reason = None

        # Identifica nós de origem e destino
        try:
            src_vnf = self.sfc.get_src_vnf()
            self.src_node = self.sfc.get_substrate_node(src_vnf)
        except Exception:
            self.src_node = 0

        try:
            self.dst_node = self.sfc.dst_node
        except Exception:
            self.dst_node = 0

        self.current_location = self.src_node

        # Define a sequência de VNFs para alocação (ordem topológica para DAGs)
        self.vnf_sequence = self._extract_vnf_sequence(sfc)
        return self.get_observations(), self.get_global_state()

    def _extract_vnf_sequence(self, sfc: SFC) -> list[VNF]:
        """Extrai as VNFs intermediárias em ordem topológica ou linear."""
        if hasattr(sfc, "is_dag") and sfc.is_dag() and hasattr(sfc, "graph"):
            try:
                topo = list(nx.topological_sort(sfc.graph))
                return [sfc.get_vnf_by_id(n) for n in topo if n not in ("src", "dst")]
            except Exception:
                pass

        # Linear fallback
        return [v for k, v in sfc.vnfs.items() if k not in ("src", "dst")]

    def set_cognitive_guidance(self, guidance: Any) -> None:
        """Atualiza dinamicamente as diretrizes cognitivas do CloudLLMPlanner."""
        self.cognitive_guidance = guidance

    def get_action_masks(self) -> dict[int, np.ndarray]:
        """
        Gera máscara booleana/binária de ações válidas para cada agente k.
        1 se o nó tem capacidade de CPU/Cache para a VNF atual e link viável, 0 caso contrário.
        Aplica Poda de Ações Cognitiva (Action Mask Pruning) baseada em avoid_nodes do LLM.
        """
        masks = {}
        if self.current_vnf_idx >= len(self.vnf_sequence):
            for r in range(self.n_regions):
                masks[r] = np.ones(self.action_dim_per_agent, dtype=np.float32)
            return masks

        curr_vnf = self.vnf_sequence[self.current_vnf_idx]
        cpu_req = curr_vnf.get_cpu_request()
        cache_req = curr_vnf.get_cache_request()
        bw_req = curr_vnf.get_outcome_interface_bandwidth()

        avoid_nodes = set(getattr(self.cognitive_guidance, "avoid_nodes", []) or [])

        for r in range(self.n_regions):
            mask = np.zeros(self.action_dim_per_agent, dtype=np.float32)
            nodes = self.region_nodes[r]

            for idx, node in enumerate(nodes):
                node_data = self.graph.nodes[node]
                cpu_free = (
                    node_data.get("cpu_capacity", 0.0) - node_data.get("cpu_used", 0.0)
                )
                cache_free = (
                    node_data.get("cache_capacity", 0.0) - node_data.get("cache_used", 0.0)
                )

                if cpu_free >= cpu_req and cache_free >= cache_req:
                    # Verifica conectividade básica
                    path = get_available_shortest_path_fast(
                        self.graph, self.current_location, node, bw_req
                    )
                    if path:
                        mask[idx] = 1.0

            # Ação de Nuvem (sempre válida como fallback seguro de capacidade)
            cloud_idx = self.max_nodes_per_region
            mask[cloud_idx] = 1.0

            # Poda Cognitiva de Ação (Action Mask Pruning via LLM Guidance):
            # Se nós da região foram marcados como congestionados no avoid_nodes,
            # mascara-os desde que reste ao menos 1 candidato viável (local ou nuvem).
            if avoid_nodes:
                for idx, node in enumerate(nodes):
                    if node in avoid_nodes and mask[idx] == 1.0 and mask.sum() > 1:
                        mask[idx] = 0.0

            # Se todos os nós locais forem inválidos, a nuvem absorve
            if mask.sum() == 0:
                mask[cloud_idx] = 1.0

            masks[r] = mask

        return masks

    def get_observations(self) -> dict[int, np.ndarray]:
        """Gera o vetor de observação local o_t^k para cada agente k."""
        obs = {}
        if self.current_vnf_idx < len(self.vnf_sequence):
            curr_vnf = self.vnf_sequence[self.current_vnf_idx]
            cpu_req = curr_vnf.get_cpu_request() / 100.0
            cache_req = curr_vnf.get_cache_request() / 1000.0
            bw_req = curr_vnf.get_outcome_interface_bandwidth() / 1000.0
            is_fork = 1.0 if len(getattr(curr_vnf, "next_vnfs", [])) > 1 else 0.0
            is_join = 1.0 if len(getattr(curr_vnf, "previous_vnfs", [])) > 1 else 0.0
            dst_reg = float(self.partitioner.get_region_for_node(self.dst_node))
            vnf_feats = np.array(
                [cpu_req, cache_req, bw_req, is_fork, is_join, dst_reg / max(1, self.n_regions)],
                dtype=np.float32,
            )
        else:
            vnf_feats = np.zeros(self.vnf_feat_dim, dtype=np.float32)

        for r in range(self.n_regions):
            nodes = self.region_nodes[r]
            node_feats = np.zeros(
                (self.max_nodes_per_region, self.node_feat_dim), dtype=np.float32
            )

            for idx, node in enumerate(nodes):
                data = self.graph.nodes[node]
                c_cap = float(data.get("cpu_capacity") or 100.0)
                ca_cap = float(data.get("cache_capacity") or 100.0)
                c_used = float(data.get("cpu_used", 0.0))
                ca_used = float(data.get("cache_used", 0.0))

                cpu_util = c_used / max(1.0, c_cap)
                cache_util = ca_used / max(1.0, ca_cap)

                # Média de banda livre dos links conectados
                incident_edges = list(self.graph.edges(node, data=True))
                if incident_edges:
                    bw_free = float(np.mean([
                        max(
                            0.0,
                            float(d.get("bandwidth_capacity") or 1000.0)
                            - float(d.get("bandwidth_used", 0.0)),
                        )
                        for _, _, d in incident_edges
                    ])) / 1000.0
                else:
                    bw_free = 0.0

                # Latência estimada da localização atual até este nó
                try:
                    lat = nx.shortest_path_length(
                        self.graph, source=self.current_location, target=node, weight="latency"
                    )
                    lat_norm = min(1.0, lat / self.lat_max)
                except Exception:
                    lat_norm = 1.0

                node_feats[idx] = [cpu_util, cache_util, bw_free, lat_norm]

            obs[r] = np.nan_to_num(
                np.concatenate([node_feats.flatten(), vnf_feats]).astype(np.float32),
                nan=0.0, posinf=1.0, neginf=0.0
            )

        return obs

    def get_global_state(self) -> np.ndarray:
        """Gera o estado centralizado s_t com Load Imbalance LB_t para o Crítico Central."""
        obs = self.get_observations()
        all_obs_concat = np.concatenate([obs[r] for r in range(self.n_regions)])

        # Cálculo do Load Imbalance (LB)
        cpu_utils = []
        for n, data in self.graph.nodes(data=True):
            if n != self.cloud_node and data.get("type") in ("server", "edge_server"):
                cap = float(data.get("cpu_capacity") or 100.0)
                used = float(data.get("cpu_used", 0.0))
                cpu_utils.append(used / max(1.0, cap))

        mean_cpu = float(np.mean(cpu_utils)) if cpu_utils else 0.0
        lb_t = float(np.std(cpu_utils)) if cpu_utils else 0.0

        # Estatísticas globais
        global_stats = np.array([
            mean_cpu,
            lb_t,
            self.total_latency / self.lat_max,
            self.current_vnf_idx / max(1, len(self.vnf_sequence)),
        ], dtype=np.float32)

        return np.nan_to_num(
            np.concatenate([all_obs_concat, global_stats]).astype(np.float32),
            nan=0.0, posinf=1.0, neginf=0.0
        )

    def step(
        self, actions: dict[int, int]
    ) -> tuple[dict[int, np.ndarray], np.ndarray, float, bool, dict[str, Any]]:
        """
        Executa um passo de decisão multi-agente cooperativo.

        Cada agente k propõe um nó candidato de sua região (ou Cloud).
        O orquestrador seleciona a melhor proposta com base na proximidade e custo total,
        aloca os recursos e avança na cadeia.
        """
        if self.current_vnf_idx >= len(self.vnf_sequence):
            return self.get_observations(), self.get_global_state(), 0.0, True, {"success": True}

        curr_vnf = self.vnf_sequence[self.current_vnf_idx]
        cpu_req = curr_vnf.get_cpu_request()
        cache_req = curr_vnf.get_cache_request()
        bw_req = curr_vnf.get_outcome_interface_bandwidth()

        # Coleta os nós candidatos propostos por cada agente
        candidates = []
        for r, act in actions.items():
            if act == self.max_nodes_per_region:
                node = self.cloud_node
            else:
                nodes_in_reg = self.region_nodes[r]
                node = nodes_in_reg[act] if act < len(nodes_in_reg) else self.cloud_node
            candidates.append(node)

        # Avalia os candidatos e escolhe o de menor custo (latência acumulada + uso)
        best_node = None
        best_cost = float("inf")
        best_path = None

        for cand in candidates:
            # Verifica recursos
            data = self.graph.nodes[cand]
            cpu_free = data.get("cpu_capacity", 0.0) - data.get("cpu_used", 0.0)
            cache_free = data.get("cache_capacity", 0.0) - data.get("cache_used", 0.0)

            if cand != self.cloud_node and (cpu_free < cpu_req or cache_free < cache_req):
                continue

            path = get_available_shortest_path_fast(
                self.graph, self.current_location, cand, bw_req
            )
            if not path and self.current_location != cand:
                continue

            comp_lat = safe_comp_latency(self.graph, cand, curr_vnf)
            comm_lat = sum(
                calculate_latency_betwen_nodes(self.graph, path[i], path[i + 1], curr_vnf)
                for i in range(len(path) - 1)
            ) if path else 0.0

            step_lat = comp_lat + comm_lat

            w_lat = 0.5
            w_lb = 0.5
            if self.cognitive_guidance is not None:
                w_lat = float(getattr(self.cognitive_guidance, "latency_weight", 0.5))
                w_lb = float(getattr(self.cognitive_guidance, "load_balance_weight", 0.5))

            cpu_util = data.get("cpu_used", 0.0) / max(1.0, data.get("cpu_capacity", 1.0))
            avoid_nodes_list = getattr(self.cognitive_guidance, "avoid_nodes", []) or []
            avoid_penalty = 50.0 if (cand in avoid_nodes_list) else 0.0

            cost = (w_lat * step_lat) + (w_lb * cpu_util * 25.0) + avoid_penalty

            if cost < best_cost:
                best_cost = cost
                best_node = cand
                best_path = path or [cand]

        if best_node is None:
            # Fallback forçado para Cloud se nenhum candidato de borda suportar
            best_node = self.cloud_node
            best_path = nx.shortest_path(
                self.graph, source=self.current_location, target=self.cloud_node, weight="latency"
            )

        # Deduz recursos no grafo
        node_data = self.graph.nodes[best_node]
        node_data["cpu_used"] = node_data.get("cpu_used", 0.0) + cpu_req
        node_data["cache_used"] = node_data.get("cache_used", 0.0) + cache_req

        if best_path and len(best_path) > 1:
            for u, v in zip(best_path[:-1], best_path[1:], strict=False):
                self.graph.edges[u, v]["bandwidth_used"] = (
                    self.graph.edges[u, v].get("bandwidth_used", 0.0) + bw_req
                )

        # Registra alocação
        self.allocated_nodes[curr_vnf.id] = best_node
        self.route_info[curr_vnf.id] = list(reversed(best_path)) if best_path else [best_node]

        # Calcula latência do passo
        comp_lat = safe_comp_latency(self.graph, best_node, curr_vnf)
        comm_lat = sum(
            calculate_latency_betwen_nodes(self.graph, best_path[i], best_path[i + 1], curr_vnf)
            for i in range(len(best_path) - 1)
        ) if best_path else 0.0
        self.total_latency += comp_lat + comm_lat

        self.current_location = best_node
        self.current_vnf_idx += 1

        done = self.current_vnf_idx >= len(self.vnf_sequence)
        reward = self._compute_reward(done)

        info = {
            "allocated_node": best_node,
            "step_latency": comp_lat + comm_lat,
            "total_latency": self.total_latency,
        }

        if done:
            self._finalize_chain()
            info["success"] = self.success
            info["route_info"] = self.route_info
            info["latency"] = self.total_latency

        return self.get_observations(), self.get_global_state(), reward, done, info

    def _compute_reward(self, done: bool) -> float:
        """
        Função de recompensa multi-agente:
        r_t = - (w1 * lat/lat_max + w2 * cpu_util + w3 * LB) - penalidades_SLA + bonus_sucesso
        """
        cpu_utils = [
            self.graph.nodes[n].get("cpu_used", 0.0)
            / max(1.0, float(self.graph.nodes[n].get("cpu_capacity") or 1.0))
            for n in self.graph.nodes()
            if n != self.cloud_node
        ]
        lb_t = float(np.std(cpu_utils)) if cpu_utils else 0.0
        avg_cpu = float(np.mean(cpu_utils)) if cpu_utils else 0.0

        w_lat = self.w_latency
        w_lb = self.w_lb

        # Dynamic Reward Shaping guiado pela Meta-Política do LLM Planner:
        if self.cognitive_guidance is not None:
            lat_w = float(getattr(self.cognitive_guidance, "latency_weight", 0.5))
            load_w = float(getattr(self.cognitive_guidance, "load_balance_weight", 0.5))
            w_lat = self.w_latency * (lat_w / 0.5)
            w_lb = self.w_lb * (load_w / 0.5)

        r_lat = w_lat * (self.total_latency / self.lat_max)
        r_cpu = self.w_cpu * avg_cpu
        r_lb = w_lb * lb_t

        reward = -(r_lat + r_cpu + r_lb)

        # Bônus cognitivo por alocação de cache nos nós recomendados
        p_cache = getattr(self.cognitive_guidance, "priority_cache_nodes", None)
        if p_cache:
            for v_id, node in self.allocated_nodes.items():
                if "MA" in str(v_id) and node in p_cache:
                    reward += 3.0

        if done:
            # Penalidade de SLA de Latência Máxima
            lat_req = self.sfc.get_latency_request() if self.sfc else self.lat_max
            if self.total_latency > lat_req:
                reward -= 15.0
            else:
                reward += 25.0  # Bônus de cumprimento de SLA

        return float(reward)

    def _finalize_chain(self):
        """Conecta o último hop ao nó de destino e calcula caminho crítico em DAGs."""
        # Caminho final da última VNF ao destino
        try:
            path_to_dst = nx.shortest_path(
                self.graph, source=self.current_location, target=self.dst_node, weight="latency"
            )
            self.route_info["dst"] = path_to_dst
            for i in range(len(path_to_dst) - 1):
                edge_data = self.graph.edges[path_to_dst[i], path_to_dst[i + 1]]
                self.total_latency += edge_data.get("latency", 1.0)
        except Exception:
            self.route_info["dst"] = [self.dst_node]

        # Caminho da origem para a primeira VNF
        try:
            first_vnf = self.vnf_sequence[0]
            first_node = self.allocated_nodes[first_vnf.id]
            path_from_src = nx.shortest_path(
                self.graph, source=self.src_node, target=first_node, weight="latency"
            )
            self.route_info["src"] = path_from_src
            for i in range(len(path_from_src) - 1):
                edge_data = self.graph.edges[path_from_src[i], path_from_src[i + 1]]
                self.total_latency += edge_data.get("latency", 1.0)
        except Exception:
            self.route_info["src"] = [self.src_node]

        self.success = True

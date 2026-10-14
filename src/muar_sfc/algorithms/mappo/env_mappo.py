import copy
import math
from typing import Any

import networkx as nx
import numpy as np

from muar_sfc.algorithms.mappo.partitioner import RegionalTopologyPartitioner
from muar_sfc.algorithms.networkUtils import (
    calculate_latency_betwen_nodes,
)
from muar_sfc.core.orchestration.placement import (
    available_path,
    bandwidth_demand,
    node_demand,
    reserve_bandwidth,
    reserve_node,
    stream_key,
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
        share: bool = False,
        shareband: bool = False,
        policy_weight: float = 0.5,
        proposal_only: bool = False,
        destination_weight: float = 0.25,
        load_balance_scale: float = 10.0,
        cache_preference_weight: float = 0.0,
    ):
        if isinstance(graph, nx.Graph):
            self.base_graph = graph
        elif hasattr(graph, "graph"):
            self.base_graph = graph.graph
        else:
            self.base_graph = graph
        self.graph = copy.deepcopy(self.base_graph)
        self.sfc = sfc
        self.n_regions = n_regions
        self.cloud_node = cloud_node
        self.sync_tolerance = sync_tolerance
        self.is_training = is_training
        self.cognitive_guidance = cognitive_guidance
        self.share = share
        self.shareband = shareband
        self.policy_weight = policy_weight
        self.proposal_only = proposal_only
        self.destination_weight = destination_weight
        self.load_balance_scale = load_balance_scale
        self.cache_preference_weight = cache_preference_weight

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
        self.graph = copy.deepcopy(self.base_graph)
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

        self.edge_routes = {}
        self.edge_latencies = {}
        self.finish_times = {"src": 0.0}
        self.comp_latencies = {}
        self.sync_differential = 0.0
        self.excluded_nodes = {}
        self.current_location = self.src_node
        self.lat_max = float(self.sfc.get_latency_request() or 50.0)

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

    def _copy_trial(self):
        graph = self.graph.copy()
        for _, data in graph.nodes(data=True):
            data["services"] = dict(data.get("services", {}))
        for _, _, data in graph.edges(data=True):
            data["services_in_transit"] = dict(data.get("services_in_transit", {}))
        return graph

    def _route_inputs(self, graph, vnf_id, node):
        routes, latencies, arrivals = {}, {}, []
        for predecessor in self.sfc.graph.predecessors(vnf_id):
            start = self.src_node if predecessor == "src" else self.allocated_nodes[predecessor]
            bw = self.sfc.graph.edges[predecessor, vnf_id]["bandwidth"]
            key = stream_key(self.sfc, predecessor, vnf_id, self.shareband)
            path = available_path(graph, start, node, bw, key, self.shareband)
            if not path:
                raise ValueError("Sem rota com banda suficiente para uma dependência.")
            sender = copy.copy(self.sfc.get_vnf_by_id(predecessor))
            sender.set_outcome_interface_bandwidth(bw)
            latency = 0.0
            for u, v in zip(path[:-1], path[1:], strict=True):
                reserve_bandwidth(graph.edges[u, v], key, bw, self.shareband)
                latency += calculate_latency_betwen_nodes(graph, u, v, sender)
            dependency = (predecessor, vnf_id)
            routes[dependency] = path
            latencies[dependency] = latency
            arrivals.append(self.finish_times[predecessor] + latency)
        differential = max(arrivals, default=0) - min(arrivals, default=0)
        if len(arrivals) > 1 and differential > self.sync_tolerance:
            raise ValueError("Sincronização inter-modal excedeu o SLA.")
        return routes, latencies, max(arrivals, default=0), differential

    def _candidate(self, node):
        vnf = self.vnf_sequence[self.current_vnf_idx]
        if node not in self.graph or node in self.excluded_nodes.get(vnf.id, set()):
            return None
        trial = self._copy_trial()
        try:
            cpu, cache = reserve_node(trial, self.sfc, vnf, node, self.share)
            routes, latencies, start, differential = self._route_inputs(trial, vnf.id, node)
        except ValueError:
            return None
        comp = safe_comp_latency(trial, node, vnf)
        finish = start + comp
        if not math.isfinite(finish) or finish > self.lat_max:
            return None
        # Do not choose a host that cannot reach the destination with residual bandwidth.
        if "dst" in self.sfc.graph.successors(vnf.id):
            bw = self.sfc.graph.edges[vnf.id, "dst"]["bandwidth"]
            key = stream_key(self.sfc, vnf.id, "dst", self.shareband)
            if not available_path(trial, node, self.dst_node, bw, key, self.shareband):
                return None
        return trial, routes, latencies, comp, finish, differential, cpu, cache

    def get_action_masks(self) -> dict[int, np.ndarray]:
        if self.current_vnf_idx >= len(self.vnf_sequence):
            return {
                r: np.ones(self.action_dim_per_agent, dtype=np.float32)
                for r in range(self.n_regions)
            }
        feasible = {
            n: self._candidate(n) is not None
            for n in {self.cloud_node, *(n for nodes in self.region_nodes.values() for n in nodes)}
        }
        return {
            r: np.array(
                [float(feasible[n]) for n in self.region_nodes[r]]
                + [0.0] * (self.max_nodes_per_region - len(self.region_nodes[r]))
                + [float(feasible[self.cloud_node])],
                dtype=np.float32,
            )
            for r in range(self.n_regions)
        }

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
                if any(
                    math.isfinite(d.get("bandwidth_capacity", 0)) for _, _, d in incident_edges
                ):
                    bw_free = (
                        float(
                            np.mean(
                                [
                                    max(
                                        0.0,
                                        float(d.get("bandwidth_capacity") or 1000.0)
                                        - float(d.get("bandwidth_used", 0.0)),
                                    )
                                    for _, _, d in incident_edges
                                    if math.isfinite(d.get("bandwidth_capacity", 0))
                                ]
                            )
                        )
                        / 1000.0
                    )
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
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
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
        global_stats = np.array(
            [
                mean_cpu,
                lb_t,
                self.total_latency / self.lat_max,
                self.current_vnf_idx / max(1, len(self.vnf_sequence)),
            ],
            dtype=np.float32,
        )

        return np.nan_to_num(
            np.concatenate([all_obs_concat, global_stats]).astype(np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def step(self, actions):
        if self.current_vnf_idx >= len(self.vnf_sequence):
            return (
                self.get_observations(),
                self.get_global_state(),
                0.0,
                True,
                {"success": self.success},
            )
        vnf = self.vnf_sequence[self.current_vnf_idx]
        proposals = set()
        for region, action in actions.items():
            if action == self.max_nodes_per_region:
                proposals.add(self.cloud_node)
            elif 0 <= action < len(self.region_nodes[region]):
                proposals.add(self.region_nodes[region][action])
        candidates = (
            proposals
            if self.is_training or self.proposal_only
            else {self.cloud_node, *(n for nodes in self.region_nodes.values() for n in nodes)}
        )
        best = None
        best_cost = float("inf")
        for node in sorted(candidates, key=str):
            result = self._candidate(node)
            if result is None:
                continue
            trial, routes, latencies, comp, finish, differential, cpu, cache = result
            data = trial.nodes[node]
            cpu_util = data.get("cpu_used", 0) / max(1.0, data.get("cpu_capacity", 0))
            cache_util = data.get("cache_used", 0) / max(1.0, data.get("cache_capacity", 0))
            lat_weight = getattr(self.cognitive_guidance, "latency_weight", 0.5)
            load_weight = getattr(self.cognitive_guidance, "load_balance_weight", 0.5)
            reused = node_demand(self.graph, self.sfc, vnf, node, self.share) == (0, 0)
            avoid = node in (getattr(self.cognitive_guidance, "avoid_nodes", []) or [])
            reserved = sum(
                bandwidth_demand(
                    self.graph.edges[u, v],
                    stream_key(self.sfc, *dep, self.shareband),
                    self.sfc.graph.edges[dep]["bandwidth"],
                    self.shareband,
                )
                for dep, path in routes.items()
                for u, v in zip(path[:-1], path[1:], strict=True)
            )
            try:
                destination_distance = nx.shortest_path_length(
                    trial, node, self.dst_node, weight="latency"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            cost = (
                lat_weight * (finish + self.destination_weight * destination_distance)
                + load_weight * (cpu_util + cache_util) * self.load_balance_scale
                + reserved / 1000
                + differential
                + float(avoid) * cpu_util * 4
                - self.cache_preference_weight
                * float(
                    vnf.get_cache_request() > 0
                    and node
                    in (getattr(self.cognitive_guidance, "priority_cache_nodes", []) or [])
                )
                - float(reused) * 4
                - float(node in proposals) * self.policy_weight
            )
            if cost < best_cost:
                best_cost = cost
                best = node, result
        if best is None:
            self.success = False
            self.fail_reason = "Sem alocação viável para recursos, rotas ou SLA."
            return (
                self.get_observations(),
                self.get_global_state(),
                -30.0,
                True,
                {"success": False, "fail_reason": self.fail_reason},
            )
        node, result = best
        (self.graph, routes, latencies, comp, finish, differential, _, _) = result
        self.edge_routes.update(routes)
        self.edge_latencies.update(latencies)
        self.comp_latencies[vnf.id] = comp
        self.finish_times[vnf.id] = finish
        self.sync_differential = max(self.sync_differential, differential)
        self.allocated_nodes[vnf.id] = node
        self.route_info[vnf.id] = list(reversed(next(iter(routes.values()), [node])))
        self.total_latency = max(self.finish_times.values())
        self.current_location = node
        self.current_vnf_idx += 1
        done = self.current_vnf_idx == len(self.vnf_sequence)
        if done:
            self._finalize_chain()
        reward = self._compute_reward(done)
        return (
            self.get_observations(),
            self.get_global_state(),
            reward,
            done,
            {
                "allocated_node": node,
                "success": self.success if done else False,
                "total_latency": self.total_latency,
                "latency": self.total_latency,
                "route_info": self.route_info,
                "sync_differential": self.sync_differential,
            },
        )

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
                reward += 30.0 if self.success else -30.0

        return float(reward)

    def _finalize_chain(self):
        """Reserve all terminal dependencies and measure the DAG critical path."""
        trial = self._copy_trial()
        try:
            routes, latencies, finish, differential = self._route_inputs(
                trial, "dst", self.dst_node
            )
        except (ValueError, KeyError):
            self.success = False
            self.fail_reason = (
                "Sem rota com banda suficiente até o destino ou sincronização inválida."
            )
            return
        self.graph = trial
        self.edge_routes.update(routes)
        self.edge_latencies.update(latencies)
        self.finish_times["dst"] = finish
        self.total_latency = finish
        self.sync_differential = max(self.sync_differential, differential)
        self.route_info["src"] = [self.src_node]
        self.route_info["dst"] = [self.dst_node]
        self.success = finish <= self.lat_max
        if not self.success:
            self.fail_reason = "Latência do caminho crítico excedeu o SLA."

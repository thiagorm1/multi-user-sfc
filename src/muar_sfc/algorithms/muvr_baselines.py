"""Explicit DAG adapters for comparison; these are not paper reproductions."""

import copy
from pathlib import Path

import networkx as nx
import numpy as np
from sb3_contrib import MaskablePPO

from muar_sfc.algorithms.dag_greedy import DAGGreedyAlgorithm
from muar_sfc.algorithms.environments.env_da_rsppo import SFC_AllocationEnv_DARSPPO
from muar_sfc.algorithms.environments.env_replic import SFC_AllocationEnv
from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env
from muar_sfc.core.orchestration.placement import is_shareable
from muar_sfc.core.scenarios.muvr_reference import plan_colocated


class MUVRBaseline:
    def __init__(self, strategy, beam_width=4):
        self.strategy = strategy
        self.beam_width = beam_width
        self.model = None
        self.model_path = None
        if strategy in ("replic", "darsppo"):
            self.model_path = Path(
                f"rl_saved_models/MASKABLEPPO_{strategy.upper()}_allocation_model.zip"
            )
            self.model = MaskablePPO.load(self.model_path, device="cpu")
            if self.model.action_space.n != 25:
                raise ValueError("Checkpoint must support exactly 25 actions")

    def __call__(self, net, batch, _strategy):
        if self.strategy == "musfico":
            return self._dp(net, batch)
        mapping = {}
        shadow = copy.deepcopy(net)
        for sfc in batch:
            if self.strategy == "greedy":
                algorithm = DAGGreedyAlgorithm()
                algorithm.install_substrate_network(shadow.graph)
                algorithm.install_SFC(sfc)
                if not algorithm.start_algorithm():
                    return None, [], "greedy_no_plan"
                order = list(nx.topological_sort(sfc.graph))
                for name in order:
                    if name in ("src", "dst"):
                        continue
                    node = mapping.get(name, algorithm.route_info[name][0])
                    try:
                        shadow.allocate_microservice(sfc, sfc.vnfs[name], node)
                    except ValueError:
                        return None, [], "greedy_resource"
                    mapping[name] = node
            else:
                # Keep the legacy reverse allocation order and exact observation functions.
                nodes = [n for n in shadow.graph if n != sfc.dst_node] + [sfc.dst_node]
                env_class = (
                    SFC_AllocationEnv if self.strategy == "replic" else SFC_AllocationEnv_DARSPPO
                )
                env = env_class(nodes, [shadow.graph], [sfc], is_training=False)
                if env.observation_space != self.model.observation_space:
                    raise ValueError("Checkpoint observation schema mismatch")
                env.graph = shadow.graph
                env.set_current_sfc(sfc)
                env.latency_used = 0.0
                successors = {"dst": sfc.dst_node}
                for name in reversed(list(nx.topological_sort(sfc.graph))):
                    if name in ("src", "dst"):
                        continue
                    vnf = sfc.vnfs[name]
                    downstream = next(iter(sfc.graph.successors(name)))
                    location = successors[downstream]
                    env.current_vnf = vnf
                    env.current_location = location
                    env.features = env._get_nodes_features(
                        vnf, vnf.get_outcome_interface_bandwidth(), location
                    )
                    observation = env._get_obs()
                    if any(not np.isfinite(value).all() for value in observation.values()):
                        raise ValueError("Non-finite policy observation")
                    if is_shareable(vnf) and name in mapping:
                        node = mapping[name]
                    else:
                        masks = env.action_masks()
                        if not masks.any():
                            return None, [], "policy_no_action"
                        action, _ = self.model.predict(
                            observation, action_masks=masks, deterministic=True
                        )
                        node = nodes[int(action)]
                    try:
                        shadow.allocate_microservice(sfc, vnf, node)
                    except ValueError:
                        return None, [], "policy_resource"
                    mapping[name] = node
                    successors[name] = node
        try:
            trial, latencies = plan_colocated(net, batch, mapping)
        except ValueError as error:
            return None, [], f"physical_validation:{error}"
        return trial, latencies, self.strategy

    def _dp(self, net, batch):
        # MuSFiCO-style DP recurrence generalized to the complete DAG. Keep up to
        # beam_width resource/route states per layer; this bounded search is explicit.
        trial = copy.deepcopy(net)
        latencies = []
        for original in batch:
            sfc = copy.deepcopy(original)
            initial = MAPPO_SFC_Env(
                trial.graph,
                sfc,
                n_regions=1,
                cloud_node=-1,
                share=True,
                shareband=True,
                policy_weight=0,
            )
            states = [initial]
            for _ in initial.vnf_sequence:
                candidates = []
                for state in states:
                    for node in state.graph:
                        result = state._candidate(node)
                        if result is None:
                            continue
                        child = copy.copy(state)
                        child.graph, routes, edges, comp, finish, sync, _, _ = result
                        for field in (
                            "allocated_nodes",
                            "route_info",
                            "edge_routes",
                            "edge_latencies",
                            "comp_latencies",
                            "finish_times",
                        ):
                            setattr(child, field, dict(getattr(state, field)))
                        vnf = child.vnf_sequence[child.current_vnf_idx]
                        child.allocated_nodes[vnf.id] = node
                        child.route_info[vnf.id] = [node]
                        child.edge_routes.update(routes)
                        child.edge_latencies.update(edges)
                        child.comp_latencies[vnf.id] = comp
                        child.finish_times[vnf.id] = finish
                        child.sync_differential = max(child.sync_differential, sync)
                        child.current_location = node
                        child.current_vnf_idx += 1
                        child.total_latency = max(child.finish_times.values())
                        if child.current_vnf_idx == len(child.vnf_sequence):
                            child._finalize_chain()
                            if not child.success:
                                continue
                        try:
                            distance = nx.shortest_path_length(
                                child.graph, node, child.dst_node, weight="latency"
                            )
                        except nx.NetworkXNoPath:
                            continue
                        candidates.append((child.total_latency + 0.25 * distance, child))
                states = [
                    state
                    for _, state in sorted(candidates, key=lambda item: item[0])[: self.beam_width]
                ]
                if not states:
                    return None, [], "dp_no_feasible_state"
            best = min(states, key=lambda state: state.total_latency)
            sfc.edge_routes = best.edge_routes
            sfc.edge_latencies = best.edge_latencies
            sfc.comp_latencies = best.comp_latencies
            trial.deploy_sfc(sfc, best.route_info)
            latencies.append(best.total_latency)
        return trial, latencies, "musfico_dag_dp_beam"

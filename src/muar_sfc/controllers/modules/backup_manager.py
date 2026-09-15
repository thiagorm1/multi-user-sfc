import logging
import random
import time
from collections import defaultdict
from typing import Any

import networkx as nx

from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC

logger = logging.getLogger(__name__)


class BackupManager:
    def __init__(self, args):
        self.sfcs_backups_instatiated: dict[str, list[dict[str, Any]]] = {}
        self.backups_sfc_instantiated: dict[str, str] = {}
        self.backups_activated: list[str] = []

        valid_backup_algs = ["replic", "ga", "vegeta"]
        
        
        try:
            is_target_alg = args.alg in valid_backup_algs
            user_wants_backup = args.backup and (args.ava != "1.0")
            self.alg = args.alg
        except AttributeError:
            is_target_alg = False
            user_wants_backup = False
            self.alg = "none"

        self.backup_activated = user_wants_backup and is_target_alg

        if user_wants_backup and not is_target_alg:
            logger.info(f"Backup proativo desativado. '{self.alg}' não suporta essa estratégia.")

        self.standard_reduction_factor = 1.0

    def identify_obsolete_backups(self) -> list[str]:
        backups_to_remove = []
        if self.alg not in ["vegeta", "ga"]:
            return []

        for backup_id in list(self.backups_sfc_instantiated.keys()):
            if backup_id in self.backups_activated:
                continue
            if random.random() < 0.7:
                backups_to_remove.append(backup_id)

        return backups_to_remove

    def cleanup_internal_state(self, backup_id: str) -> None:
        original_sfc = self.backups_sfc_instantiated.get(backup_id)

        if original_sfc and original_sfc in self.sfcs_backups_instatiated:
            backups_list = self.sfcs_backups_instatiated[original_sfc]
            self.sfcs_backups_instatiated[original_sfc] = [
                b for b in backups_list if b.get("sfc_backup_id") != backup_id
            ]

            if not self.sfcs_backups_instatiated[original_sfc]:
                del self.sfcs_backups_instatiated[original_sfc]

        if backup_id in self.backups_sfc_instantiated:
            del self.backups_sfc_instantiated[backup_id]

        if backup_id in self.backups_activated:
            self.backups_activated.remove(backup_id)

    def register_backup_deployment(
        self, original_sfc_id: str, backup_sfc_id: str, vnf_id: str, route_info: dict
    ) -> None:
        if original_sfc_id not in self.sfcs_backups_instatiated:
            self.sfcs_backups_instatiated[original_sfc_id] = []

        self.sfcs_backups_instatiated[original_sfc_id].append(
            {"sfc_backup_id": backup_sfc_id, "vnf_id": vnf_id, "route_info": route_info}
        )
        self.backups_sfc_instantiated[backup_sfc_id] = original_sfc_id

    def create_backups(self, network: Net2, agent_ref: Any = None) -> tuple[list[Any], str]:
        if not self.backup_activated:
            return [], "none"

        GLOBAL_THRESHOLD = 0.80
        current_utilization = network.get_network_only_processing_utilization()

        if current_utilization > GLOBAL_THRESHOLD:
            logger.warning(f"Criação suspensa. Rede saturada: {current_utilization:.2%}")
            return [], "network_saturated"

        MIN_LIFETIME_FOR_BACKUP = 5.0
        current_time = time.time()
        sfc_lifecycle_map = {}

        for sfc_id, sfc in network.sfc_dict.items():
            try:
                if sfc.is_backup or "backup" in sfc_id:
                    continue
            except AttributeError:
                if "backup" in sfc_id:
                    continue

            try:
                start_t = sfc.arrival_time
            except AttributeError:
                start_t = current_time

            try:
                total_duration = sfc.duration
            except AttributeError:
                total_duration = 100.0

            elapsed = current_time - start_t
            remaining_time = total_duration - elapsed

            if remaining_time < MIN_LIFETIME_FOR_BACKUP:
                continue

            sfc_lifecycle_map[sfc_id] = {
                "timer": start_t,
                "duration": total_duration,
                "remaining": remaining_time,
            }

        backups_mount = []
        strategy_name = "greedy"

        if self.alg == "replic" and agent_ref is not None:
            backups_mount = self.rl_based_strategy(network, sfc_lifecycle_map, agent_ref)
            strategy_name = "rl_based"
        elif self.alg in ["vegeta", "ga"]:
            backups_mount = self.seletive_strategy(network, sfc_lifecycle_map)
            strategy_name = "seletive"
        else:
            backups_mount = self.greedy_strategy(network, sfc_lifecycle_map)
            strategy_name = "greedy"

        self._commit_backup_state(backups_mount)
        return backups_mount, strategy_name

    def _commit_backup_state(self, backups_groups: list[list[Any]]) -> None:
        if not backups_groups:
            return

        for group in backups_groups:
            for backup_sfc in group:
                try:
                    original_id = backup_sfc.original_sfc_id
                    target_vnf = backup_sfc.target_vnf_id
                except AttributeError:
                    continue
                
                try:
                    route_info = backup_sfc.pre_calculated_route
                except AttributeError:
                    route_info = {}

                if original_id and target_vnf:
                    self.register_backup_deployment(
                        original_sfc_id=original_id,
                        backup_sfc_id=backup_sfc.id,
                        vnf_id=target_vnf,
                        route_info=route_info,
                    )

    def _calc_virtual_reliability(
        self, network: Net2, sfc_id: str, pending_backups_sfcs: list[Any]
    ) -> tuple[float, list[dict]]:
        if sfc_id not in network.sfc_route_info:
            return 0.0, []

        route_info = network.sfc_route_info[sfc_id]
        backup_reliability_map = {}

        if sfc_id in self.sfcs_backups_instatiated:
            for b in self.sfcs_backups_instatiated[sfc_id]:
                vnf_id = b.get("vnf_id")
                bk_route = b.get("route_info", {})
                bk_node_list = next(
                    (v for k, v in bk_route.items() if k.endswith("_b") and v), None
                )
                if vnf_id and bk_node_list:
                    node_id = bk_node_list[0]
                    backup_reliability_map[vnf_id] = network.get_node_reliability(node_id)

        for mini_sfc in pending_backups_sfcs:
            try:
                target_vnf = mini_sfc.target_vnf_id
            except AttributeError:
                continue

            if target_vnf and target_vnf not in backup_reliability_map:
                try:
                    route = mini_sfc.pre_calculated_route
                except AttributeError:
                    route = {}

                bk_key = f"{target_vnf}_b"
                if route and bk_key in route and route[bk_key]:
                    node_id = route[bk_key][0]
                    backup_reliability_map[target_vnf] = network.get_node_reliability(node_id)

        node_groups = defaultdict(list)
        for vnf_id, path in route_info.items():
            if vnf_id in ["src", "dst"] or not path:
                continue
            node_groups[path[0]].append(vnf_id)

        total_reliability = 1.0
        candidates = []

        for node_id, vnfs_list in node_groups.items():
            try:
                reliability_primary = network.get_node_reliability(node_id)
            except (AttributeError, KeyError):
                reliability_primary = 1.0

            all_vnfs_protected = True
            prod_failure_backups = 1.0

            for vnf_id in vnfs_list:
                reliability_backup = backup_reliability_map.get(vnf_id, 0.0)
                if reliability_backup > 0.0:
                    prod_failure_backups *= 1.0 - reliability_backup
                else:
                    all_vnfs_protected = False
                    candidates.append(
                        {"vnf_id": vnf_id, "node_rel": reliability_primary, "node_id": node_id}
                    )

            if all_vnfs_protected:
                prob_primary_fail = 1.0 - reliability_primary
                group_reliability = 1.0 - (prob_primary_fail * prod_failure_backups)
            else:
                group_reliability = reliability_primary

            total_reliability *= group_reliability

        sorted_candidates = sorted(candidates, key=lambda x: x["node_rel"])
        return total_reliability, sorted_candidates

    def rl_based_strategy(
        self, network: Net2, sfc_id_duration: dict, agent: Any
    ) -> list[list[Any]]:
        from muar_sfc.algorithms.environments.env_replic import SFC_AllocationEnv

        backups_mount = []
        target_reliability = 0.96
        MAX_BACKUPS_PER_SFC = 4
        
        # OTIMIZAÇÃO O(1): Cópia isolada rasa de TODOS os atributos (incluindo cache e ips).
        # Resolve o KeyError do RL sem acionar o gargalo do deepcopy.
        simulation_graph = nx.Graph()
        for node, data in network.graph.nodes(data=True):
            simulation_graph.add_node(node, **data.copy())
            
        for u, v, data in network.graph.edges(data=True):
            simulation_graph.add_edge(u, v, **data.copy())

        sorted_sfcs = sorted(list(sfc_id_duration.keys()))

        for sfc_id in sorted_sfcs:
            if "backup" in sfc_id or sfc_id not in network.sfc_dict:
                continue
            sfc = network.get_sfc_by_id(sfc_id)
            pending_sfcs_this_cycle = []
            loop_safety_counter = 0

            while len(pending_sfcs_this_cycle) < MAX_BACKUPS_PER_SFC:
                loop_safety_counter += 1
                if loop_safety_counter > 5:
                    break

                current_r, candidates = self._calc_virtual_reliability(
                    network, sfc_id, pending_sfcs_this_cycle
                )

                if current_r >= target_reliability or not candidates:
                    break

                target_info = candidates[0]
                target_vnf = target_info["vnf_id"]
                weak_node = target_info["node_id"]

                mini_sfc = self.create_contextual_mini_sfc(network, sfc, target_vnf, weak_node)
                if not mini_sfc:
                    break

                # OTIMIZAÇÃO: Cópia isolada rasa por iteração (livre de ponteiros indesejados)
                graph_for_env = nx.Graph()
                for node, data in simulation_graph.nodes(data=True):
                    graph_for_env.add_node(node, **data.copy())
                for u, v, data in simulation_graph.edges(data=True):
                    graph_for_env.add_edge(u, v, **data.copy())

                self._enrich_graph_with_mobility(graph_for_env, network, mini_sfc)

                valid_types = ["server", "mobile_device"]
                all_servers = [
                    n for n, d in graph_for_env.nodes(data=True) if d.get("type") in valid_types
                ]

                if not all_servers:
                    break

                env = SFC_AllocationEnv(
                    valid_nodes=all_servers,
                    list_graph=[graph_for_env],
                    list_sfc=[mini_sfc],
                    is_training=False,
                )

                primary_nodes_used = set()
                original_route_info = network.sfc_route_info.get(sfc_id, {})
                for vnf_p, path_p in original_route_info.items():
                    if vnf_p not in ["src", "dst"] and path_p:
                        primary_nodes_used.add(path_p[0])

                for pending_bk in pending_sfcs_this_cycle:
                    try:
                        for b_path in pending_bk.pre_calculated_route.values():
                            if b_path:
                                primary_nodes_used.add(b_path[0])
                    except AttributeError:
                        continue

                forbidden = []
                for node in primary_nodes_used:
                    forbidden.append(node)
                    if isinstance(node, float):
                        forbidden.append(int(node))
                    elif isinstance(node, int):
                        forbidden.append(float(f"{node}.1"))

                env.set_forbidden_nodes(forbidden)
                agent.install_substrate_network(graph_for_env)
                agent.install_SFC(mini_sfc)

                try:
                    success = agent.start_algorithm(env)
                except Exception as e:
                    logger.exception(f"Erro no Agente RL para {sfc_id}: {e}")
                    success = False

                if success:
                    route_info_backup = agent.get_route_info()
                    if not route_info_backup:
                        break

                    mini_sfc.pre_calculated_route = route_info_backup
                    pending_sfcs_this_cycle.append(mini_sfc)
                    backups_mount.append([mini_sfc])
                    self._apply_virtual_reservation(simulation_graph, mini_sfc, route_info_backup)
                else:
                    break

        return backups_mount

    def _apply_virtual_reservation(self, graph: Any, sfc: SFC, route_info: dict) -> None:
        for vnf_name, path in route_info.items():
            if vnf_name in ["src", "dst"] or "virt" in vnf_name or not path:
                continue

            node_id = path[0]
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                vnf_obj = sfc.get_vnf_by_id(vnf_name)
                if vnf_obj:
                    node["cpu_used"] = node.get("cpu_used", 0) + vnf_obj.get_cpu_request()
                    node["cache_used"] = node.get("cache_used", 0) + vnf_obj.get_cache_request()

            if len(path) > 1:
                vnf_obj = sfc.get_vnf_by_id(vnf_name)
                bw_req = vnf_obj.get_outcome_interface_bandwidth() if vnf_obj else 0
                for u, v in zip(path[:-1], path[1:], strict=False):
                    if graph.has_edge(u, v):
                        graph.edges[u, v]["bandwidth_used"] = (
                            graph.edges[u, v].get("bandwidth_used", 0) + bw_req
                        )

    def _enrich_graph_with_mobility(self, graph, network, mini_sfc):
        try:
            mobile_node_id = mini_sfc.mobile_node
            closer_router_id = mini_sfc.closer_router
        except AttributeError:
            mobile_node_id = None
            closer_router_id = None

        if mobile_node_id and mobile_node_id in network.md_graph and mobile_node_id not in graph:
            graph.add_node(mobile_node_id, **network.md_graph.nodes[mobile_node_id])
            if closer_router_id and closer_router_id in graph:
                w_free = max(
                    0.0,
                    graph.nodes[closer_router_id].get("w_channel_capacity", 0.0)
                    - graph.nodes[closer_router_id].get("w_channel_used", 0.0),
                )
                graph.add_edge(
                    mobile_node_id,
                    closer_router_id,
                    bandwidth_capacity=w_free,
                    bandwidth_used=0.0,
                    latency=1,
                )

    def greedy_strategy(self, network: Net2, sfc_id_duration: dict) -> list[list[Any]]:
        backups_mount = []
        sfcs_id = list(sfc_id_duration.keys())
        random.shuffle(sfcs_id)

        for sfc_id in sfcs_id:
            parts = sfc_id.split("_")
            if (
                (len(parts) > 2 and parts[2] == "backup")
                or sfc_id not in network.sfc_dict
                or sfc_id in self.sfcs_backups_instatiated
            ):
                continue

            if random.random() < 0.6:
                continue

            current_time = time.time()
            rem = max(
                10,
                sfc_id_duration[sfc_id]["duration"]
                - (current_time - sfc_id_duration[sfc_id]["timer"])
                + 10,
            )
            sfc = network.get_sfc_by_id(sfc_id)

            new_vnfs_dict = []
            for info in sfc.vnfs_dict:
                new_info = info.copy()
                for aspect in ["CPU", "cache", "in_bw", "out_bw"]:
                    new_info[aspect] = info[aspect] * self.standard_reduction_factor
                new_info["name"] = info["name"] + "_b"
                new_vnfs_dict.append(new_info)

            try:
                lat_req = sfc.latency_request
            except AttributeError:
                lat_req = 10

            player_dict = {
                "name": f"{parts[0]}_{parts[1]}_backup_{parts[2]}_{parts[3]}",
                "vnf_list": new_vnfs_dict,
                "bandwidth": sfc.input_throughput,
                "src_node": sfc.src.substrate_node,
                "dst_node": sfc.dst.substrate_node,
                "duration": rem,
                "latency": lat_req,
            }

            new_sfc = SFCGenerator(player_dict).generate()
            new_sfc.original_sfc_id = sfc_id
            new_sfc.is_backup = True
            backups_mount.append([new_sfc])

        return backups_mount

    def seletive_strategy(
        self, network: Net2, sfc_id_duration: dict, threshold: float = 0
    ) -> list[list[Any]]:
        backups_mount = []
        nodes_fail_p = network.get_servers_reliability_dict()
        nodes_candidates = {n: r for n, r in nodes_fail_p.items() if r > threshold}

        if not nodes_candidates:
            return []

        sorted_reliable_servers = sorted(nodes_candidates, key=nodes_candidates.get, reverse=True)
        node_resources = {
            n: {
                "cpu": network.get_node_cpu_free(n),
                "cache": network.get_node_cache_free(n),
                "is_active": network.get_node_is_active(n),
            }
            for n in sorted_reliable_servers
        }
        link_usage = defaultdict(float)

        def get_path(src, tgt, bw):
            if src == tgt:
                return [src]

            def filt(u, v):
                edge = network.graph[u][v]
                return (
                    edge["bandwidth_capacity"]
                    - edge["bandwidth_used"]
                    - link_usage[frozenset((u, v))]
                ) >= bw

            try:
                return nx.dijkstra_path(
                    nx.subgraph_view(network.graph, filter_edge=filt), src, tgt, weight="latency"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        for sfc_id in list(sfc_id_duration.keys()):
            if "backup" in sfc_id:
                continue
            try:
                sfc = network.get_sfc_by_id(sfc_id)
            except KeyError:
                continue

            for vnf_info in sfc.vnfs_dict:
                v_id = vnf_info["name"]
                if v_id in ["src", "dst"] or "virt" in v_id:
                    continue

                sfc_rf = network.sfc_route_info.get(sfc_id, {})
                if v_id not in sfc_rf or not sfc_rf[v_id]:
                    continue

                orig_loc = sfc_rf[v_id][0]
                cpu_req, cache_req, bw_req = (
                    vnf_info["CPU"],
                    vnf_info["cache"],
                    vnf_info.get("out_bw", 0.0),
                )

                target_server = next(
                    (
                        c
                        for c in sorted_reliable_servers
                        if str(c) != str(orig_loc)
                        and node_resources[c]["is_active"]
                        and node_resources[c]["cpu"] >= cpu_req
                        and node_resources[c]["cache"] >= cache_req
                    ),
                    None,
                )

                if not target_server:
                    continue

                try:
                    lat_req = sfc.latency_request
                except AttributeError:
                    lat_req = 10

                dst_n, src_n, lat_req_adj = self.escolher_src_dst(
                    sfc_rf, v_id, lat_req
                )
                if lat_req_adj is None or lat_req_adj < 0:
                    continue

                p_in = get_path(src_n, target_server, bw_req)
                if not p_in:
                    continue

                for u, v in zip(p_in[:-1], p_in[1:], strict=False):
                    link_usage[frozenset((u, v))] += bw_req

                p_out = get_path(target_server, dst_n, bw_req)
                if not p_out:
                    for u, v in zip(p_in[:-1], p_in[1:], strict=False):
                        link_usage[frozenset((u, v))] -= bw_req
                    continue

                for u, v in zip(p_out[:-1], p_out[1:], strict=False):
                    link_usage[frozenset((u, v))] += bw_req

                node_resources[target_server]["cpu"] -= cpu_req
                node_resources[target_server]["cache"] -= cache_req

                s_parts = sfc_id.split("_")
                p0, p1, p2, p3 = s_parts[0], s_parts[1], s_parts[2], s_parts[3]
                new_sfc_name = f"{p0}_{p1}_backup_{v_id}_{p2}_{p3}"

                try:
                    closer_router = sfc.closer_router
                except AttributeError:
                    closer_router = None

                new_sfc = SFCGenerator(
                    {
                        "name": new_sfc_name,
                        "vnf_list": [
                            {
                                "type": 2,
                                "name": "src_virt",
                                "CPU": 0,
                                "cache": 0,
                                "in_bw": 0,
                                "out_bw": 0,
                                "latency": 0,
                                "location": src_n,
                            },
                            {
                                "type": 2,
                                "name": v_id + "_b",
                                "CPU": cpu_req,
                                "cache": cache_req,
                                "in_bw": 0,
                                "out_bw": bw_req,
                                "latency": 0,
                                "original_loc": orig_loc,
                                "original_sfc": sfc_id,
                            },
                            {
                                "type": 2,
                                "name": "dst_virt",
                                "CPU": 0,
                                "cache": 0,
                                "in_bw": 0,
                                "out_bw": 0,
                                "latency": 0,
                                "location": dst_n,
                            },
                        ],
                        "bandwidth": sfc.input_throughput,
                        "src_node": src_n,
                        "dst_node": dst_n,
                        "duration": max(
                            10,
                            sfc_id_duration[sfc_id]["duration"]
                            - (time.time() - sfc_id_duration[sfc_id]["timer"])
                            + 10,
                        ),
                        "latency": lat_req_adj,
                        "closer_router": closer_router,
                    }
                ).generate()

                new_sfc.original_sfc_id = sfc_id
                new_sfc.is_backup = True
                new_sfc.target_vnf_id = v_id
                new_sfc.pre_calculated_route = {
                    "src_virt": p_in,
                    v_id + "_b": p_out,
                    "dst_virt": [],
                }
                backups_mount.append([new_sfc])

        return backups_mount

    def escolher_src_dst(self, dicionario: dict, vnf_escolhida: str, latency_limit: int = 10):
        chaves = [k for k in dicionario if k not in ["src", "dst"]]
        if vnf_escolhida not in chaves:
            return None, None, None
        idx = chaves.index(vnf_escolhida)

        if idx == 0:
            dst, src, lat_d = chaves[idx], chaves[idx + 1], len(dicionario[chaves[idx + 1]]) - 1
        elif idx == len(chaves) - 1:
            dst, src, lat_d = chaves[idx - 1], chaves[idx], len(dicionario[vnf_escolhida]) - 1
        else:
            dst, src, lat_d = (
                chaves[idx - 1],
                chaves[idx + 1],
                (len(dicionario[chaves[idx + 1]]) - 1) + (len(dicionario[vnf_escolhida]) - 1),
            )

        return dicionario[dst][0], dicionario[src][0], latency_limit - lat_d

    def create_contextual_mini_sfc(
        self, network: Net2, original_sfc: SFC, vnf_to_replicate_id: str, primary_node_id: str
    ) -> SFC | None:
        target_info = next(
            (i for i in original_sfc.vnfs_dict if i["name"] == vnf_to_replicate_id), None
        )
        if not target_info or not network.sfc_route_info.get(original_sfc.id):
            return None

        cur_v = original_sfc.get_vnf_by_id(vnf_to_replicate_id)
        p_v = original_sfc.get_previous_vnf(cur_v)
        p_n = (
            original_sfc.src.substrate_node
            if p_v.id == "src"
            else network.sfc_route_info[original_sfc.id][p_v.id][0]
        )
        n_v = original_sfc.get_next_vnf(cur_v)
        n_n = (
            original_sfc.dst.substrate_node
            if n_v.id == "dst"
            else network.sfc_route_info[original_sfc.id][n_v.id][0]
        )

        try:
            arr_time = original_sfc.arrival_time
        except AttributeError:
            arr_time = time.time()
            
        try:
            lat_req = original_sfc.latency_request
        except AttributeError:
            lat_req = 10
            
        try:
            closer_r = original_sfc.closer_router
        except AttributeError:
            closer_r = None
            
        try:
            mobile_n = original_sfc.dst_node
        except AttributeError:
            mobile_n = None

        mini_sfc = SFCGenerator(
            {
                "name": f"{original_sfc.id}_backup_{vnf_to_replicate_id}",
                "vnf_list": [
                    {
                        "type": 2,
                        "name": "src_virt",
                        "CPU": 0,
                        "cache": 0,
                        "in_bw": 0,
                        "out_bw": 0,
                        "latency": 0,
                        "location": p_n,
                    },
                    {
                        "type": 2,
                        "name": vnf_to_replicate_id + "_b",
                        "CPU": target_info["CPU"],
                        "cache": target_info["cache"],
                        "in_bw": 0,
                        "out_bw": 0,
                        "latency": 0,
                        "original_sfc": original_sfc.id,
                    },
                    {
                        "type": 2,
                        "name": "dst_virt",
                        "CPU": 0,
                        "cache": 0,
                        "in_bw": 0,
                        "out_bw": 0,
                        "latency": 0,
                        "location": n_n,
                    },
                ],
                "bandwidth": original_sfc.input_throughput,
                "src_node": p_n,
                "dst_node": n_n,
                "duration": max(10, original_sfc.duration - (time.time() - arr_time) + 10),
                "latency": lat_req,
                "closer_router": closer_r,
                "mobile_node": mobile_n,
            }
        ).generate()

        mini_sfc.original_sfc_id = original_sfc.id
        mini_sfc.is_backup = True
        mini_sfc.target_vnf_id = vnf_to_replicate_id

        try:
            mini_sfc.session_id = original_sfc.session_id
        except AttributeError:
            mini_sfc.session_id = original_sfc.id.split("_")[-1]
            
        mini_sfc.src_virt = p_n
        mini_sfc.dst_virt = n_n

        return mini_sfc

    def get_backups_instantiated_q(self) -> int:
        return len(self.backups_sfc_instantiated) * (
            len(self.sfcs_backups_instatiated) if self.alg == "ga" else 4
        )
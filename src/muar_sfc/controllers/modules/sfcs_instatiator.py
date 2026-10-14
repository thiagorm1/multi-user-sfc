import copy
import time

import networkx as nx
from loguru import logger

from muar_sfc.algorithms.environments.env_da_rsppo import SFC_AllocationEnv_DARSPPO
from muar_sfc.algorithms.environments.env_replic import SFC_AllocationEnv as SFC_AllocationEnv_SCRC
from muar_sfc.algorithms.environments.environment import SFC_AllocationEnv
from muar_sfc.algorithms.environments.hephaestus_env import SFC_AllocationEnv_hephaestus
from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
)
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC
from muar_sfc.utils.network_utils import check_vnf_reusability

SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")


class SFCInstatiator:
    """
    Controlador de Admissão e Roteamento.
    Isola recursos e contadores aninhados em um Shadow State para testes em lote,
    garantindo que a IA veja o consumo sequencial sem gargalar a CPU.
    """

    def __init__(self, alg, args=None):
        self.alg = alg
        self.args = args
        self.verbose = True
        self.env = None

    def _create_shadow_graph(self, substrate_network: Net2) -> nx.Graph:
        """Copia recursos e contadores aninhados sem alterar a rede física."""
        return copy.deepcopy(substrate_network.graph)

    def search_solution(
        self, sfc_list: list[SFC], substrate_network: Net2, is_backup=False
    ) -> tuple[dict, bool]:
        default_solution_format = {
            sfc.id: {"route_info": None, "latency": None, "run_duration": None} for sfc in sfc_list
        }

        self.alg.clear_all()

        # OTIMIZAÇÃO: Adeus deepcopy.
        infra_to_use = self._create_shadow_graph(substrate_network)

        # EAFP: Segurança contra atributos inexistentes
        try:
            dst_node_id = sfc_list[0].dst_node
            if dst_node_id in substrate_network.md_graph:
                self.add_mobile_user_to_graph(infra_to_use, substrate_network, sfc_list)
        except AttributeError:
            pass

        valid_nodes = [
            node for node, data in infra_to_use.nodes(data=True) if data.get("type") != "router"
        ]

        # Match/Case moderno do Python 3.10+
        match type(self.alg).__name__:
            case "Kuririn":
                self.env = SFC_AllocationEnv(
                    valid_nodes=valid_nodes,
                    list_graph=[infra_to_use],
                    list_sfc=[sfc_list[0]],
                    is_training=False,
                )
            case "DARSPPO":
                self.env = SFC_AllocationEnv_DARSPPO(
                    valid_nodes=valid_nodes,
                    list_graph=[infra_to_use],
                    list_sfc=[sfc_list[0]],
                    is_training=False,
                )
            case "hephaestus":
                self.env = SFC_AllocationEnv_hephaestus(
                    valid_nodes=valid_nodes,
                    list_graph=[infra_to_use],
                    list_sfc=[sfc_list[0]],
                    is_training=False,
                )
            case "REPLIC":
                self.env = SFC_AllocationEnv_SCRC(
                    valid_nodes=valid_nodes,
                    list_graph=[infra_to_use],
                    list_sfc=[sfc_list[0]],
                    is_training=False,
                )

        return self.sequential_search(self.alg, sfc_list, infra_to_use, default_solution_format)

    def sequential_search(
        self, algorithm, sfc_list: list[SFC], graph: nx.Graph, solution_format: dict
    ) -> tuple[dict, bool]:
        search_success = True
        total_elapsed_ms = 0.0
        approved_sfcs = []
        rejected_sfcs = []

        for sfc in sfc_list:
            # OTIMIZAÇÃO VITAL: Passamos a referência direta do shadow graph.
            # Como a simulação (dry-run) deduz recursos diretamente dele no laço,
            # a IA recebe a topologia devidamente atualizada a cada passo do lote.
            algorithm.install_substrate_network(graph)
            algorithm.install_SFC(sfc)

            try:
                algorithm.allow_md_host = bool(self.args.allow_md_host)
            except AttributeError:
                algorithm.allow_md_host = False

            s_time = time.time()
            alg_success = False

            if type(algorithm).__name__ in ("Kuririn", "DARSPPO", "hephaestus", "REPLIC"):
                if self.env:
                    if type(algorithm).__name__ == "REPLIC":
                        alg_success = algorithm.start_algorithm(self.env, args=self.args)
                    else:
                        alg_success = algorithm.start_algorithm(self.env)
                else:
                    logger.error(
                        f"Tentativa de usar {algorithm.name} sem um ambiente inicializado."
                    )
            elif type(algorithm).__name__ == "MAPPOAlgorithm":
                alg_success = algorithm.start_algorithm(args=self.args)
            else:
                alg_success = algorithm.start_algorithm()

            e_time = time.time()
            total_elapsed_ms += (e_time - s_time) * 1000

            total_latency = comp_latency = comm_latency = None

            if alg_success:
                try:
                    route_info = algorithm.get_route_info()
                    # DRY-RUN: Se passar na validação estrita, o grafo virtual é deduzido
                    total_latency, comp_latency, comm_latency, res_info = self._simulate_dry_run(
                        graph, sfc, route_info
                    )
                    approved_sfcs.append(sfc.id)

                except ValueError as ve:
                    logger.warning(
                        f" ⚠️ [DRY-RUN] Controle de Admissão bloqueou SFC {sfc.id}: {ve}"
                    )
                    algorithm.handle_failure()
                    alg_success = False
                    rejected_sfcs.append(sfc.id)
            else:
                rejected_sfcs.append(sfc.id)

            solution_format[sfc.id] = {
                "route_info": algorithm.get_route_info() if alg_success else None,
                "latency": total_latency,
                "comp_latency": comp_latency,
                "comm_latency": comm_latency,
                "run_duration": e_time - s_time,
                "resource_info": res_info if alg_success else 0,
            }

            if not alg_success:
                search_success = False

        if self.verbose:
            logger.info(
                f"Alg '{algorithm.name}' processou lote [{', '.join(s.id for s in sfc_list)}] "
                f"em {round(total_elapsed_ms, 3)} ms. Status: {search_success}"
            )
            if approved_sfcs:
                logger.debug(f"Aprovadas: {', '.join(approved_sfcs)}")
            if rejected_sfcs:
                logger.debug(f"Rejeitadas: {', '.join(rejected_sfcs)}")

        return solution_format, search_success

    def add_mobile_user_to_graph(
        self, graph: nx.Graph, substrate_network: Net2, sfc_list: list[SFC]
    ) -> None:
        mobile_device_id = sfc_list[0].dst_node

        try:
            closer_router = sfc_list[0].closer_router
        except AttributeError:
            closer_router = None

        try:
            md_info = substrate_network.md_graph.nodes[mobile_device_id].copy()
        except KeyError:
            return

        graph.add_node(mobile_device_id, **md_info)

        if closer_router and closer_router in graph.nodes:
            router = graph.nodes[closer_router]
            wireless_free = router.get("w_channel_capacity", 0.0) - router.get(
                "w_channel_used", 0.0
            )

            graph.add_edge(
                mobile_device_id,
                closer_router,
                bandwidth_capacity=wireless_free,
                bandwidth_used=0.00,
                latency=1,
                services_in_transit={},
            )

    def _simulate_dry_run(self, graph: nx.Graph, sfc: SFC, route_info: dict):
        """
        Validação do Shadow State.
        Usa o Grafo Virtual para confirmar se a IA tomou uma decisão matematicamente possível.
        """
        if getattr(sfc, "edge_routes", None):
            from muar_sfc.core.orchestration.placement import (
                reserve_bandwidth,
                reserve_node,
                stream_key,
            )

            used_cpu = 0.0
            for name, vnf in sfc.vnfs.items():
                cpu, _ = reserve_node(graph, sfc, vnf, route_info[name][0], self.args.share)
                used_cpu += cpu
            for (u, v), path in sfc.edge_routes.items():
                key = stream_key(sfc, u, v, self.args.shareband)
                for left, right in zip(path[:-1], path[1:], strict=True):
                    reserve_bandwidth(
                        graph.edges[left, right],
                        key,
                        sfc.graph.edges[u, v]["bandwidth"],
                        self.args.shareband,
                    )
            latency = sfc.get_critical_path_latency(sfc.edge_latencies, sfc.comp_latencies)
            if latency > sfc.get_latency_request():
                raise ValueError("Caminho crítico excedeu o SLA.")
            return (
                round(latency, 2),
                round(sum(sfc.comp_latencies.values()), 2),
                round(sum(sfc.edge_latencies.values()), 2),
                used_cpu,
            )

        try:
            session_id = sfc.session_id
        except AttributeError:
            session_id = sfc.id.split("_")[-1]

        total_latency = 0.0
        total_resources_consumed = 0.0
        tsaber = {"computacao": {}, "comunicacao": {}}

        def allocate_microservice(vnf, node_id, sess_id):
            service_id = vnf.id
            service_key = (service_id, sess_id)
            cpu_req = vnf.get_cpu_request()
            cache_req = vnf.get_cache_request()
            node = graph.nodes[node_id]

            latency = calculate_computational_latency(graph, node_id, vnf)
            if node_id == 0:
                return 0, 0.0

            if node.get("type") not in ["server", "mobile_device"]:
                raise ValueError(f"Recurso {node_id} não é um servidor.")

            clean_id = service_id.replace("_b", "")
            compatible = False

            # Delegação SRP: A verificação de reuso agora é unificada e livre de falsos positivos
            if self.is_shareable(service_id) or self.is_shareable(clean_id):
                services_dict = node.get("services", {})
                if check_vnf_reusability(services_dict, clean_id, sess_id):
                    compatible = True

            node.setdefault("services", {})
            if service_key in node["services"]:
                node["services"][service_key]["copys"] += 1
                if not self.is_shareable(service_id):
                    if node.get("cpu_used", 0) + cpu_req > node.get("cpu_capacity", 0):
                        node["services"][service_key]["copys"] -= 1
                        raise ValueError(f"CPU insuficiente no nó {node_id}")
                    if node.get("cache_used", 0) + cache_req > node.get("cache_capacity", 0):
                        node["services"][service_key]["copys"] -= 1
                        raise ValueError(f"Cache insuficiente no nó {node_id}")

                node["cpu_used"] = node.get("cpu_used", 0) + cpu_req
                node["cache_used"] = node.get("cache_used", 0) + cache_req
                return latency, cpu_req

            cost_c = 0.0 if compatible else cpu_req
            cost_ca = 0.0 if compatible else cache_req

            if node.get("cpu_used", 0) + cost_c > node.get("cpu_capacity", 0) + 1e-5:
                raise ValueError(f"CPU insuficiente no nó {node_id}")
            if node.get("cache_used", 0) + cost_ca > node.get("cache_capacity", 0) + 1e-5:
                raise ValueError(f"Cache insuficiente no nó {node_id}")

            node["services"][service_key] = {"cpu": cost_c, "cache": cost_ca, "copys": 1}
            node["cpu_used"] = node.get("cpu_used", 0) + cost_c
            node["cache_used"] = node.get("cache_used", 0) + cost_ca

            if self.is_shareable(service_id) or self.is_shareable(clean_id):
                node.setdefault("reuse", []).append(vnf)

            return latency, cost_c

        def allocate_bandwidth(node1, node2, vnf, ms_name):
            bw_req = vnf.get_outcome_interface_bandwidth()
            latency = calculate_latency_betwen_nodes(graph, node1, node2, vnf)
            edge = graph.edges[node1, node2]

            if edge.get("bandwidth_used", 0) + bw_req > edge.get("bandwidth_capacity", 0):
                raise ValueError(f"Banda insuficiente entre {node1} e {node2}")

            edge.setdefault("services_in_transit", {})
            if ms_name in edge["services_in_transit"]:
                edge["services_in_transit"][ms_name]["copys"] += 1
            else:
                edge["services_in_transit"][ms_name] = {"copys": 1, "bw_used": 0.0}

            edge["services_in_transit"][ms_name]["bw_used"] += bw_req
            edge["bandwidth_used"] = edge.get("bandwidth_used", 0) + bw_req
            return latency

        is_dag_sfc = hasattr(sfc, "is_dag") and sfc.is_dag() and sfc.graph.number_of_edges() > 0

        for ms_name, path in route_info.items():
            if ms_name in ["src", "dst"]:
                continue

            vnf = sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]

            comp_latency, res_consumed = allocate_microservice(vnf, node_allocated, session_id)
            total_latency += comp_latency
            total_resources_consumed += res_consumed

            tsaber["computacao"][ms_name] = {"latencia_comp": comp_latency}

            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:], strict=False):
                    comm_latency = allocate_bandwidth(u, v, vnf, ms_name)
                    total_latency += comm_latency
                    tsaber["comunicacao"].setdefault(ms_name, []).append(
                        {"latencia_comm": comm_latency}
                    )

        # Validação específica para DAG com Bifurcação e Sincronização (MUAR)
        if is_dag_sfc:
            fork_nodes = [n for n, deg in sfc.graph.out_degree() if deg > 1]
            join_nodes = [n for n, deg in sfc.graph.in_degree() if deg > 1]

            if fork_nodes and join_nodes:
                fork_id, join_id = fork_nodes[0], join_nodes[0]
                # Encontra todos os caminhos entre o ponto de bifurcação e o ponto de junção
                branch_paths = list(nx.all_simple_paths(sfc.graph, source=fork_id, target=join_id))

                if len(branch_paths) >= 2:
                    branch_lats = []
                    for b_path in branch_paths:
                        b_lat = 0.0
                        for v_name in b_path[1:-1]:
                            b_lat += tsaber["computacao"].get(v_name, {}).get("latencia_comp", 0.0)
                            comm_list = tsaber["comunicacao"].get(v_name, [])
                            b_lat += sum(item["latencia_comm"] for item in comm_list)
                        branch_lats.append(b_lat)

                    delta_sync = abs(branch_lats[0] - branch_lats[1])
                    sync_tol = getattr(sfc, "sync_tolerance", 5.0)

                    if delta_sync > sync_tol:
                        raise ValueError(
                            f"Violação de Sincronização Inter-Modal: "
                            f"delta_sync={delta_sync:.2f}ms "
                            f"> tolerância={sync_tol}ms"
                        )

                    # No DAG, os ramos bifurcados rodam em paralelo:
                    # subtrai o ramo mais rápido da soma total (latência
                    # ponta a ponta é dada pelo caminho crítico)
                    diff_to_subtract = min(branch_lats[0], branch_lats[1])
                    total_latency = max(0.0, total_latency - diff_to_subtract)

        tot_comp = sum(d["latencia_comp"] for d in tsaber["computacao"].values())
        tot_comm = sum(
            item["latencia_comm"] for items in tsaber["comunicacao"].values() for item in items
        )

        return (
            round(total_latency, 2),
            round(tot_comp, 2),
            round(tot_comm, 2),
            round(total_resources_consumed, 4),
        )

    def is_shareable(self, service_name: str) -> bool:
        try:
            share_val = self.args.share
            share_enabled = (
                share_val if isinstance(share_val, bool) else str(share_val).lower() == "y"
            )
        except AttributeError:
            share_enabled = False

        return service_name.startswith(SHAREABLE_PREFIXES) if share_enabled else False

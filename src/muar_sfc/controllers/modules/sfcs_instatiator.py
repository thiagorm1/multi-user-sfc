# import copy
# import time

# from loguru import logger

# from muar_sfc.algorithms.darsppo import DARSPPO
# from muar_sfc.algorithms.environments.env_da_rsppo import SFC_AllocationEnv_DARSPPO
# from muar_sfc.algorithms.environments.env_replic import SFC_AllocationEnv as SFC_AllocationEnv_SCRC

# # Environments
# from muar_sfc.algorithms.environments.environment import SFC_AllocationEnv
# from muar_sfc.algorithms.environments.hephaestus_env import SFC_AllocationEnv_hephaestus
# from muar_sfc.algorithms.hephaestus import hephaestus
# from muar_sfc.algorithms.kuririn import Kuririn
# from muar_sfc.algorithms.networkUtils import (
#     calculate_computational_latency,
#     calculate_latency_betwen_nodes,
# )
# from muar_sfc.algorithms.replic import REPLIC

# # Core e Classes
# from muar_sfc.core.net_v2 import Net2
# from muar_sfc.core.sfc import SFC

# # Algoritmos e Utils
# from muar_sfc.utils.k_shortest_paths import k_shortest_paths

# SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")


# class SFCInstatiator:
#     def __init__(self, alg, args=None):
#         self.alg = alg
#         self.args = args
#         self.sfc_list = []
#         self.sfc_queue = []
#         self.sfcs_routing_info = {}
#         self.sfcs_that_deployed = []
#         self.session_id = None
#         self.current_graph = None
#         self.sfcs_that_crashed = []
#         self.verbose = True
#         self.env = None
#         self.list_graph = []
#         self.lists_sfcs = []

#     # ==========================================
#     # Main Search Methods
#     # ==========================================

#     def search_solution(self, sfc_list, substrate_network: Net2, is_backup=False):
#         default_solution_format = {
#             sfc.id: {"route_info": None, "latency": None, "run_duration": None} for sfc in sfc_list
#         }

#         algorithm = self.alg
#         algorithm.clear_all()

#         infra_to_use = copy.deepcopy(substrate_network.graph)
#         graph = infra_to_use

#         dst_node_id = sfc_list[0].dst_node
#         if dst_node_id in substrate_network.md_graph:
#             self.add_mobile_user_to_graph(graph, substrate_network, sfc_list)

#         valid_nodes = [node for node in graph.nodes() if graph.nodes[node]["type"] != "router"]

#         if isinstance(algorithm, Kuririn):
#             self.env = SFC_AllocationEnv(
#                 valid_nodes=valid_nodes,
#                 list_graph=[graph],
#                 list_sfc=[sfc_list[0]],
#                 is_training=False,
#             )
#         elif isinstance(algorithm, DARSPPO):
#             self.env = SFC_AllocationEnv_DARSPPO(
#                 valid_nodes=valid_nodes,
#                 list_graph=[graph],
#                 list_sfc=[sfc_list[0]],
#                 is_training=False,
#             )
#         elif isinstance(algorithm, hephaestus):
#             self.env = SFC_AllocationEnv_hephaestus(
#                 valid_nodes=valid_nodes,
#                 list_graph=[graph],
#                 list_sfc=[sfc_list[0]],
#                 is_training=False,
#             )
#         elif isinstance(algorithm, REPLIC):
#             self.env = SFC_AllocationEnv_SCRC(
#                 valid_nodes=valid_nodes,
#                 list_graph=[graph],
#                 list_sfc=[sfc_list[0]],
#                 is_training=False,
#             )

#         solution, is_success = self.sequential_search(
#             algorithm, sfc_list, infra_to_use, default_solution_format
#         )

#         return solution, is_success

#     def sequential_search(
#         self, algorithm, sfc_list: list[SFC], graph: object, solution_format, graph_backup=None
#     ) -> tuple[dict, bool]:
#         search_success = True

#         total_elapsed_ms = 0.0
#         approved_sfcs = []
#         rejected_sfcs = []

#         for sfc in sfc_list:
#             algorithm.install_substrate_network(copy.deepcopy(graph))
#             algorithm.install_SFC(sfc)

#             # REFATORAÇÃO: EAFP Estrito para leitura de propriedades
#             try:
#                 # Caso self.args seja um Pydantic Settings no modelo moderno (booleano)
#                 algorithm.allow_md_host = bool(self.args.allow_md_host)
#             except AttributeError:
#                 algorithm.allow_md_host = False

#             s = time.time()
#             alg_success = False

#             if isinstance(algorithm, (Kuririn, DARSPPO, hephaestus, REPLIC)):
#                 if self.env:
#                     if isinstance(algorithm, REPLIC):
#                         alg_success = algorithm.start_algorithm(self.env, args=self.args)
#                     else:
#                         alg_success = algorithm.start_algorithm(self.env)
#                 else:
#                     logger.error(f"Tentativa de usar {algorithm.name} sem um ambiente inicializado.")
#                     alg_success = False
#             else:
#                 alg_success = algorithm.start_algorithm()

#             s2 = time.time()
#             total_elapsed_ms += (s2 - s) * 1000

#             total_latency = None
#             comp_latency = None
#             comm_latency = None

#             if alg_success:
#                 try:
#                     route_info = algorithm.get_route_info()
#                     total_latency, comp_latency, comm_latency, res_info = self.submit_solution(
#                         graph, sfc, route_info
#                     )
#                     approved_sfcs.append(sfc.id)

#                 # O ValueError é a exceção oficial de "Rede Saturada" (Simulação válida)
#                 except ValueError as ve:
#                     logger.warning(f"Falha de admissão (recursos insuficientes) para SFC {sfc.id}: {ve}")
#                     algorithm.handle_failure()
#                     alg_success = False
#                     rejected_sfcs.append(sfc.id)

#                 # REFATORAÇÃO: O bloco `except Exception` genérico foi obliterado daqui.
#                 # Se o algoritmo retornar um grafo corrompido, o Python lançará a exceção na tela.
#                 # Isso protege os resultados científicos da sua simulação contra poluição por bugs.
#             else:
#                 rejected_sfcs.append(sfc.id)

#             solution_format[sfc.id] = {
#                 "route_info": algorithm.get_route_info(),
#                 "latency": total_latency,
#                 "comp_latency": comp_latency,
#                 "comm_latency": comm_latency,
#                 "run_duration": s2 - s,
#                 "resource_info": res_info if alg_success else 0,
#             }

#             if not alg_success:
#                 search_success = False

#         if self.verbose:
#             sfc_names = ", ".join([s.id for s in sfc_list])
#             logger.info(f"Alg '{self.alg.name}' processou o lote [{sfc_names}] no tempo de {round(total_elapsed_ms, 3)} ms. {search_success}")

#             status_msg = []
#             if approved_sfcs:
#                 status_msg.append(f"Aprovadas: {', '.join(approved_sfcs)}")
#             if rejected_sfcs:
#                 status_msg.append(f"Rejeitadas: {', '.join(rejected_sfcs)}")

#             logger.debug(f"Status lógico do lote -> {' | '.join(status_msg)}")

#         return solution_format, search_success

#     # ==========================================
#     # Graph Manipulation & Allocation
#     # ==========================================

#     def add_mobile_user_to_graph(self, graph, substrate_network, sfc_list):
#         mobile_device_id = sfc_list[0].dst_node

#         try:
#             closer_router = sfc_list[0].closer_router
#         except AttributeError:
#             closer_router = None

#         md_info = copy.deepcopy(substrate_network.md_graph._node[mobile_device_id])

#         graph.add_node(
#             mobile_device_id,
#             type="mobile_device",
#             cpu_capacity=md_info["cpu_capacity"],
#             cache_capacity=md_info["cache_capacity"],
#             cpu_used=md_info["cpu_used"],
#             cache_used=md_info["cache_used"],
#             position=md_info["position"],
#             services=md_info["services"],
#             ips=md_info["ips"],
#             reuse=md_info["reuse"],
#         )

#         if closer_router:
#             router = graph._node[closer_router]
#             wireless_free = router["w_channel_capacity"] - router["w_channel_used"]

#             signal_latency = 1
#             graph.add_edge(
#                 mobile_device_id,
#                 closer_router,
#                 bandwidth_capacity=wireless_free,
#                 bandwidth_used=0.00,
#                 latency=signal_latency,
#                 services_in_transit={},
#             )

#     def submit_solution(self, graph, sfc, route_info):

#         def allocate_microservice(vnf, node_id, session_id):
#             service_id = vnf.id
#             service_key = (service_id, session_id)
#             cpu_required = vnf.get_cpu_request()
#             cache_required = vnf.get_cache_request()

#             if not isinstance(graph, Net2):
#                 node = graph.nodes[node_id]
#             else:
#                 node = graph.graph.nodes[node_id]

#             latency = calculate_computational_latency(graph, node_id, vnf)
#             allocated_resources = 0.0

#             if node_id == 0:
#                 return 0, 0.0

#             if node["type"] not in ["server", "mobile_device"]:
#                 raise ValueError("Serviços só podem ser alocados em servidores ou usuários.")

#             clean_current_id = service_id.replace("_b", "")
#             compatible_instance_found = False

#             if self.is_shareable(service_id) or self.is_shareable(clean_current_id):
#                 for existing_id, existing_session in node["services"]:
#                     existing_clean = existing_id.replace("_b", "")
#                     if existing_clean == clean_current_id and existing_session == session_id:
#                         compatible_instance_found = True
#                         break

#             if service_key in node["services"]:
#                 node["services"][service_key]["copys"] += 1

#                 if not self.is_shareable(service_id):
#                     if node["cpu_used"] + cpu_required > node["cpu_capacity"]:
#                         node["services"][service_key]["copys"] -= 1
#                         raise ValueError(f"CPU excedida no nó {node_id}")

#                     if node["cache_used"] + cache_required > node["cache_capacity"]:
#                         node["services"][service_key]["copys"] -= 1
#                         raise ValueError(f"Cache excedido no nó {node_id}")

#                 node["cpu_used"] += cpu_required
#                 node["cache_used"] += cache_required
#                 allocated_resources = cpu_required

#             else:
#                 cost_cpu = cpu_required
#                 cost_cache = cache_required

#                 if compatible_instance_found:
#                     cost_cpu = 0.0
#                     cost_cache = 0.0

#                 if node["cpu_used"] + cost_cpu > node["cpu_capacity"]:
#                     raise ValueError(f"CPU excedida no nó {node_id}")

#                 if node["cache_used"] + cost_cache > node["cache_capacity"]:
#                     raise ValueError(f"Cache excedido no nó {node_id}")

#                 node["services"][service_key] = {"cpu": cost_cpu, "cache": cost_cache, "copys": 1}

#                 node["cpu_used"] += cost_cpu
#                 node["cache_used"] += cost_cache

#                 allocated_resources = cost_cpu

#                 if self.is_shareable(service_id) or self.is_shareable(clean_current_id):
#                     node["reuse"].append(vnf)

#             return latency, allocated_resources

#         def allocate_bandwidth(node1, node2, vnf, ms_name):
#             bw_required = vnf.get_outcome_interface_bandwidth()
#             latency = calculate_latency_betwen_nodes(graph, node1, node2, vnf)
#             edge = graph.edges[node1, node2]

#             if edge["bandwidth_used"] + bw_required > edge["bandwidth_capacity"]:
#                 raise ValueError(f"Banda excedida entre {node1} e {node2} para {ms_name}")

#             if ms_name in edge["services_in_transit"]:
#                 edge["services_in_transit"][ms_name]["copys"] += 1
#                 edge["bandwidth_used"] += bw_required
#             else:
#                 edge["services_in_transit"][ms_name] = {"copys": 1, "bw_used": bw_required}
#                 edge["bandwidth_used"] += bw_required

#             return latency

#         session = sfc.id.split("_")[-1]
#         total_latency = 0
#         total_resources_consumed = 0.0

#         tsaber = {"computacao": {}, "comunicacao": {}}

#         for ms_name, path in route_info.items():
#             if ms_name in ["src", "dst"]:
#                 continue

#             vnf = sfc.get_vnf_by_id(ms_name)
#             node_allocated = path[0]

#             comp_latency, res_consumed = allocate_microservice(vnf, node_allocated, session)

#             total_latency += comp_latency
#             total_resources_consumed += res_consumed

#             tsaber["computacao"][ms_name] = {
#                 "node": node_allocated,
#                 "latencia_comp": comp_latency,
#                 "recursos_consumidos": res_consumed,
#             }

#             if len(path) > 1:
#                 for u, v in zip(path[:-1], path[1:], strict=False):
#                     comm_latency = allocate_bandwidth(u, v, vnf, ms_name)
#                     total_latency += comm_latency

#                     if ms_name not in tsaber["comunicacao"]:
#                         tsaber["comunicacao"][ms_name] = []

#                     tsaber["comunicacao"][ms_name].append(
#                         {"de": u, "para": v, "latencia_comm": comm_latency}
#                     )

#         total_comp_latency = sum(d["latencia_comp"] for d in tsaber["computacao"].values())
#         total_comm_latency = sum(
#             item["latencia_comm"] for items in tsaber["comunicacao"].values() for item in items
#         )

#         return (
#             round(total_latency, 2),
#             round(total_comp_latency, 2),
#             round(total_comm_latency, 2),
#             round(total_resources_consumed, 4),
#         )

#     def is_shareable(self, service_name: str) -> bool:
#         # REFATORAÇÃO: EAFP em vez de getattr
#         try:
#             share_val = self.args.share
#             share_enabled = share_val if isinstance(share_val, bool) else str(share_val).lower() == "y"
#         except AttributeError:
#             share_enabled = False

#         if share_enabled:
#             return service_name.startswith(SHAREABLE_PREFIXES)

#         return False

#     def musfico_method(self, sfc, substrate_network):
#         route_info = copy.deepcopy(self.sfcs_routing_info[sfc.id])

#         dst_vnf = sfc.get_dst_vnf()
#         previous_vnf = sfc.get_previous_vnf(dst_vnf)
#         dst_substrate_node = sfc.get_substrate_node(dst_vnf)
#         prev_vnf_node = route_info[previous_vnf.id][0]

#         shortest_path = k_shortest_paths(
#             substrate_network, prev_vnf_node, dst_substrate_node, k=1, weight="latency"
#         )

#         route_info[previous_vnf.id] = shortest_path[0]
#         latency = 0

#         for vnf_id in route_info:
#             if vnf_id == "src":
#                 continue
#             path = route_info[vnf_id]
#             for i in range(len(path) - 1):
#                 edge_latency = substrate_network.get_link_latency(path[i], path[i + 1])
#                 latency += edge_latency

#         try:
#             lat_req = sfc.latency_request
#         except AttributeError:
#             lat_req = sfc.get_latency_request() # Fallback

#         if latency > lat_req or latency < 0:
#             route_info = False
#             latency = None

#         return latency, route_info


import time
import networkx as nx
from loguru import logger

from muar_sfc.algorithms.darsppo import DARSPPO
from muar_sfc.algorithms.environments.env_da_rsppo import SFC_AllocationEnv_DARSPPO
from muar_sfc.algorithms.environments.env_replic import SFC_AllocationEnv as SFC_AllocationEnv_SCRC

from muar_sfc.algorithms.environments.environment import SFC_AllocationEnv
from muar_sfc.algorithms.environments.hephaestus_env import SFC_AllocationEnv_hephaestus
from muar_sfc.algorithms.hephaestus import hephaestus
from muar_sfc.algorithms.kuririn import Kuririn
from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
)
from muar_sfc.algorithms.replic import REPLIC

from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC
from muar_sfc.utils.network_utils import check_vnf_reusability

SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")


class SFCInstatiator:
    """
    Controlador de Admissão e Roteamento.
    Usa clonagem rasa O(1) para criar um Shadow State em lotes (Dry-Run),
    garantindo que a IA veja o consumo sequencial sem gargalar a CPU.
    """
    def __init__(self, alg, args=None):
        self.alg = alg
        self.args = args
        self.verbose = True
        self.env = None

    def _create_shadow_graph(self, substrate_network: Net2) -> nx.Graph:
        """Cria uma cópia física segura e super rápida para testes da IA."""
        shadow_graph = nx.Graph()
        for n, d in substrate_network.graph.nodes(data=True):
            shadow_graph.add_node(n, **d.copy())
        for u, v, d in substrate_network.graph.edges(data=True):
            shadow_graph.add_edge(u, v, **d.copy())
        return shadow_graph

    def search_solution(self, sfc_list: list[SFC], substrate_network: Net2, is_backup=False) -> tuple[dict, bool]:
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
                self.env = SFC_AllocationEnv(valid_nodes=valid_nodes, list_graph=[infra_to_use], list_sfc=[sfc_list[0]], is_training=False)
            case "DARSPPO":
                self.env = SFC_AllocationEnv_DARSPPO(valid_nodes=valid_nodes, list_graph=[infra_to_use], list_sfc=[sfc_list[0]], is_training=False)
            case "hephaestus":
                self.env = SFC_AllocationEnv_hephaestus(valid_nodes=valid_nodes, list_graph=[infra_to_use], list_sfc=[sfc_list[0]], is_training=False)
            case "REPLIC":
                self.env = SFC_AllocationEnv_SCRC(valid_nodes=valid_nodes, list_graph=[infra_to_use], list_sfc=[sfc_list[0]], is_training=False)

        return self.sequential_search(self.alg, sfc_list, infra_to_use, default_solution_format)

    def sequential_search(self, algorithm, sfc_list: list[SFC], graph: nx.Graph, solution_format: dict) -> tuple[dict, bool]:
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
                    logger.error(f"Tentativa de usar {algorithm.name} sem um ambiente inicializado.")
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
                    logger.warning(f" ⚠️ [DRY-RUN] Controle de Admissão bloqueou SFC {sfc.id}: {ve}")
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
            logger.info(f"Alg '{algorithm.name}' processou lote [{', '.join(s.id for s in sfc_list)}] em {round(total_elapsed_ms, 3)} ms. Status: {search_success}")
            if approved_sfcs: logger.debug(f"Aprovadas: {', '.join(approved_sfcs)}")
            if rejected_sfcs: logger.debug(f"Rejeitadas: {', '.join(rejected_sfcs)}")

        return solution_format, search_success

    def add_mobile_user_to_graph(self, graph: nx.Graph, substrate_network: Net2, sfc_list: list[SFC]) -> None:
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
            wireless_free = router.get("w_channel_capacity", 0.0) - router.get("w_channel_used", 0.0)

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

        for ms_name, path in route_info.items():
            if ms_name in ["src", "dst"]: continue

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
                    tsaber["comunicacao"].setdefault(ms_name, []).append({"latencia_comm": comm_latency})

        tot_comp = sum(d["latencia_comp"] for d in tsaber["computacao"].values())
        tot_comm = sum(item["latencia_comm"] for items in tsaber["comunicacao"].values() for item in items)

        return round(total_latency, 2), round(tot_comp, 2), round(tot_comm, 2), round(total_resources_consumed, 4)

    def is_shareable(self, service_name: str) -> bool:
        try:
            share_val = self.args.share
            share_enabled = share_val if isinstance(share_val, bool) else str(share_val).lower() == "y"
        except AttributeError:
            share_enabled = False

        return service_name.startswith(SHAREABLE_PREFIXES) if share_enabled else False
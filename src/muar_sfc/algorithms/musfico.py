"""
Consideration:
Algorithm should not do any modification on substrate network

It should only use network information and sfc information to
solve and give out a mapping and route info, that

route info :=
{
    src:  [1, 2, 3],
    vnf1: [3, 4, 5],
    vnf2: [5, 6, 7],
    vnf3: [7, 8 ,9],
    dst:  []
}

"""

import copy
import logging

import networkx as nx

# Imports adicionados para lidar com a rede como um Grafo e
# calcular a latência de comunicação realista
from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
    get_link_bandwidth_free,
)
from muar_sfc.config import ROOT_DIR

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.FileHandler(ROOT_DIR / "logs" / "musfico.log")
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class Musfico:
    def __init__(self):
        self.name = "musfico"
        self.graph = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.latency_minus_dst = None

    def clear_all(self):
        self.graph = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None

    def install_substrate_network(self, graph):
        self.graph = graph

        self.single_source_minimum_latency_path = {}
        for node in self.graph.nodes():
            self.single_source_minimum_latency_path[node] = nx.single_source_dijkstra(
                self.graph, source=node, cutoff=None, weight="latency"
            )
        return self.graph

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.route_info = {}

        src_vnf = self.sfc.get_src_vnf()
        src_substrate_node = self.sfc.get_substrate_node(src_vnf)
        dst_vnf = self.sfc.get_dst_vnf()
        dst_substrate_node = self.sfc.get_substrate_node(dst_vnf)

        for node in self.graph.nodes():
            self.node_info[node] = {}
            for vnf_id, _vnf in list(sfc.vnfs.items()):
                self.node_info[node][vnf_id] = {}
                self.node_info[node][vnf_id]["flag"] = False
                self.node_info[node][vnf_id]["latency"] = float("inf")
                self.node_info[node][vnf_id]["path"] = []
                self.node_info[node][vnf_id]["src_path"] = []
                self.node_info[node][vnf_id]["previous_substrate_node"] = None
                self.node_info[node][vnf_id]["current_substrate_nodes"] = []
                self.node_info[node][vnf_id]["bandwidth_usage_info"] = {}

            self.node_info[node][src_vnf.id] = {}
            self.node_info[node][src_vnf.id]["flag"] = False
            self.node_info[node][dst_vnf.id] = {}

        self.node_info[src_substrate_node][src_vnf.id]["flag"] = True
        self.node_info[src_substrate_node][src_vnf.id]["latency"] = 0
        self.node_info[src_substrate_node][src_vnf.id]["src_path"] = []
        self.node_info[src_substrate_node][src_vnf.id]["path"] = []
        self.node_info[src_substrate_node][src_vnf.id]["current_substrate_nodes"] = [
            src_substrate_node
        ]

        if dst_substrate_node not in self.node_info:
            self.node_info[dst_substrate_node] = {
                dst_vnf.id: {
                    "flag": False,
                    "latency": float("inf"),
                    "src_path": [],
                    "path": [],
                    "current_substrate_nodes": [],
                    "bandwidth_usage_info": {},
                }
            }
        else:
            self.node_info[dst_substrate_node][dst_vnf.id]["flag"] = False
            self.node_info[dst_substrate_node][dst_vnf.id]["latency"] = float("inf")
            self.node_info[dst_substrate_node][dst_vnf.id]["src_path"] = []
            self.node_info[dst_substrate_node][dst_vnf.id]["path"] = []
            self.node_info[dst_substrate_node][dst_vnf.id]["current_substrate_nodes"] = []

        self.node_info[src_substrate_node][src_vnf.id]["bandwidth_usage_info"] = {}
        self.node_info[dst_substrate_node][dst_vnf.id]["bandwidth_usage_info"] = {}

        return self.sfc

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def start_algorithm(self):
        if self.algorithm():
            if self.latency is not None:
                if self.latency < 0:
                    print("Latencia negativa")
                    self.latency = None
                    self.route_info = False
                    return False

                if self.latency > self.sfc.get_latency_request():
                    self.latency = None
                    self.route_info = False
                    return False
            return True
        return False

    def algorithm(self):
        sfc = self.sfc

        if getattr(self.sfc, "is_dag", lambda: False)():
            return self._solve_dag_musfico()

        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        self.src_substrate_node = src_substrate_node
        self.dst_substrate_node = dst_substrate_node

        (node_latency, node_path) = self.single_source_minimum_latency_path[dst_substrate_node]

        vnf1 = src_vnf.get_next_vnf()
        self._dp(src_substrate_node, vnf1)

        vnf = vnf1.get_next_vnf()
        while vnf.id != dst_vnf.id:
            for node in self.graph.nodes():
                self._dp(node, vnf)
            vnf = vnf.get_next_vnf()

        previous_vnf = sfc.get_previous_vnf(dst_vnf)
        previous_vnf_id = previous_vnf.id

        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, dst_vnf.id)

        for node in self.graph.nodes():
            if node in (dst_substrate_node, src_substrate_node):
                continue

            path = node_path.get(node)
            if not path:
                continue

            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[node][previous_vnf_id]["bandwidth_usage_info"]
            )
            length = len(path)

            real_comm_latency = 0.0

            for i in range(0, length - 1):
                u, v = path[i], path[i + 1]
                edge_key = frozenset((u, v))
                residual_bandwidth = None

                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = (
                        get_link_bandwidth_free(self.graph, u, v) - bandwidth_request
                    )

                if residual_bandwidth < 0:
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth

                # BOA PRÁTICA: Ignora a latência das arestas caso a VNF seja src ou dst,
                # mantendo paridade com o Instantiator
                if previous_vnf_id not in ["src", "dst"]:
                    real_comm_latency += calculate_latency_betwen_nodes(
                        self.graph, u, v, previous_vnf
                    )

            if not is_bandwidth_sufficient:
                continue

            _latency = self.node_info[node][previous_vnf_id]["latency"]
            total_latency = _latency + real_comm_latency

            if total_latency > 35:
                continue

            if (
                not self.node_info[dst_substrate_node][dst_vnf.id]["latency"]
                or total_latency < self.node_info[dst_substrate_node][dst_vnf.id]["latency"]
            ):
                self.node_info[dst_substrate_node][dst_vnf.id]["latency"] = total_latency
                self.node_info[dst_substrate_node][dst_vnf.id]["path"] = path
                self.node_info[dst_substrate_node][dst_vnf.id]["path"].reverse()
                self.node_info[dst_substrate_node][dst_vnf.id]["current_substrate_nodes"] = (
                    self.node_info[node][previous_vnf_id]["current_substrate_nodes"][:]
                )
                self.node_info[dst_substrate_node][dst_vnf.id]["current_substrate_nodes"].append(
                    dst_substrate_node
                )
                self.node_info[dst_substrate_node][dst_vnf.id]["src_path"] = (
                    self.node_info[node][previous_vnf_id]["src_path"][:]
                    + self.node_info[dst_substrate_node]["dst"]["path"][:]
                )
                self.node_info[dst_substrate_node][dst_vnf.id]["flag"] = True

        if self.node_info[dst_substrate_node][dst_vnf.id]["flag"]:
            previous_vnf = dst_vnf
            previous_substrate_node = dst_substrate_node

            while True:
                path = self.node_info[previous_substrate_node][previous_vnf.id]["path"]
                if not path:
                    break
                previous_substrate_node = path[0]
                previous_vnf = sfc.get_previous_vnf(previous_vnf)
                if previous_vnf:
                    self.route_info[previous_vnf.id] = path
                else:
                    break

            self.route_info[dst_vnf.id] = []
            self.latency = self.node_info[dst_substrate_node][dst_vnf.id]["latency"]

            # COMO AGORA A LATÊNCIA FOI ACUMULADA CORRETAMENTE (SEM O SRC),
            # AQUELE LAÇO CONFUSO QUE SUBTRAÍA A LATÊNCIA NO FINAL FOI DELETADO.

            if self.latency > sfc.get_latency_request() or self.latency < 0:
                self.route_info = {}
                self.latency = None
                return False
            return True
        else:
            return False

    def handle_failure(self):
        self.route_info = False
        self.latency = None

    def _dp(self, substrate_node, vnf):
        sfc = self.sfc
        previous_vnf = sfc.get_previous_vnf(vnf)
        previous_vnf_id = previous_vnf.id

        vnf_id = vnf.id
        (node_latency, node_path) = self.single_source_minimum_latency_path[substrate_node]

        _latency = self.node_info[substrate_node][previous_vnf_id]["latency"]

        cpu_request = sfc.get_vnf_cpu_request(vnf)
        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, vnf_id)
        cache_request = sfc.get_vnf_cache_request(vnf)

        for node in self.graph.nodes():
            # Guard Clauses
            if node == substrate_node:
                continue
            if node in self.node_info[substrate_node][previous_vnf_id]["current_substrate_nodes"]:
                continue
            if node == self.src_substrate_node or node == self.dst_substrate_node:
                continue

            cpu_available = (
                self.graph.nodes[node]["cpu_capacity"] - self.graph.nodes[node]["cpu_used"]
            )
            cache_available = (
                self.graph.nodes[node]["cache_capacity"] - self.graph.nodes[node]["cache_used"]
            )

            if cpu_available == 0 or cache_available == 0:
                continue

            if cpu_request > cpu_available or cache_request > cache_available:
                continue

            path = node_path.get(node)
            if not path:
                continue

            if path[0] != substrate_node:
                path.reverse()
            length = len(path)

            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[substrate_node][previous_vnf_id]["bandwidth_usage_info"]
            )

            real_comm_latency = 0.0

            for i in range(0, length - 1):
                u, v = path[i], path[i + 1]
                edge_key = frozenset((u, v))
                residual_bandwidth = None

                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = (
                        get_link_bandwidth_free(self.graph, u, v) - bandwidth_request
                    )

                if residual_bandwidth < 0:
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth

                # BOA PRÁTICA: Ignora a latência de comunicação se a VNF for src ou dst
                if previous_vnf_id not in ["src", "dst"]:
                    real_comm_latency += calculate_latency_betwen_nodes(
                        self.graph, u, v, previous_vnf
                    )

            if not is_bandwidth_sufficient:
                continue

            total_latency = _latency + real_comm_latency

            self.node_info[node][vnf_id]["bandwidth_usage_info"] = bandwidth_usage_info

            if (
                not self.node_info[node][vnf_id]["latency"]
                or total_latency <= self.node_info[node][vnf_id]["latency"]
            ):
                self.node_info[node][vnf_id]["latency"] = total_latency
                self.node_info[node][vnf_id]["path"] = path
                self.node_info[node][vnf_id]["flag"] = True
                self.node_info[node][vnf_id]["previous_substrate_node"] = substrate_node
                self.node_info[node][vnf_id]["current_substrate_nodes"] = self.node_info[
                    substrate_node
                ][previous_vnf_id]["current_substrate_nodes"][:]
                self.node_info[node][vnf_id]["current_substrate_nodes"].append(node)
                self.node_info[node][vnf_id]["src_path"] = (
                    self.node_info[substrate_node][previous_vnf_id]["src_path"][:] + path[:-1]
                )

        return True

    def _solve_dag_musfico(self) -> bool:
        """
        Resolução ótima baseada em Programação Dinâmica para SFCs em DAG com bifurcação e junção sincronizada:
        1. Aloca IA_DET_FT minimizando a latência até src.
        2. Resolve os ramos concorrentes:
           - MA_region (Cache)
           - UNI_player (Únicos)
        3. Encontra a melhor alocação de RE_player que minimiza a latência crítica
           sujeita à restrição de sincronização inter-modal: |Lat(Cache) - Lat(Unico)| <= sync_tol.
        4. Aloca EC_TC_player minimizando a latência até dst.
        """
        sfc = self.sfc
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()
        src_node = sfc.get_substrate_node(src_vnf)
        dst_node = sfc.get_substrate_node(dst_vnf)

        sync_tol = getattr(sfc, "sync_tolerance", 5.0)

        # Identificação das VNFs do DAG
        ia_vnf = None
        ma_vnf = None
        uni_vnf = None
        re_vnf = None
        ec_tc_vnf = None

        for v_id, v_obj in sfc.vnfs.items():
            if v_id.startswith("IA_DET_FT"): ia_vnf = v_obj
            elif v_id.startswith("MA_region"): ma_vnf = v_obj
            elif v_id.startswith("UNI_"): uni_vnf = v_obj
            elif v_id.startswith("RE_"): re_vnf = v_obj
            elif v_id.startswith("EC_TC_"): ec_tc_vnf = v_obj

        if not (ia_vnf and ma_vnf and uni_vnf and re_vnf and ec_tc_vnf):
            return False

        def get_candidates(vnf, req_cpu, req_cache=0.0):
            cands = []
            for n, d in self.graph.nodes(data=True):
                if d.get("type") not in ("server", "mobile_device"):
                    continue
                if n in (src_node, dst_node):
                    continue
                c_free = d.get("cpu_capacity", 0.0) - d.get("cpu_used", 0.0)
                ca_free = d.get("cache_capacity", 0.0) - d.get("cache_used", 0.0)
                if c_free >= req_cpu and ca_free >= req_cache:
                    cands.append(n)
            return cands

        def get_path(u, v):
            try:
                return nx.shortest_path(self.graph, source=u, target=v, weight="latency")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []

        def path_latency(p):
            if not p or len(p) < 2:
                return 0.0
            lat = 0.0
            for i in range(len(p) - 1):
                lat += self.graph.edges[p[i], p[i + 1]].get("latency", 1.0)
            return lat

        # 1. Candidatos IA
        ia_cands = get_candidates(ia_vnf, ia_vnf.get_cpu_request(), ia_vnf.get_cache_request())
        if not ia_cands:
            return False
        ia_node = min(ia_cands, key=lambda n: path_latency(get_path(src_node, n)))
        path_src_ia = get_path(src_node, ia_node)

        # 2. Candidatos MA (Cache) e UNI (Únicos)
        ma_cands = get_candidates(ma_vnf, ma_vnf.get_cpu_request(), ma_vnf.get_cache_request())
        uni_cands = get_candidates(uni_vnf, uni_vnf.get_cpu_request(), uni_vnf.get_cache_request())
        re_cands = get_candidates(re_vnf, re_vnf.get_cpu_request(), re_vnf.get_cache_request())
        ec_cands = get_candidates(ec_tc_vnf, ec_tc_vnf.get_cpu_request(), ec_tc_vnf.get_cache_request())

        if not (ma_cands and uni_cands and re_cands and ec_cands):
            return False

        # Programação Dinâmica sobre as tuplas (ma, uni, re)
        best_tuple = None
        best_crit_lat = float("inf")
        best_delta = float("inf")

        for cand_re in re_cands:
            # Para cada nó RE, busca o melhor nó MA e o melhor nó UNI
            valid_ma_for_re = []
            for m in ma_cands:
                p_ia_m = get_path(ia_node, m)
                p_m_re = get_path(m, cand_re)
                if p_ia_m and p_m_re:
                    comp_m = calculate_computational_latency(self.graph, m, ma_vnf)
                    lat_m = path_latency(p_ia_m) + comp_m + path_latency(p_m_re)
                    valid_ma_for_re.append((m, lat_m, p_ia_m, p_m_re))

            valid_uni_for_re = []
            for u in uni_cands:
                p_ia_u = get_path(ia_node, u)
                p_u_re = get_path(u, cand_re)
                if p_ia_u and p_u_re:
                    comp_u = calculate_computational_latency(self.graph, u, uni_vnf)
                    lat_u = path_latency(p_ia_u) + comp_u + path_latency(p_u_re)
                    valid_uni_for_re.append((u, lat_u, p_ia_u, p_u_re))

            if not valid_ma_for_re or not valid_uni_for_re:
                continue

            # Seleciona o par (ma, uni) que minimiza |lat_m - lat_u| e o caminho crítico max(lat_m, lat_u)
            for m_info in valid_ma_for_re:
                for u_info in valid_uni_for_re:
                    m_node, lat_m, p_ia_m, p_m_re = m_info
                    u_node, lat_u, p_ia_u, p_u_re = u_info

                    delta = abs(lat_m - lat_u)
                    crit = max(lat_m, lat_u)

                    if delta <= sync_tol:
                        if crit < best_crit_lat:
                            best_crit_lat = crit
                            best_delta = delta
                            best_tuple = (cand_re, m_info, u_info)

        # Fallback de sincronização se não encontrar estritamente dentro da tolerância
        if best_tuple is None:
            for cand_re in re_cands:
                for m in ma_cands:
                    p_ia_m = get_path(ia_node, m)
                    p_m_re = get_path(m, cand_re)
                    if not (p_ia_m and p_m_re): continue
                    comp_m = calculate_computational_latency(self.graph, m, ma_vnf)
                    lat_m = path_latency(p_ia_m) + comp_m + path_latency(p_m_re)

                    for u in uni_cands:
                        p_ia_u = get_path(ia_node, u)
                        p_u_re = get_path(u, cand_re)
                        if not (p_ia_u and p_u_re): continue
                        comp_u = calculate_computational_latency(self.graph, u, uni_vnf)
                        lat_u = path_latency(p_ia_u) + comp_u + path_latency(p_u_re)

                        delta = abs(lat_m - lat_u)
                        crit = max(lat_m, lat_u)
                        if delta < best_delta:
                            best_delta = delta
                            best_crit_lat = crit
                            best_tuple = (cand_re, (m, lat_m, p_ia_m, p_m_re), (u, lat_u, p_ia_u, p_u_re))

        if best_tuple is None or best_delta > sync_tol * 2:
            return False

        re_node, (ma_node, _, path_ia_ma, path_ma_re), (uni_node, _, path_ia_uni, path_uni_re) = best_tuple

        # 4. Alocar EC_TC minimizando o caminho RE -> EC_TC -> dst
        ec_node = min(ec_cands, key=lambda n: path_latency(get_path(re_node, n)) + path_latency(get_path(n, dst_node)))
        path_re_ec = get_path(re_node, ec_node)
        path_ec_dst = get_path(ec_node, dst_node)

        # Montagem do route_info padronizado
        self.route_info = {
            "src": path_src_ia if path_src_ia else [src_node],
            ia_vnf.id: [ia_node],
            ma_vnf.id: list(reversed(path_ia_ma)) if path_ia_ma else [ma_node],
            uni_vnf.id: list(reversed(path_ia_uni)) if path_ia_uni else [uni_node],
            re_vnf.id: list(reversed(path_ma_re)) if path_ma_re else [re_node],
            ec_tc_vnf.id: list(reversed(path_re_ec)) if path_re_ec else [ec_node],
            "dst": path_ec_dst if path_ec_dst else [dst_node],
        }

        comp_ia = calculate_computational_latency(self.graph, ia_node, ia_vnf)
        comp_re = calculate_computational_latency(self.graph, re_node, re_vnf)
        comp_ec = calculate_computational_latency(self.graph, ec_node, ec_tc_vnf)

        self.latency = round(comp_ia + best_crit_lat + comp_re + path_latency(path_re_ec) + comp_ec + path_latency(path_ec_dst), 2)
        return True

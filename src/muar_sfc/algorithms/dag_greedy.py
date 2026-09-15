import copy
import logging
import networkx as nx
from loguru import logger

from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
    get_link_bandwidth_free,
)
from muar_sfc.core.sfc import SFC

logger_local = logging.getLogger(__name__)


class DAGGreedyAlgorithm:
    """
    Algoritmo de Alocação de SFC em DAG com Sincronização Inter-Modal (6G Metaverse/XR).
    
    Características:
    1. Aloca VNFs seguindo a ordem topológica do DAG.
    2. Suporta bifurcação (Fork) de IA_DET_FT para MA (Cache) e UNI (Únicos).
    3. Suporta junção (Join) sincronizada em RE (Renderização), impondo:
       | Latencia(Ramo Cache) - Latencia(Ramo Unico) | <= delta_sync (tolerancia)
    4. Minimiza a latência ponta a ponta pelo Caminho Crítico:
       T^{e2e} = max(Latencia_Cache, Latencia_Unico)
    """

    def __init__(self):
        self.name = "DAG_Greedy"
        self.graph = None
        self.sfc: SFC | None = None
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.sync_differential = 0.0
        self.fail_reason = None
        self.allow_md_host = False

    def clear_all(self):
        self.graph = None
        self.sfc = None
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.sync_differential = 0.0
        self.fail_reason = None

    def install_substrate_network(self, graph: nx.Graph):
        self.graph = graph
        return self.graph

    def install_SFC(self, sfc: SFC):
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.sync_differential = 0.0
        return self.sfc

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def get_fail_reason(self):
        return self.fail_reason

    def handle_failure(self):
        self.route_info = {}
        self.latency = None

    def start_algorithm(self, env=None, args=None) -> bool:
        if not self.graph or not self.sfc:
            self.fail_reason = "Grafo ou SFC não instalados."
            self.handle_failure()
            return False

        success = self._solve_dag()
        if not success:
            self.handle_failure()
            return False

        max_lat = self.sfc.get_latency_request()
        if max_lat and self.latency > max_lat:
            self.fail_reason = f"Latência crítica {self.latency:.2f}ms excedeu limite {max_lat}ms."
            self.handle_failure()
            return False

        return True

    def _get_valid_candidate_nodes(self, vnf, required_cpu, required_cache=0.0):
        candidates = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") not in ("server", "mobile_device"):
                continue
            if not self.allow_md_host and data.get("type") == "mobile_device":
                continue

            cpu_free = data.get("cpu_capacity", 0.0) - data.get("cpu_used", 0.0)
            cache_free = data.get("cache_capacity", 0.0) - data.get("cache_used", 0.0)

            if cpu_free >= required_cpu and cache_free >= required_cache:
                candidates.append(node_id)
        return candidates

    def _get_shortest_path(self, u_node, v_node):
        try:
            return nx.shortest_path(self.graph, source=u_node, target=v_node, weight="latency")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _path_latency(self, path):
        if not path or len(path) < 2:
            return 0.0
        lat = 0.0
        for i in range(len(path) - 1):
            lat += self.graph.edges[path[i], path[i + 1]].get("latency", 1.0)
        return lat

    def _solve_dag(self) -> bool:
        src_vnf = self.sfc.get_src_vnf()
        dst_vnf = self.sfc.get_dst_vnf()

        src_node = self.sfc.get_substrate_node(src_vnf)
        dst_node = self.sfc.dst_node

        vnfs = self.sfc.vnfs
        sync_tol = getattr(self.sfc, "sync_tolerance", 5.0)

        # Identificação dos componentes-chave do DAG MUAR
        ia_vnf = None
        ma_vnf = None
        uni_vnf = None
        re_vnf = None
        ec_tc_vnf = None

        for vnf_id, vnf in vnfs.items():
            if vnf_id.startswith("IA_DET_FT"):
                ia_vnf = vnf
            elif vnf_id.startswith("MA_region"):
                ma_vnf = vnf
            elif vnf_id.startswith("UNI_"):
                uni_vnf = vnf
            elif vnf_id.startswith("RE_"):
                re_vnf = vnf
            elif vnf_id.startswith("EC_TC_"):
                ec_tc_vnf = vnf

        if not (ia_vnf and ma_vnf and uni_vnf and re_vnf and ec_tc_vnf):
            # Fallback para DAG genérico se os nomes não baterem com o MUAR padrão
            return self._solve_generic_dag()

        # 1. Alocar IA_DET_FT (mais próximo do nó de origem ou do roteador de acesso)
        ia_candidates = self._get_valid_candidate_nodes(ia_vnf, ia_vnf.get_cpu_request(), ia_vnf.get_cache_request())
        if not ia_candidates:
            self.fail_reason = f"Sem candidatos de CPU para IA_DET_FT."
            return False

        # Escolhe o nó com menor latência até a origem
        ia_node = min(ia_candidates, key=lambda n: self._path_latency(self._get_shortest_path(src_node, n)))
        path_src_to_ia = self._get_shortest_path(src_node, ia_node)

        # 2. Alocar MA_region (Ramo de Cache)
        ma_candidates = self._get_valid_candidate_nodes(ma_vnf, ma_vnf.get_cpu_request(), ma_vnf.get_cache_request())
        if not ma_candidates:
            self.fail_reason = "Sem candidatos com Cache suficiente para MA_region."
            return False

        # Prioriza o próprio nó do IA ou o nó com maior cache livre próximo
        ma_node = min(ma_candidates, key=lambda n: self._path_latency(self._get_shortest_path(ia_node, n)))
        path_ia_to_ma = self._get_shortest_path(ia_node, ma_node)

        # 3. Alocar UNI_player (Ramo de Objetos Únicos)
        uni_candidates = self._get_valid_candidate_nodes(uni_vnf, uni_vnf.get_cpu_request(), uni_vnf.get_cache_request())
        if not uni_candidates:
            self.fail_reason = "Sem candidatos para UNI."
            return False

        # Escolhe nó com CPU livre próximo ao IA
        uni_node = min(uni_candidates, key=lambda n: self._path_latency(self._get_shortest_path(ia_node, n)))
        path_ia_to_uni = self._get_shortest_path(ia_node, uni_node)

        # 4. Alocar RE_player (Convergência com Sincronização)
        re_candidates = self._get_valid_candidate_nodes(re_vnf, re_vnf.get_cpu_request(), re_vnf.get_cache_request())
        if not re_candidates:
            self.fail_reason = "Sem candidatos para RE."
            return False

        best_re_node = None
        best_crit_lat = float("inf")
        best_delta_sync = float("inf")
        best_paths = None

        comp_ma = calculate_computational_latency(self.graph, ma_node, ma_vnf)
        comp_uni = calculate_computational_latency(self.graph, uni_node, uni_vnf)

        for cand_re in re_candidates:
            path_ma_re = self._get_shortest_path(ma_node, cand_re)
            path_uni_re = self._get_shortest_path(uni_node, cand_re)
            if not path_ma_re or not path_uni_re:
                continue

            lat_branch_ma = self._path_latency(path_ia_to_ma) + comp_ma + self._path_latency(path_ma_re)
            lat_branch_uni = self._path_latency(path_ia_to_uni) + comp_uni + self._path_latency(path_uni_re)

            delta_sync = abs(lat_branch_ma - lat_branch_uni)
            crit_lat = max(lat_branch_ma, lat_branch_uni)

            # Restrição estrita de sincronização: delta_sync <= sync_tol
            if delta_sync <= sync_tol:
                if crit_lat < best_crit_lat:
                    best_crit_lat = crit_lat
                    best_delta_sync = delta_sync
                    best_re_node = cand_re
                    best_paths = (path_ma_re, path_uni_re)

        # Se nenhum nó atendeu à sincronização estrita, seleciona o de menor violação
        if best_re_node is None:
            for cand_re in re_candidates:
                path_ma_re = self._get_shortest_path(ma_node, cand_re)
                path_uni_re = self._get_shortest_path(uni_node, cand_re)
                if path_ma_re and path_uni_re:
                    lat_branch_ma = self._path_latency(path_ia_to_ma) + comp_ma + self._path_latency(path_ma_re)
                    lat_branch_uni = self._path_latency(path_ia_to_uni) + comp_uni + self._path_latency(path_uni_re)
                    delta_sync = abs(lat_branch_ma - lat_branch_uni)
                    if delta_sync < best_delta_sync:
                        best_delta_sync = delta_sync
                        best_re_node = cand_re
                        best_crit_lat = max(lat_branch_ma, lat_branch_uni)
                        best_paths = (path_ma_re, path_uni_re)

            if best_delta_sync > sync_tol * 2:
                self.fail_reason = f"Violação excessiva de Sincronização: {best_delta_sync:.2f}ms > {sync_tol}ms."
                return False

        re_node = best_re_node
        path_ma_to_re, path_uni_to_re = best_paths

        # 5. Alocar EC_TC_player (Codificação e Envio ao Destino)
        ec_candidates = self._get_valid_candidate_nodes(ec_tc_vnf, ec_tc_vnf.get_cpu_request(), ec_tc_vnf.get_cache_request())
        if not ec_candidates:
            self.fail_reason = "Sem candidatos para EC_TC."
            return False

        ec_node = min(ec_candidates, key=lambda n: self._path_latency(self._get_shortest_path(re_node, n)) + self._path_latency(self._get_shortest_path(n, dst_node)))
        path_re_to_ec = self._get_shortest_path(re_node, ec_node)
        path_ec_to_dst = self._get_shortest_path(ec_node, dst_node)

        # Montagem do Route Info (Mapeamento completo com nó hospedeiro sempre em path[0])
        self.route_info = {
            "src": path_src_to_ia if path_src_to_ia else [src_node],
            ia_vnf.id: [ia_node],
            ma_vnf.id: list(reversed(path_ia_to_ma)) if path_ia_to_ma else [ma_node],
            uni_vnf.id: list(reversed(path_ia_to_uni)) if path_ia_to_uni else [uni_node],
            re_vnf.id: list(reversed(path_ma_to_re)) if path_ma_to_re else [re_node],
            ec_tc_vnf.id: list(reversed(path_re_to_ec)) if path_re_to_ec else [ec_node],
            "dst": path_ec_to_dst if path_ec_to_dst else [dst_node],
        }

        # Cálculo da Latência Total pelo Caminho Crítico
        comp_ia = calculate_computational_latency(self.graph, ia_node, ia_vnf)
        comp_re = calculate_computational_latency(self.graph, re_node, re_vnf)
        comp_ec = calculate_computational_latency(self.graph, ec_node, ec_tc_vnf)

        comm_src_ia = self._path_latency(path_src_to_ia)
        comm_re_ec = self._path_latency(path_re_to_ec)
        comm_ec_dst = self._path_latency(path_ec_to_dst)

        self.latency = round(comm_src_ia + comp_ia + best_crit_lat + comp_re + comm_re_ec + comp_ec + comm_ec_dst, 2)
        self.sync_differential = round(best_delta_sync, 2)

        return True

    def _solve_generic_dag(self) -> bool:
        # Heurística topológica genérica para qualquer DAG
        try:
            topo_order = list(nx.topological_sort(self.sfc.graph))
        except nx.NetworkXUnfeasible:
            self.fail_reason = "Grafo da SFC contém ciclos (não é um DAG)."
            return False

        src_node = self.sfc.get_substrate_node(self.sfc.get_src_vnf())
        dst_node = self.sfc.dst_node

        alloc_nodes = {"src": src_node, "dst": dst_node}
        total_lat = 0.0

        for vnf_id in topo_order:
            if vnf_id in ("src", "dst"):
                continue
            vnf = self.sfc.get_vnf_by_id(vnf_id)
            cands = self._get_valid_candidate_nodes(vnf, vnf.get_cpu_request(), vnf.get_cache_request())
            if not cands:
                self.fail_reason = f"Sem candidatos para VNF {vnf_id}."
                return False

            preds = list(self.sfc.graph.predecessors(vnf_id))
            pred_nodes = [alloc_nodes.get(p, src_node) for p in preds]

            best_node = min(cands, key=lambda n: sum(self._path_latency(self._get_shortest_path(pn, n)) for pn in pred_nodes))
            alloc_nodes[vnf_id] = best_node
            self.route_info[vnf_id] = [best_node]
            total_lat += calculate_computational_latency(self.graph, best_node, vnf)

        self.route_info["src"] = [src_node]
        self.route_info["dst"] = [dst_node]
        self.latency = round(total_lat, 2)
        return True

import copy
import logging
from typing import Any, Protocol

import networkx as nx

from muar_sfc.algorithms.networkUtils import get_link_bandwidth_free, get_link_latency
from muar_sfc.config import ROOT_DIR

# =====================================================================
# Configuração de Observabilidade Estruturada
# =====================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Modernização Orientada a Objetos Multiplataforma (pathlib)
log_path = ROOT_DIR / "logs" / "DynamicProgrammingAlgorithm.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

ch = logging.FileHandler(log_path)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

SHAREABLE_PREFIXES = ("IA_DET_FT_", "RE_region_", "MA_region_")

# =====================================================================
# 1. CONTRATOS E TIPAGEM (Subtipagem Estrutural / Duck Typing)
# =====================================================================
NodeID = Any
Latency = float
PathType = list[NodeID]

# Tabela de roteamento: Origem -> (Dict[Destino, Latência], Dict[Destino, Caminho])
RoutingTable = dict[NodeID, tuple[dict[NodeID, Latency], dict[NodeID, PathType]]]


class SubstrateNetworkInterface(Protocol):
    """
    Protocolo que define as operações mínimas que a rede deve suportar
    para que o algoritmo MSF funcione (Desacoplamento).
    """

    def nodes(self) -> Any: ...
    def pre_get_single_source_minimum_latency_path(self) -> RoutingTable: ...


# =====================================================================
# 2. ALGORITMO CORE
# =====================================================================


class MSF:
    """
    Algoritmo de Programação Dinâmica (Minimum Shortest First) para alocação de SFC.
    """

    def __init__(self, max_allowed_latency: float = 13.0):
        """
        Inicializa o algoritmo MSF.

        Args:
            max_allowed_latency (float): O limite máximo tolerado de latência para
                                         poda de rotas inviáveis (Evita Magic Numbers).
        """
        self.name: str = "msf"
        self.sfc: Any | None = None
        self.node_info: dict[NodeID, dict[str, Any]] = {}
        self.route_info: dict[str, PathType] = {}
        self.latency: float | None = None

        # Estado do Grafo
        self.graph: SubstrateNetworkInterface | None = None
        self.src_substrate_node: NodeID | None = None
        self.dst_substrate_node: NodeID | None = None

        # Cache de rotas
        self.single_source_minimum_latency_path: RoutingTable = {}

        # Parâmetros Injetáveis
        self.max_allowed_latency: float = max_allowed_latency
        self.forbidden_matches: dict[str, NodeID] = {}
        self.shareable_sfs: Any | None = None

    def clear_all(self) -> None:
        """Limpa o estado da instância para um novo ciclo de execução."""
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = {}
        self.graph = None
        self.latency = None
        self.shareable_sfs = None

    def install_substrate_network(self, graph, shareable_sfs: list | None = None):
        """
        Instala a cópia local do grafo.
        O MSF é autossuficiente: ele mesmo pré-calcula o cache de latência
        usando a biblioteca networkx, desacoplando totalmente da classe Net2.
        """
        if shareable_sfs is None:
            shareable_sfs = []

        self.graph = graph

        # Construção interna do cache de rotas
        self.single_source_minimum_latency_path = {}
        for node in self.graph.nodes():
            self.single_source_minimum_latency_path[node] = nx.single_source_dijkstra(
                self.graph, source=node, cutoff=None, weight="latency"
            )

        return self.graph

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        src_vnf = self.sfc.get_src_vnf()
        src_substrate_node = self.sfc.get_substrate_node(src_vnf)
        dst_vnf = self.sfc.get_dst_vnf()
        dst_substrate_node = self.sfc.get_substrate_node(dst_vnf)

        for node in self.graph.nodes():
            self.node_info[node] = {}
            for vnf_id, _vnf in list(sfc.vnfs.items()):
                # Not include src and dst.
                self.node_info[node][vnf_id] = {}
                self.node_info[node][vnf_id]["flag"] = (
                    False  # whether vnf/id can be placed on node
                )
                self.node_info[node][vnf_id]["latency"] = float("inf")
                self.node_info[node][vnf_id]["path"] = []
                self.node_info[node][vnf_id]["src_path"] = []
                self.node_info[node][vnf_id]["previous_substrate_node"] = None
                self.node_info[node][vnf_id][
                    "current_substrate_nodes"
                ] = []  # The meta information
                self.node_info[node][vnf_id]["bandwidth_usage_info"] = {}

            self.node_info[node][src_vnf.id] = {}
            self.node_info[node][src_vnf.id]["flag"] = (
                False  # src cannot be placed except src node
            )
            self.node_info[node][dst_vnf.id] = {}

        self.node_info[src_substrate_node][src_vnf.id]["flag"] = True
        self.node_info[src_substrate_node][src_vnf.id]["latency"] = 0
        self.node_info[src_substrate_node][src_vnf.id]["src_path"] = []
        self.node_info[src_substrate_node][src_vnf.id]["path"] = []
        self.node_info[src_substrate_node][src_vnf.id]["current_substrate_nodes"] = [
            src_substrate_node
        ]
        self.node_info[dst_substrate_node][dst_vnf.id]["flag"] = False
        self.node_info[dst_substrate_node][dst_vnf.id]["latency"] = float("inf")
        self.node_info[dst_substrate_node][dst_vnf.id]["src_path"] = []
        self.node_info[dst_substrate_node][dst_vnf.id]["path"] = []
        self.node_info[dst_substrate_node][dst_vnf.id]["current_substrate_nodes"] = []
        self.node_info[src_substrate_node][src_vnf.id]["bandwidth_usage_info"] = {}
        self.node_info[dst_substrate_node][dst_vnf.id]["bandwidth_usage_info"] = {}

        return self.sfc

    def handle_failure(self):
        self.route_info = False
        self.latency = None

    def check_solution(self) -> bool:
        """
        Valida a integridade da solução encontrada.
        Refatorado para suportar SFCs de tamanhos dinâmicos (OCP).
        """
        if (
            not isinstance(self.latency, (int, float))
            or self.latency < 0
            or self.latency > self.sfc.get_latency_request()
            or not self.route_info
        ):
            return False

        # OCP: Validação baseada no tamanho real da SFC (VNFs internas + src + dst)
        expected_length = len(self.sfc.vnfs) + 2
        if len(self.route_info) != expected_length:
            return False

        prev_path_end = None
        # O dicionário route_info é populado de trás para frente no MSF
        for sf, path in self.route_info.items():
            if sf == "dst":
                continue

            # Valida se o fim do caminho atual conecta com o início do próximo (SIM102 resolvido)
            if prev_path_end is not None and path and path[-1] != prev_path_end:
                logger.debug(
                    f"Inconsistência de rota antes de {sf}: "
                    f"{prev_path_end} != {path[-1]}"
                )
                return False

            if path:
                prev_path_end = path[0]

        return True

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def start_algorithm(self):
        self.algorithm()
        is_success = self.check_solution()
        if is_success:
            try:
                logger.info("Finished algorithm, success")
                return True
            except Exception as e:
                logger.exception(f"Erro ao processar o sucesso do algoritmo MSF: {e}")
                self.handle_failure()
                return False
        else:
            self.handle_failure()
            logger.info("End algorithm, failed")
            return False

    def algorithm(self) -> bool:
        """Executa a rotina principal de roteamento e alocação dinâmica."""
        nodes = self.graph.nodes()
        src_vnf = self.sfc.get_src_vnf()
        dst_vnf = self.sfc.get_dst_vnf()

        self.src_substrate_node = self.sfc.get_substrate_node(src_vnf)
        self.dst_substrate_node = self.sfc.get_substrate_node(dst_vnf)

        node_latency, node_path = self.single_source_minimum_latency_path[self.dst_substrate_node]

        # Forward tracking
        vnf1 = src_vnf.get_next_vnf()
        self._dp(self.src_substrate_node, vnf1)

        vnf = vnf1.get_next_vnf()
        while vnf.id != dst_vnf.id:
            for node in nodes:
                self._dp(node, vnf)
            vnf = vnf.get_next_vnf()

        # Dst mapping processing
        previous_vnf = self.sfc.get_previous_vnf(dst_vnf)
        previous_vnf_id = previous_vnf.id
        bandwidth_request = self.sfc.get_link_bandwidth_request(previous_vnf_id, dst_vnf.id)

        # Uso direto de .items() (sem list())
        for node, latency in node_latency.items():
            if node == self.dst_substrate_node or node == self.src_substrate_node:
                continue

            # Verificação de banda
            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[node][previous_vnf_id]["bandwidth_usage_info"]
            )

            path = node_path[node]
            length = len(path)

            for i in range(0, length - 1):
                edge_key = frozenset((path[i], path[i + 1]))

                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = (
                        get_link_bandwidth_free(self.graph, path[i], path[i + 1])
                        - bandwidth_request
                    )

                if residual_bandwidth < 0:
                    is_bandwidth_sufficient = False
                    break

                bandwidth_usage_info[edge_key] = residual_bandwidth

            if not is_bandwidth_sufficient:
                continue

            _latency = self.node_info[node][previous_vnf_id]["latency"]
            aux1 = self.node_info[self.dst_substrate_node][dst_vnf.id]["latency"]
            aux2 = self.node_info[self.dst_substrate_node][dst_vnf.id]["latency"]

            if not aux1 or _latency + latency < aux2:
                self.node_info[self.dst_substrate_node][dst_vnf.id]["latency"] = _latency + latency
                self.node_info[self.dst_substrate_node][dst_vnf.id]["path"] = node_path[node]
                self.node_info[self.dst_substrate_node][dst_vnf.id]["path"].reverse()

                self.node_info[self.dst_substrate_node][dst_vnf.id]["current_substrate_nodes"] = (
                    self.node_info[node][previous_vnf_id]["current_substrate_nodes"][:]
                )

                self.node_info[self.dst_substrate_node][dst_vnf.id][
                    "current_substrate_nodes"
                ].append(self.dst_substrate_node)

                self.node_info[self.dst_substrate_node][dst_vnf.id]["src_path"] = (
                    self.node_info[node][previous_vnf_id]["src_path"][:]
                    + self.node_info[self.dst_substrate_node]["dst"]["path"][:]
                )

                self.node_info[self.dst_substrate_node][dst_vnf.id]["flag"] = True

        if self.node_info[self.dst_substrate_node][dst_vnf.id]["flag"]:
            # Backtracking
            previous_vnf = dst_vnf
            previous_substrate_node = self.dst_substrate_node

            while True:
                path = self.node_info[previous_substrate_node][previous_vnf.id]["path"]
                if not path:
                    break

                previous_substrate_node = path[0]
                previous_vnf = self.sfc.get_previous_vnf(previous_vnf)

                if previous_vnf:
                    self.route_info[previous_vnf.id] = path
                else:
                    break

            self.route_info[dst_vnf.id] = []

            if "src" not in self.route_info:
                return True

            # Reconstrução limpa da latência
            total_latency = 0.0
            for _sf, path in self.route_info.items():
                if path and len(path) > 1:
                    for i in range(len(path) - 1):
                        total_latency += get_link_latency(self.graph, path[i], path[i + 1])

            self.latency = total_latency

            # Validação final
            if self.latency > self.sfc.get_latency_request() or self.latency < 0:
                self.route_info = {}
                self.latency = None
                return False

            expected_length = len(self.sfc.vnfs) + 2
            if len(self.route_info) != expected_length:
                self.route_info = False
                self.latency = None
                return False

            return True

        else:
            return False

    def _dp(self, substrate_node: NodeID, vnf: Any) -> bool:
        """
        Calcula caminhos e latências a partir do substrate_node para os vizinhos viáveis.
        """
        previous_vnf = self.sfc.get_previous_vnf(vnf)
        previous_vnf_id = previous_vnf.id
        vnf_id = vnf.id

        node_latency, node_path = self.single_source_minimum_latency_path[substrate_node]
        _latency = self.node_info[substrate_node][previous_vnf_id]["latency"]

        cpu_request = self.sfc.get_vnf_cpu_request(vnf)
        cache_request = self.sfc.get_vnf_cache_request(vnf)
        bandwidth_request = self.sfc.get_link_bandwidth_request(previous_vnf_id, vnf_id)

        # Iteração direta (view) — sem list()
        for node, latency in node_latency.items():
            sfc_max_latency = self.sfc.get_latency_request()

            # Se o melhor cenário possível já estoura o limite da SFC, poda a rota!
            if (_latency + latency) > sfc_max_latency:
                continue

            forbidden = any(
                node_f == node and vnf_f == vnf_id
                for vnf_f, node_f in self.forbidden_matches.items()
            )
            if forbidden:
                continue

            if node in (substrate_node, self.src_substrate_node):
                continue

            if node in self.node_info[substrate_node][previous_vnf_id]["current_substrate_nodes"]:
                continue

            # Recursos CPU / Cache
            cpu_available = (
                self.graph.nodes[node]["cpu_capacity"] - self.graph.nodes[node]["cpu_used"]
            )
            cache_available = (
                self.graph.nodes[node]["cache_capacity"] - self.graph.nodes[node]["cache_used"]
            )

            # Router não pode hospedar VNF
            if self.graph.nodes[node].get("type") == "router":
                continue

            if cpu_request > cpu_available + 0.00001 or cache_request > cache_available + 0.00001:
                continue

            # Verificação de banda
            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[substrate_node][previous_vnf_id]["bandwidth_usage_info"]
            )

            path = node_path[node]
            if path[0] != substrate_node:
                path.reverse()

            length = len(path)

            for i in range(0, length - 1):
                edge_key = frozenset((path[i], path[i + 1]))

                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = (
                        get_link_bandwidth_free(self.graph, path[i], path[i + 1])
                        - bandwidth_request
                    )

                if residual_bandwidth < 0:
                    is_bandwidth_sufficient = False
                    break

                bandwidth_usage_info[edge_key] = residual_bandwidth

            if not is_bandwidth_sufficient:
                continue

            self.node_info[node][vnf_id]["bandwidth_usage_info"] = bandwidth_usage_info

            if (
                not self.node_info[node][vnf_id]["latency"]
                or (_latency + latency) <= self.node_info[node][vnf_id]["latency"]
            ):
                self.node_info[node][vnf_id]["latency"] = _latency + latency
                self.node_info[node][vnf_id]["path"] = node_path[node]
                self.node_info[node][vnf_id]["flag"] = True
                self.node_info[node][vnf_id]["previous_substrate_node"] = substrate_node

                self.node_info[node][vnf_id]["current_substrate_nodes"] = self.node_info[
                    substrate_node
                ][previous_vnf_id]["current_substrate_nodes"][:]

                self.node_info[node][vnf_id]["current_substrate_nodes"].append(node)

                self.node_info[node][vnf_id]["src_path"] = (
                    self.node_info[substrate_node][previous_vnf_id]["src_path"][:]
                    + node_path[node][:-1]
                )

        return True

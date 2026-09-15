
import networkx as nx

from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
)
from muar_sfc.core.vnf import VNF


def calculate_comunication_latency(graph: nx.Graph, path: list, vnf: VNF):
    """
    Calcula a latência total de um caminho dado e de uma VNF.

    A latência de rede entre os nós ao longo do caminho.

    :param graph: O grafo que representa a rede, com informações sobre os links e servidores.
    :param path: Lista de nós representando o caminho de alocação do serviço.
    :param vnf: O VNF (função de rede virtual) que está sendo alocado.
    :return: A latência total (latência computacional + latência de rede).
    """
    total_latency = 0

    # Latência de rede (entre os nós do caminho)
    edge_latency = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        # Cálculo da latência de rede entre os nós u e v
        edge_latency += calculate_latency_betwen_nodes(graph, u, v, vnf)
    total_latency += edge_latency

    return total_latency


def create_route_info_from_allocation_results(dst, graph: nx.Graph, allocation_results):
    if not allocation_results["dst"]:
        allocation_results["dst"] = {"allocated_server": dst, "path": [], "cost": 0}
    route_info = {key: list(reversed(value["path"])) for key, value in allocation_results.items()}
    src_node = next(reversed(route_info.values()))[0]
    path_to_src = list(reversed(nx.dijkstra_path(graph, src_node, 0, weight="weight")))
    route_info["src"] = path_to_src
    return route_info


def calculate_total_latency(graph: nx.Graph, path: list, vnf: VNF):
    """
    Calcula a latência total de um caminho dado e de uma VNF.

    A latência total é composta pela latência computacional no último nó
    (onde a VNF é alocada) e pela latência de rede entre os nós ao longo do caminho.

    :param graph: O grafo que representa a rede, com informações sobre os links e servidores.
    :param path: Lista de nós representando o caminho de alocação do serviço.
    :param vnf: O VNF (função de rede virtual) que está sendo alocado.
    :return: A latência total (latência computacional + latência de rede).
    """
    total_latency = 0

    # Latência de rede (entre os nós do caminho)
    edge_latency = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        # Cálculo da latência de rede entre os nós u e v
        edge_latency += calculate_latency_betwen_nodes(graph, u, v, vnf)
    total_latency += edge_latency

    # Latência computacional (apenas no último nó, onde a VNF é alocada)
    last_server = path[-1]  # Último nó do caminho
    comp_latency = calculate_computational_latency(graph, last_server, vnf)
    total_latency += comp_latency

    return total_latency

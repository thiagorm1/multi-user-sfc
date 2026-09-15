
import networkx as nx

from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
)
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC
from muar_sfc.core.vnf import VNF


class EnergyCalculator:
    """
    Objeto responsável por calcular o consumo de energia de uma rede com base
    nas regras do artigo e nas modificações solicitadas.
    """

    def __init__(self):
        """
        Inicializa o calculador com os dados de potência (em Watts) da Tabela 7 do artigo.
        A estrutura armazena os níveis de potência como 'low', 'medium' e 'high' para
        cada tipo de nó.
        """
        self._server_specs = {
            "a": {
                "low": 30,  # Frequência 2.0 GHz
                "medium": 40,  # Frequência 2.5 GHz
                "high": 50,  # Frequência 3.0 GHz
            },
            "b": {
                "low": 75,  # Frequência 4.0 GHz
                "medium": 100,  # Frequência 5.0 GHz
                "high": 120,  # Frequência 6.5 GHz
            },
            "c": {
                "low": 150,  # Frequência 8.0 GHz
                "medium": 175,  # Frequência 10.0 GHz
                "high": 220,  # Frequência 12.0 GHz
            },
        }

    def _get_power_for_node(self, graph, node) -> float:
        """
        Calcula a potência de um único nó com base na sua utilização de CPU,
        aumentando linearmente (servidores) ou em faixas (dispositivos móveis).
        """
        node_data = graph.nodes[node]
        if node_data["cpu_capacity"] == 0:
            return 0

        # Calcula a porcentagem de utilização da CPU
        cpu_utilization = node_data["cpu_used"] / node_data["cpu_capacity"]

        # --- LÓGICA ATUALIZADA PARA MOBILE DEVICE ---
        if node_data["type"] == "mobile_device":
            if cpu_utilization > 0.75:
                return 30.0  # Acima de 75%
            elif cpu_utilization > 0.50:
                return 18.0  # Entre 50% e 75%
            elif cpu_utilization > 0.25:
                return 14.4  # Entre 25% e 50%
            else:
                return 10.0  # Entre 0% e 25%
        # --- FIM DA ATUALIZAÇÃO ---

        # Seleciona as especificações do servidor com base no seu nível
        node_type_specs = self._server_specs[node_data["level_server"]]
        low_power = node_type_specs["low"]
        medium_power = node_type_specs["medium"]
        high_power = node_type_specs["high"]

        # Define os limites (thresholds)
        MEDIUM_THRESHOLD = 1 / 3
        HIGH_THRESHOLD = 2 / 3

        # Regra 1: Acima de 2/3 de utilização, o gasto é sempre o máximo.
        if cpu_utilization > HIGH_THRESHOLD:
            return high_power

        # Regra 2: Entre 1/3 e 2/3, interpolação linear entre 'medium' e 'high'.
        elif cpu_utilization > MEDIUM_THRESHOLD:
            # Calcula a proporção do uso de CPU dentro deste intervalo específico
            # O intervalo vai de 1/3 a 2/3, então seu tamanho é 1/3.
            progress_in_range = (cpu_utilization - MEDIUM_THRESHOLD) / MEDIUM_THRESHOLD
            # Aplica a interpolação linear
            return medium_power + (high_power - medium_power) * progress_in_range

        # Regra 3: De 0 a 1/3, interpolação linear entre 'low' e 'medium'.
        elif cpu_utilization > 0:
            # Calcula a proporção do uso de CPU dentro do primeiro intervalo (0 a 1/3)
            progress_in_range = cpu_utilization / MEDIUM_THRESHOLD
            # Aplica a interpolação linear
            return low_power + (medium_power - low_power) * progress_in_range

        # Caso base: Se a utilização for 0, o consumo é o mínimo.
        else:
            return low_power

    def calculate_total_network_power(self, network: Net2) -> float:
        """
        Calcula a potência total instantânea da rede (em Watts).

        Este método itera sobre todos os nós na rede, calcula a potência
        de cada um com base na sua carga de CPU atual e soma tudo.

        Args:
            network: Um objeto de rede que contém o atributo 'nodes'.

        Returns:
            A potência total consumida pela rede em Watts (Joules por segundo).
        """
        graph = network.graph
        total_power = 0.0
        for node in graph.nodes:
            power_for_node = self._get_power_for_node(graph, node)
            # A lógica customizada para nós específicos foi mantida
            if node % 1 == 0.1:
                power_for_node *= 5.5
            total_power += power_for_node
        return total_power

    # --- NOVOS MÉTODOS ADICIONADOS ---

    def calculate_total_server_power(self, network: Net2) -> float:
        """
        Calcula a potência total instantânea gasta apenas pelos servidores.

        Itera sobre todos os nós, mas soma apenas aqueles que NÃO são
        do tipo 'mobile_device', aplicando a mesma lógica de cálculo
        do método principal.

        Args:
            network: Um objeto de rede que contém o atributo 'nodes'.

        Returns:
            A potência total consumida pelos servidores em Watts.
        """
        graph = network.graph
        total_server_power = 0.0
        for node in graph.nodes:
            node_data = graph.nodes[node]

            # Filtra apenas por servidores (excluindo dispositivos móveis)
            if node_data.get("type") != "mobile_device":
                power_for_node = self._get_power_for_node(graph, node)

                # Mantém a lógica customizada do método original
                if node % 1 == 0.1:
                    power_for_node *= 1.2
                total_server_power += power_for_node

        return total_server_power

    def calculate_total_mobile_device_power(self, network: Net2) -> float:
        """
        Calcula a potência total instantânea gasta apenas pelos dispositivos móveis.

        Itera sobre todos os nós, mas soma apenas aqueles que são
        do tipo 'mobile_device'.

        Args:
            network: Um objeto de rede que contém o atributo 'nodes'.

        Returns:
            A potência total consumida pelos dispositivos móveis em Watts.
        """
        graph = network.md_graph
        total_mobile_power = 0.0
        for node in graph.nodes:
            node_data = graph.nodes[node]

            # Filtra apenas por dispositivos móveis
            if node_data.get("type") == "mobile_device":
                power_for_node = self._get_power_for_node(graph, node)

                # Mantém a lógica customizada do método original
                if float(node) % 1 == 0.1:
                    power_for_node *= 1.2
                total_mobile_power += power_for_node

        return total_mobile_power


def calcular_percentual_cpu_total(G: nx.Graph) -> float:
    """
    Calcula o percentual de uso de CPU total em um grafo do NetworkX.

    A função itera sobre todos os nós do grafo, soma os valores dos atributos
    'cpu_used' and 'cpu_capacity', e retorna o percentual total de uso.

    Args:
        G (nx.Graph): O grafo do NetworkX cujos nós contêm os atributos
                      'cpu_used' e 'cpu_capacity'.

    Returns:
        float: O percentual total de uso de CPU (de 0.0 a 100.0).
               Retorna 0.0 se a capacidade total for 0 para evitar divisão por zero.
    """
    total_cpu_used = 0.0
    total_cpu_capacity = 0.0

    # Iteramos sobre os nós com seus dados (atributos)
    for _node_id, node_data in G.nodes(data=True):
        # Usamos .get(atributo, 0) para o caso de um nó não ter o atributo.
        # Isso torna a função mais robusta e evita erros.
        total_cpu_used += node_data.get("cpu_used", 0)
        total_cpu_capacity += node_data.get("cpu_capacity", 0)

    # Verifica se a capacidade total é zero para evitar erro de divisão
    if total_cpu_capacity == 0:
        return 0.0

    # Calcula o percentual
    percentual_uso = (total_cpu_used / total_cpu_capacity) * 100

    return percentual_uso


def get_graph_processing_utilization_simplified(graph: nx.Graph) -> float:
    """
    Calcula a porcentagem de utilização de processamento total (CPU + GPU)
    para um grafo específico, tratando todos os nós de processamento
    da mesma forma.
    """
    total_processing_used = 0.0
    total_processing_capacity = 0.0

    # Itera sobre todos os nós no grafo fornecido
    for _node_id, node_data in graph.nodes(data=True):
        # Verifica se o nó tem capacidade de processamento
        if "cpu_capacity" in node_data:
            total_processing_capacity += node_data.get("cpu_capacity", 0.0)
            total_processing_used += node_data.get("cpu_used", 0.0)

    # Evita divisão por zero
    if total_processing_capacity == 0:
        return 0.0

    # Calcula e retorna a porcentagem
    utilization_percentage = (total_processing_used / total_processing_capacity) * 100
    return utilization_percentage


def calcular_percentual_cache_total(G: nx.Graph) -> float:
    """
    Calcula o percentual de uso de cache total em um grafo do NetworkX.

    A função itera sobre todos os nós do grafo, soma os valores dos atributos
    'cache_used' and 'cache_capacity', e retorna o percentual total de uso.

    Args:
        G (nx.Graph): O grafo do NetworkX cujos nós contêm os atributos
                      'cache_used' e 'cache_capacity'.

    Returns:
        float: O percentual total de uso de cache (de 0.0 a 100.0).
               Retorna 0.0 se a capacidade total for 0 para evitar divisão por zero.
    """
    total_cache_used = 0.0
    total_cache_capacity = 0.0

    # Iteramos sobre os nós com seus dados (atributos)
    for _node_id, node_data in G.nodes(data=True):
        # Usamos .get(atributo, 0) para o caso de um nó não ter o atributo.
        # Isso torna a função mais robusta e evita erros.
        total_cache_used += node_data.get("cache_used", 0)
        total_cache_capacity += node_data.get("cache_capacity", 0)

    # Verifica se a capacidade total é zero para evitar erro de divisão
    if total_cache_capacity == 0:
        return 0.0

    # Calcula o percentual
    percentual_uso = (total_cache_used / total_cache_capacity) * 100

    return percentual_uso


def calcular_percentual_banda_total(graph: nx.Graph) -> float:
    """
    Calcula o percentual de uso da largura de banda total da rede.

    Args:
        graph: O grafo da rede.

    Returns:
        A porcentagem de banda utilizada (de 0 a 100).
    """
    total_banda_usada = 0.0
    total_banda_capacidade = 0.0

    # Itera sobre todas as arestas (links) do grafo
    for _u, _v, data in graph.edges(data=True):
        total_banda_usada += data.get("bandwidth_used", 0)
        total_banda_capacidade += data.get("bandwidth_capacity", 0)

    # Evita divisão por zero se a rede não tiver capacidade
    if total_banda_capacidade == 0:
        return 0.0

    # Retorna o resultado como uma porcentagem
    return (total_banda_usada / total_banda_capacidade) * 100


def get_sfc_latency_from_route(graph: nx.Graph, sfc: SFC, route_info, md_graph: nx.Graph = None):
    """
    Calcula a latência total de uma solução de rota para uma SFC (função pura).

    Esta função é 'read-only': ela lê os dados do grafo para calcular a latência,
    mas NÃO aloca recursos nem modifica o estado do grafo. Ideal para avaliar
    uma solução antes de implementá-la.

    Args:
        graph (object): O grafo da rede do substrato.
        sfc (object): A Service Function Chain a ser avaliada.
        route_info (dict): O dicionário contendo a rota da solução.

    Returns:
        float: A latência total calculada para a SFC.
    """
    total_latency = 0.0

    # Assume-se que as funções `calculate_computational_latency` e
    # `calculate_latency_betwen_nodes` estão acessíveis (ex: importadas ou
    # são métodos da classe).

    for ms_name, path in route_info.items():
        if ms_name in ["src", "dst"]:
            continue

        vnf = sfc.get_vnf_by_id(ms_name)
        node_allocated = path[0]

        # 1. Calcula e soma a latência computacional
        # Esta chamada apenas calcula, sem alocar CPU/cache.
        if node_allocated in graph:
            comp_latency = calculate_computational_latency(graph, node_allocated, vnf=vnf)
        else:
            comp_latency = calculate_computational_latency(md_graph, node_allocated, vnf=vnf)
        total_latency += comp_latency

        # 2. Calcula e soma a latência de comunicação para cada enlace
        if len(path) > 1:
            for u, v in zip(path[:-1], path[1:], strict=True):
                # Esta chamada apenas calcula, sem alocar banda.
                comm_latency = calculate_latency_betwen_nodes(graph, u, v, vnf, md_graph)
                total_latency += comm_latency

    return round(total_latency, 2)


def calculate_average_sfc_latency(substrate_network: Net2):
    """
    Calcula a latência média ponta a ponta de todas as SFCs ativas na rede.

    A latência de uma SFC é a soma das latências computacionais de suas VNFs
    e das latências de comunicação dos caminhos entre elas. A função retorna
    a média dessa latência total sobre todas as SFCs implantadas.

    Returns:
        float: A latência média por SFC, ou 0 se nenhuma SFC estiver ativa.
    """
    if not substrate_network.sfc_dict:
        return 0.0

    total_latency_all_sfcs = 0.0

    for sfc_id, sfc in substrate_network.sfc_dict.items():
        graph = substrate_network.graph
        md_graph = substrate_network.md_graph
        route_info = substrate_network.sfc_route_info[sfc_id]
        total_latency_all_sfcs += get_sfc_latency_from_route(graph, sfc, route_info, md_graph)

    return total_latency_all_sfcs / len(substrate_network.sfc_dict)


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

def check_vnf_reusability(node_services: dict, target_clean_id: str, target_session_id: str | int) -> bool:
    """
    Verifica de forma estrita se uma VNF pode ser reusada no nó,
    garantindo sincronia absoluta de tipos para evitar alucinações na IA.
    """
    if not node_services:
        return False

    for (ex_id, ex_sess) in node_services.keys():
        clean_ex_id = ex_id.replace("_b", "")
        
        # Validação estrita: sem cast de str() obscuro. 
        # O tipo do ex_sess deve bater exatamente com o target_session_id.
        if clean_ex_id == target_clean_id and ex_sess == target_session_id:
            return True
            
    return False

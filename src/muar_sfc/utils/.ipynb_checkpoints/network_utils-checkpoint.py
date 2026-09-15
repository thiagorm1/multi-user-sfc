from muar_sfc.core.net_v2 import Net2
import networkx as nx

class EnergyCalculator:
    """
    Objeto responsável por calcular o consumo de energia de uma rede com base
    nas regras do artigo e nas modificações solicitadas.
    """
    def __init__(self):
        """
        Inicializa o calculador com os dados de potência (em Watts) da Tabela 7 do artigo.
        A estrutura armazena os níveis de potência como 'low', 'medium' e 'high' para cada tipo de nó.
        """
        self._server_specs = {
            'a': {
                'low': 30,     # Frequência 2.0 GHz
                'medium': 40,  # Frequência 2.5 GHz
                'high': 50,    # Frequência 3.0 GHz
            },
            'b': {
                'low': 75,     # Frequência 4.0 GHz
                'medium': 100, # Frequência 5.0 GHz
                'high': 120,   # Frequência 6.5 GHz
            },
            'c': {
                'low': 150,    # Frequência 8.0 GHz
                'medium': 175, # Frequência 10.0 GHz
                'high': 220,   # Frequência 12.0 GHz
            }
        }

    def _get_power_for_node(self, graph, node) -> float:
        """
        Calcula a potência de um único nó com base na sua utilização de CPU,
        aumentando linearmente entre os níveis de gasto e atingindo o máximo
        após 2/3 de utilização.
        """
        node_data = graph.nodes[node]
        if node_data['cpu_capacity'] == 0:
            return 0
        
        # Calcula a porcentagem de utilização da CPU
        cpu_utilization = node_data["cpu_used"] / node_data["cpu_capacity"]

        if node_data["type"] == "mobile_device":
            return 7.5

        # Seleciona as especificações do servidor com base no seu nível
        node_type_specs = self._server_specs[node_data["level_server"]]
        low_power = node_type_specs['low']
        medium_power = node_type_specs['medium']
        high_power = node_type_specs['high']
        
        # Define os limites (thresholds)
        MEDIUM_THRESHOLD = 1/3
        HIGH_THRESHOLD = 2/3

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
                power_for_node *= 1.2
            total_power += power_for_node
        return total_power
    
    
    

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
    for node_id, node_data in G.nodes(data=True):
        # Usamos .get(atributo, 0) para o caso de um nó não ter o atributo.
        # Isso torna a função mais robusta e evita erros.
        total_cpu_used += node_data.get('cpu_used', 0)
        total_cpu_capacity += node_data.get('cpu_capacity', 0)

    # Verifica se a capacidade total é zero para evitar erro de divisão
    if total_cpu_capacity == 0:
        return 0.0

    # Calcula o percentual
    percentual_uso = (total_cpu_used / total_cpu_capacity) * 100
    
    return percentual_uso


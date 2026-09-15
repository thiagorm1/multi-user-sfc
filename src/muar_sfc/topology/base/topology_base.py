from abc import ABC, abstractmethod


class TopologyBase(ABC):
    @abstractmethod
    def generate_substrate_network(self):
        """Gera a rede substrato."""
        pass

    @abstractmethod
    def get_topology_info(self):
        """Retorna informações da topologia como nós, arestas e servidores de processamento."""
        pass

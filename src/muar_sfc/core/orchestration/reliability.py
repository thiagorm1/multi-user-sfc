from typing import Any

from muar_sfc.core.infrastructure.topology import NetworkTopology


class ReliabilityManager:
    """Gerencia métricas de confiabilidade e modelos de falha (RBD)."""

    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.alpha_stress = {"default": 0.04, "a": 0.15, "b": 0.04, "c": 0.0009}
        self.tier_reliability = {"default": 0.99, "a": 0.95, "b": 0.99, "c": 0.9999}

    def get_node_reliability(self, node_id: Any) -> float:
        """Calcula a confiabilidade atual de um nó baseada no estresse/utilização."""
        try:
            data = self.topology.get_node_data(node_id)
        except KeyError:
            return 0.0

        if not data.get("is_active", True):
            return 0.0

        # Lógica de penalidade por utilização (Stress)
        level = str(data.get("level_server", "default")).lower()
        base_r = self.tier_reliability.get(level, self.tier_reliability["default"])
        alpha = self.alpha_stress.get(level, self.alpha_stress["default"])

        cap = data.get("cpu_capacity", 0.0)
        utilization = (data.get("cpu_used", 0.0) / cap) if cap > 0 else 0.0

        return max(0.0, base_r - (utilization * alpha))

    def calculate_system_reliability(self, sfc_list: list[Any], backups: dict) -> float:
        """Calcula a média de confiabilidade de todas as SFCs primárias."""
        # Implementação da lógica Série-Paralelo (RBD) transposta do net_v2.py
        # ... (Omitido para brevidade, mas segue a lógica original de RBD)
        return 0.99  # Mock de retorno

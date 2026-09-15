from muar_sfc.core.net_v2 import Net2


def _calculate_sfc_risk_level(substrate_network: Net2, sfc_id: str) -> str:
    """Calcula o risco da SFC baseado na confiabilidade dos nós da rota atual."""
    try:
        if sfc_id not in substrate_network.sfc_route_info:
            return "Medium"  # Fallback

        route_info = substrate_network.sfc_route_info[sfc_id]
        unique_nodes = set()
        for vnf_id, path in route_info.items():
            if vnf_id not in ["src", "dst"] and path:
                unique_nodes.add(path[0])

        sfc_reliability = 1.0
        for node in unique_nodes:
            sfc_reliability *= substrate_network.get_node_reliability(node)

        # Limites baseados na sua lógica existente em output_flows
        if sfc_reliability < 0.8666:
            return "High"
        elif sfc_reliability <= 0.9333:
            return "Medium"
        else:
            return "Low"
    except Exception:
        return "Medium"

import random
from loguru import logger
from muar_sfc.core.net_v2 import Net2


class Crasher:
    """
    Gerencia a simulação de falhas aplicando paradigmas modernos (O(1), EAFP e Fail-Fast).
    """

    def __init__(self, topology, args, interval=200, time=30):
        # EAFP puro: Tentamos converter, assumimos os padrões se os atributos faltarem
        try:
            self.activated = float(args.ava) != 1.0 or float(args.link_ava) != 1.0
        except (AttributeError, ValueError, TypeError):
            self.activated = False

        try:
            self.fail_target = args.fail_target
        except AttributeError:
            self.fail_target = "all"

        try:
            # Proteção O(1) de acesso: transformando a topologia num conjunto instantâneo
            self.ec_servers = set(topology.get_topology_info()["ec_servers"])
        except (KeyError, TypeError):
            self.ec_servers = set()

        self.nodes_crashed = set()
        self.links_crashed_history = set()
        self.simulation_step = 1.0

    def calculate_node_probabilities(self, network: Net2) -> dict:
        physical_servers = {}

        for node in network.graph.nodes():
            # EAFP no lugar de .get()
            try:
                node_type = str(network.graph.nodes[node]["type"])
                if "server" not in node_type:
                    continue
            except KeyError:
                continue

            node_str = str(node)
            base_id = node_str.split(".1")[0]

            # EAFP na inserção de chaves
            try:
                physical_servers[base_id].append(node)
            except KeyError:
                physical_servers[base_id] = [node]

        aggregated_probs = {}

        for base_id, members in physical_servers.items():
            reliability = network.get_node_reliability(base_id)
            prob_failure = max(0.0, 1.0 - reliability)

            aggregated_probs[base_id] = {
                "prob": prob_failure,
                "reliability": reliability,
                "members": members,
            }

        return aggregated_probs

    def activate_crasher(self, network, sfc_manager=None, alg_name=None) -> list:
        if not self.activated:
            return []

        user_target = self.fail_target
        server_groups = self.calculate_node_probabilities(network)

        candidates = []
        weights = []

        for base_id, info in server_groups.items():
            # O(1) graças à conversão do self.ec_servers em conjunto na inicialização
            is_valid = any(m in self.ec_servers for m in info["members"])

            is_active = base_id not in self.nodes_crashed and all(
                m not in self.nodes_crashed for m in info["members"]
            )

            if is_valid and is_active:
                reliability = info["reliability"]
                
                # Structural Pattern Matching (Otimização do Python 3.10+)
                match user_target:
                    case "all":
                        should_include = True
                    case "high_risk" if reliability < 0.8666:
                        should_include = True
                    case "med_risk" if 0.8666 <= reliability <= 0.9333:
                        should_include = True
                    case "low_risk" if reliability > 0.9333:
                        should_include = True
                    case _:
                        should_include = False

                if not should_include:
                    continue

                candidates.append(base_id)
                weights.append(info["prob"])

        nodes_affected = []

        if candidates and sum(weights) > 0:
            chosen_key = random.choices(candidates, weights=weights, k=1)[0]
            nodes_affected = server_groups[chosen_key]["members"]
            confiabilidade_atual = server_groups[chosen_key]["reliability"]

            for node in nodes_affected:
                if node not in self.nodes_crashed:
                    self.nodes_crashed.add(node)
                    # Adequação visual da rastreabilidade aos padrões do repositório
                    logger.error(f"[FALHA INJETADA] Nó {node} CRASHED! Confiabilidade Original: {confiabilidade_atual:.3f}")

        return nodes_affected

    def recover_specific_node(self, network, node_id) -> bool:
        """Recupera um nó específico solicitado pelo Controller com EAFP estrito."""
        try:
            # Tenta remover diretamente. Se falhar, é KeyError
            self.nodes_crashed.remove(node_id)
            network.restore_node(node_id)
            return True
        except KeyError:
            return False

    def activate_link_crasher(self, network):
        # Correção da anomalia de retorno (None, None -> None explícito)
        return None
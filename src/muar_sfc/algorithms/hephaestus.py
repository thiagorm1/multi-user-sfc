import logging
import os

import networkx as nx
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO

from muar_sfc.algorithms.environments.environment import SFC_AllocationEnv
from muar_sfc.config import ROOT_DIR
from muar_sfc.core.sfc import SFC

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(ROOT_DIR, "logs/hephaestus.log"))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Constants
N_STEPS = 256
IS_TRAINING = 0
VERBOSE = False
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Desabilita o uso da GPU


class hephaestus:
    ### MODIFICADO ###
    # O construtor agora carrega o modelo imediatamente e recebe o TIPO do modelo.
    def __init__(self, model_name: str):
        # --- Atributos ---
        self.model_name = model_name.upper()  # Ex: "PPO" ou "MASKABLEPPO"
        self.name = f"{self.model_name}_hephaestus"  # Nome agora reflete o tipo

        self.model_path = f"rl_saved_models/{self.name}_allocation_model.zip"

        # --- ADICIONADO: Carregamento do modelo na inicialização ---
        # O modelo é carregado UMA ÚNICA VEZ e mantido na memória.
        self.model = self._load_model()

        # Demais atributos da sua classe (mantidos do original)
        # ...

        # Demais atributos da sua classe (mantidos do original)
        self.graph = None
        self.sfc = None
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.latency_request = None
        self.single_source_minimum_latency_path = None
        self.fail_reason = None
        self.can_host_multiple_sfs = True
        self.is_backup = False
        self.valid_nodes = None
        self.last_propose = None
        self.precomputed_paths = {}
        # self.env foi removido pois não pertence mais à classe.

        # Pesos de custo (mantidos do original)
        self.cpu_factor = 5
        self.cache_factor = 5
        self.band_factor = 2
        self.latency_factor = 2
        self.boot_factor = 0

    ### ADICIONADO ###
    # Método privado para carregar o modelo, chamado apenas uma vez.
    def _load_model(self):
        """Carrega o modelo de RL do arquivo, sem precisar de um ambiente."""
        if not os.path.exists(self.model_path):
            logger.error(f"Arquivo do modelo não encontrado: {self.model_path}")
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")

        logger.info(f"Carregando modelo de: {self.model_path} (tipo esperado: {self.model_name})")

        if self.model_name == "MASKABLEPPO":
            return MaskablePPO.load(self.model_path, device="cpu")
        elif self.model_name == "PPO":
            return PPO.load(self.model_path, device="cpu")
        elif self.model_name == "DQN":
            return DQN.load(self.model_path, device="cpu")
        else:
            raise ValueError(
                f"Nome do modelo inválido: '{self.model_name}'. Use 'PPO', 'MaskablePPO' ou 'DQN'."
            )

    # O método clear_all foi mantido como no original.
    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        self.src_substrate_node = None
        self.single_source_minimum_latency_path = None

    # O método install_substrate_network foi mantido como no original.
    def install_substrate_network(self, graph, shareable_sfs=None):
        if shareable_sfs is None:
            shareable_sfs = []
        self.graph = graph
        self.valid_nodes = [
            node for node in self.graph.nodes() if self.graph.nodes[node]["type"] != "router"
        ]

        if self.precomputed_paths is None:
            self.precomputed_paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight="weight"))

    # O método install_SFC foi mantido como no original.
    def install_SFC(self, sfc: SFC):
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        is_backup = sfc.id.split("_")[2] == "backup"

        self.latency_request = sfc.get_latency_request()
        self.min_latency = 0

        service_requirements = {}
        services = []
        sfs_dict = sfc.vnfs_dict

        for item in sfs_dict:
            nome = item["name"]
            services.append(nome)
            service_requirements[nome] = {
                "CPU": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        if not is_backup:
            services.append("dst")
            service_requirements["dst"] = {"CPU": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

        self.service_requirements = service_requirements
        self.services = services

        return self.sfc

    # Todos os métodos getters e de utilidade foram mantidos como no original.
    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def get_fail_reason(self):
        return self.fail_reason

    def handle_failure(self):
        self.route_info = False
        self.latency = None

    def check_solution(self):
        if (
            not isinstance(self.latency, (int, float))
            or not (0 <= self.latency <= self.sfc.get_latency_request())
            or not self.route_info
        ):
            return False
        if len(self.route_info) != 6:
            return False
        prev_path_end = None
        for sf, path in self.route_info.items():
            prev_sf = None
            if sf == "dst":
                continue

            if prev_path_end and path[-1] != prev_path_end:
                print(f"Inconsistência entre {prev_sf} e {sf}: {prev_path_end} != {path[0]}")
                return False
            prev_path_end = path[0]
            prev_sf = sf
        return True

    def set_costs(self, costs_parameters):
        self.cpu_factor, self.cache_factor, self.band_factor = costs_parameters

    # O método principal agora RECEBE a instância do ambiente.
    def start_algorithm(self, env: SFC_AllocationEnv):
        if not self.valid_nodes or not self.sfc or not self.graph:
            self.fail_reason = (
                "Erro: Rede ou SFC não foram instalados antes de chamar start_algorithm."
            )
            logger.error(self.fail_reason)
            self.handle_failure()
            return False

        # Garante que o ambiente está sincronizado com o grafo e SFC que o hephaestus recebeu.
        env.is_training = False
        self.fail_reason = None
        env.valid_nodes = self.valid_nodes
        env._set_list_graph_sfcs([self.graph], [self.sfc])
        self.model.set_env(env)  # Associa o modelo ao ambiente configurado

        # Chama a lógica principal do algoritmo, passando o ambiente
        self.algorithm(env)

        # A lógica de verificação e retorno foi mantida do original.
        if self.check_solution():
            try:
                logger.info("Finished algorithm, success")
                return True
            except Exception:
                self.handle_failure()
                return False
        else:
            self.handle_failure()
            logger.info(f"End algorithm, failed: {self.fail_reason}")
            return False

    # O método algorithm agora recebe e repassa o ambiente.
    def algorithm(self, env: SFC_AllocationEnv):
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())

        # Passa o ambiente para o método que executa o loop de predição.
        route_info, latency = self.find_best_allocation_for_sfc(env, dst)

        return self.evaluate_result(latency, route_info)

    # Este método foi simplificado para apenas executar o loop de predição.
    def find_best_allocation_for_sfc(self, env: SFC_AllocationEnv, dst):
        # A criação e reset do ambiente agora são feitos externamente.
        # Apenas executamos o loop de decisão.
        obs, _ = env.reset()
        env.is_training = False
        env.allocation_results["dst"] = {"allocated_server": dst, "path": [], "cost": 0}

        done = False
        while not done:
            # --- MODIFICADO: Chamada condicional do predict ---
            if self.model_name == "MASKABLEPPO":
                action_masks = env.action_masks()
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=False)
            else:
                # PPO Padrão e DQN não usam máscaras
                action, _ = self.model.predict(obs, deterministic=False)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if not env.success:
            if not VERBOSE:
                print(f"Causa Falha: {env.fail_reason}")
                print(f"Alocação: [{env.servers_used}] || Custo latência: {env.latency_used}")
            self.fail_reason = env.fail_reason
            return [], None

        print(f"SFC: {self.sfc.id}: {env.servers_used} || latência usada: {env.latency_used}")

        env.allocation_results["dst"] = {"allocated_server": dst, "path": [], "cost": 0}

        route_info = {
            key: list(reversed(value["path"])) for key, value in env.allocation_results.items()
        }

        src_node = next(reversed(route_info.values()))[0]
        path_to_src = list(reversed(nx.dijkstra_path(self.graph, src_node, 0, weight="weight")))
        route_info["src"] = path_to_src
        total_latency = env.latency_used + (len(path_to_src) - 1)
        return route_info, total_latency

    # O método evaluate_result foi mantido como no original.
    def evaluate_result(self, latency, route_info):
        if self.fail_reason in ["resource", "latency", "bandwidth"]:
            self.route_info = False
            self.latency = None
            return False
        self.latency = latency
        self.route_info = route_info
        return True

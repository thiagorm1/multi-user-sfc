import contextlib
import logging
import os
import sys

import networkx as nx
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO

from muar_sfc.algorithms.environments.env_replic import SFC_AllocationEnv
from muar_sfc.config import ROOT_DIR
from muar_sfc.core.sfc import SFC

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_dir = ROOT_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "REPLIC.log"

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

IS_TRAINING = 0
VERBOSE = False
# EAFP: O padrão TF/Torch para forçar CPU exige "-1" explícito.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@contextlib.contextmanager
def suppress_output():
    """Silencia o stdout (prints) dentro deste bloco."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class REPLIC:
    def __init__(self, model_name: str):
        self.model_name = str(model_name).upper()
        self.model_path = ROOT_DIR / "rl_saved_models" / f"{self.model_name}_REPLIC_allocation_model.zip"
        self.name = f"REPLIC{self.model_name}"

        self.model = self._load_model()

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

        self.cpu_factor = 5
        self.cache_factor = 5
        self.band_factor = 2
        self.latency_factor = 2
        self.boot_factor = 0

    def _load_model(self):
        """Carrega o modelo via EAFP e Pattern Matching moderno."""
        logger.info(f"Carregando modelo de: {self.model_path}")
        path_str = str(self.model_path)

        try:
            match self.model_name:
                case "MASKABLEPPO":
                    return MaskablePPO.load(path_str, device="cpu")
                case "PPO":
                    return PPO.load(path_str, device="cpu")
                case "DQN":
                    return DQN.load(path_str, device="cpu")
                case _:
                    raise ValueError(f"Nome do modelo inválido: '{self.model_name}'. Use 'PPO', 'MaskablePPO' ou 'DQN'.")
        except OSError as e:
            logger.error(f"Falha ao carregar o arquivo do modelo no caminho {self.model_path}: {e}")
            raise FileNotFoundError(f"Arquivo do modelo não encontrado ou corrompido: {self.model_path}") from e

    def clear_all(self):
        self.graph = None
        self.sfc = None
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        self.single_source_minimum_latency_path = None

    def install_substrate_network(self, graph, shareable_sfs=None):
        self.graph = graph
        
        # Acesso O(1) usando os atributos nativos do dict do NetworkX
        self.valid_nodes = [
            node for node, data in self.graph.nodes(data=True) if data.get("type") != "router"
        ]

        if not self.precomputed_paths:
            self.precomputed_paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight="weight"))

    def install_SFC(self, sfc: SFC):
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.is_backup = "backup" in sfc.id

        self.latency_request = sfc.get_latency_request()
        self.min_latency = 0

        self.services = []
        self.service_requirements = {}

        # O(1) Inserção de dados
        for item in sfc.vnfs_dict:
            nome = item["name"]
            self.services.append(nome)
            self.service_requirements[nome] = {
                "CPU": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        if not self.is_backup:
            self.services.append("dst")
            self.service_requirements["dst"] = {"CPU": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

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

    def check_solution(self):
        if self.latency is None or not self.route_info:
            return False

        if len(self.route_info) < 2:
            return False

        next_hop_start = None
        next_sf = None
        
        for sf, path in self.route_info.items():
            if sf == "dst":
                continue

            if next_hop_start is not None and path:
                if path[-1] != next_hop_start:
                    logger.warning(
                        f"Inconsistência de rota: O fim de '{sf}' (Nó {path[-1]}) "
                        f"não se conecta ao início de '{next_sf}' (Nó {next_hop_start})."
                    )
                    return False
                
            if path:
                next_hop_start = path[0]
                
            next_sf = sf
            
        return True

    def set_costs(self, costs_parameters):
        self.cpu_factor, self.cache_factor, self.band_factor = costs_parameters

    def start_algorithm(self, env: SFC_AllocationEnv, args=None):
        if not self.valid_nodes or not self.sfc or not self.graph:
            self.fail_reason = "Erro: Rede ou SFC não foram instalados."
            logger.error(self.fail_reason)
            self.handle_failure()
            return False

        env.is_training = False
        self.fail_reason = None
        env.valid_nodes = self.valid_nodes
        env._set_list_graph_sfcs([self.graph], [self.sfc])

        if args:
            env.reliability_config = {
                "tiers": {
                    "default": getattr(args, "rel_normal", 0.99),
                    "a": getattr(args, "rel_low", 0.95),
                    "b": getattr(args, "rel_normal", 0.98),
                    "c": getattr(args, "rel_high", 0.999),
                },
                "stress": {
                    "default": getattr(args, "stress_normal", 0.04),
                    "a": getattr(args, "stress_low", 0.15),
                    "b": getattr(args, "stress_normal", 0.08),
                    "c": getattr(args, "stress_high", 0.02),
                },
            }

        with suppress_output():
            self.model.set_env(env)

        self.algorithm(env)

        if self.check_solution():
            if not self.is_backup:
                logger.info("Finished algorithm, success")
            return True
        else:
            self.handle_failure()
            if not self.is_backup:
                logger.info(f"End algorithm, failed: {self.fail_reason}")
            return False

    def algorithm(self, env: SFC_AllocationEnv):
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        route_info, latency = self.find_best_allocation_for_sfc(env, dst)
        return self.evaluate_result(latency, route_info)

    def find_best_allocation_for_sfc(self, env: SFC_AllocationEnv, dst):
        obs, _ = env.reset()
        env.is_training = False

        env.allocation_results["dst"] = {"allocated_server": dst, "path": [], "cost": 0}
        done = False
        
        while not done:
            # CORREÇÃO VITAL: Imposição do deterministic=True para exterminar escolhas estocásticas
            if self.model_name == "MASKABLEPPO":
                action_masks = env.action_masks()
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if not env.success:
            self.fail_reason = env.fail_reason
            return {}, None

        route_info = {}
        for key, value in env.allocation_results.items():
            route_info[key] = list(reversed(value["path"])) if value["path"] else []

        total_latency = env.latency_used

        if not self.is_backup:
            if route_info:
                # O(1): Resgate do último elemento sem converter dicionário em lista
                first_vnf_key = next(reversed(route_info))
                
                if route_info[first_vnf_key]:
                    src_node_network = route_info[first_vnf_key][0]
                else:
                    src_node_network = env.allocation_results[first_vnf_key]["allocated_server"]

            try:
                path_to_src = list(
                    nx.dijkstra_path(self.graph, 0, src_node_network, weight="weight")
                )
                route_info["src"] = path_to_src
                total_latency += len(path_to_src) - 1
            except nx.NetworkXNoPath:
                self.fail_reason = "Sem rota para Cloud"
                return {}, None

        return route_info, total_latency

    def evaluate_result(self, latency, route_info):
        if self.fail_reason in ["resource", "latency", "bandwidth"]:
            self.handle_failure()
            return False
            
        self.latency = latency
        self.route_info = route_info
        return True
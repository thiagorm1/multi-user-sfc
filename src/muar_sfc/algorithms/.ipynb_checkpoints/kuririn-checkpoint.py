from typing import List, Union
from muar_sfc.core.sfc import SFC, VNF
import logging
import re
import networkx as nx
from stable_baselines3 import  DQN
from sb3_contrib import MaskablePPO

from muar_sfc.algorithms.environment import SFC_AllocationEnv
import os
from config import ROOT_PATH
# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(ROOT_PATH, 'logs/Kuririn.log'))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Constants
N_STEPS = 256
IS_TRAINING = 0
VERBOSE = False
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Desabilita o uso da GPU


class Kuririn:
    def __init__(self, model_name):
        # --- Atributos ---
        self.model_name = model_name
        self.model_path = f'rl_saved_models/{self.model_name}_allocation_model.zip'
        self.name = "kuririn"
        
        # O modelo é inicializado como None. Ele será carregado na primeira execução.
        self.model = None
        
        # Demais atributos da sua classe
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
        self.model = None
        self.env = None  # Apenas para type-hinting

        # Pesos de custo
        self.cpu_factor = 5
        self.cache_factor = 5
        self.band_factor = 2
        self.latency_factor = 2
        self.boot_factor = 0


    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        self.src_substrate_node = None
        self.single_source_minimum_latency_path = None

    def install_substrate_network(self, graph, shareable_sfs=[]):
        self.graph = graph

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        # self.dst_vnf = self.sfc.get_dst_vnf()
        # if self.env is None:
        #     self.env = SFC_AllocationEnv(
        #         valid_nodes=self.valid_nodes,
        #         list_graph=[self.graph],
        #         list_sfc=[self.sfc]
        #     )
        return sfc

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

    def start_algorithm(self):
        if not self.valid_nodes:
            self.valid_nodes = [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] != 'router' and node != 0]
        self._initialize_environment_and_model()

        self.algorithm()
        if self.check_solution():
            try:
                logger.info("Finished algorithm, success")
                if IS_TRAINING:
                    self._save_model()
                return True  
            except Exception:
                self.handle_failure()
                return False
        else:
            self.handle_failure()
            logger.info(f"End algorithm, failed: {self.fail_reason}")
            if IS_TRAINING:
                self._save_model()
            return False

    def algorithm(self):
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())

        route_info, latency = self.find_best_allocation_for_sfc(dst)

        return self.evaluate_result(latency, route_info)
    

    def find_best_allocation_for_sfc(self,dst):
        # 1. Não recria o ambiente, apenas reseta o necessário se já estiver inicializado
        if self.env is None:
            self.env = SFC_AllocationEnv(
                valid_nodes=self.valid_nodes,
                list_graph=[self.graph],
                list_sfc=[self.sfc]
            )
        else:
            # Apenas reseta o ambiente sem criar uma nova instância
            
            self.reset_environment([self.graph], [self.sfc])

        # 2. Lógica de carregamento adaptativo do modelo. Só carrega o modelo se ele não estiver carregado ainda
        if self.model is None:
            self.load_model(self.env)

        # 3. Prepara e reseta o ambiente para o início do episódio, mas não recria o ambiente
        obs, _ = self.env.reset()
        self.env.is_training = False  # Garante que está em modo de inferência
        self.env.allocation_results['dst'] = {'allocated_server': dst, 'path': [], 'cost': 0}

        # 4. Loop de predição para tomar decisões até o fim do episódio
        done = False
        while not done:
            action_masks = self.env.action_masks()
            action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

        # 5. Processa o resultado final do episódio
        if not self.env.success:
            if not VERBOSE:
                print(f"Causa Falha: {self.env.fail_reason}")
                print(f"Alocação: [{self.env.servers_used}] || Custo latência: {self.env.latency_used}")
            self.fail_reason = self.env.fail_reason
            return [], None

        print(f"SFC: {self.sfc.id}: {self.env.servers_used} || latência usada: {self.env.latency_used}")

        # Monta o route_info a partir dos resultados bem-sucedidos do ambiente
        route_info = {
            key: list(reversed(value['path']))
            for key, value in self.env.allocation_results.items()
        }

        current_location = self.env.current_location
        if (current_location, 0) not in self.precomputed_paths:
            self.precomputed_paths[(current_location, 0)] = nx.dijkstra_path(self.graph, current_location, 0, weight='weight')

        path_to_src = self.precomputed_paths[(current_location, 0)]
        route_info['src'] = list(reversed(path_to_src))

        # Calcula a latência total (excluindo os nós, contando apenas os links)
        total_latency = self.env.latency_used + (len(path_to_src) - 1)

        return route_info, total_latency


    
    def evaluate_result(self, latency, route_info):
        if self.fail_reason in ['resource', 'latency','bandwidth']:
            self.route_info = False
            self.latency = None
            return False
        self.latency = latency
        self.route_info = route_info
        return True

    def _load_or_create_model(self, env):
        if self.model_name == "ppo":
            self.model = MaskablePPO.load(self.model_path, env=env)
        elif self.model_name == "dqn":
            if os.path.exists(self.model_path + ".zip"):
                model = DQN.load(self.model_path)
                self.model = DQN("MlpPolicy", env, verbose=0, learning_rate=0.00003, batch_size=64, buffer_size=100_000_000, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=256, device='cpu')
                self.model.policy.load_state_dict(model.policy.state_dict())
            else:
                self.model = DQN("MlpPolicy", env, verbose=0, learning_rate=0.00003, batch_size=64, buffer_size=100_000_000, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=256, device='cpu')

    def load_model(self, env):
        # Carregar modelo uma vez, se não carregado
        if self.model is None:
            if self.model_name == "ppo":
                self.model = MaskablePPO.load(self.model_path, env=env)
            elif self.model_name == "dqn":
                self.model = DQN.load(self.model_path, env=env)
        self.model.set_env(env)
        
    def reset_environment(self, list_graph, list_sfc):
        # Resetando variáveis importantes do ambiente
        self.env.is_training = False
        self.env._set_list_graph_sfcs(list_graph, list_sfc)
        # self.env.reset()
        
        
        
    def _initialize_environment_and_model(self):
        # Inicializa o ambiente e o modelo no começo
        if self.env is None:
            self.env = SFC_AllocationEnv(valid_nodes=self.valid_nodes, list_graph=[self.graph], list_sfc=[self.sfc])

        if self.model is None:
            self.load_model(self.env)
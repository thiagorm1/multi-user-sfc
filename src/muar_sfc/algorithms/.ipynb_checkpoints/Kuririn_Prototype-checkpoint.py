import copy
import logging
import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
#from .enviroments_rl.environment import NetworkEnv
from muar_sfc.algorithms.environment import NetworkEnv
from muar_sfc.algorithms.networkUtils import get_shortest_path_length,get_shortest_path,pre_get_single_source_minimum_latency_path, get_link_latency

import os

from config import ROOT_PATH

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(ROOT_PATH + './logs/Kuririn.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
N_STEPS=256
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Kuririn:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = f'saved_models_rl/PPO_sfc_allocation'
        self.name = "kuririn"
        self.env = None
        self.model = None
        self.substrate_network = self.sfc = None
        self.route_info = {}
        self.graph = None
        
        self.latency = self.latency_request = None
        self.num_nodes = None
        self.servers_used = []
        self.single_source_minimum_latency_path = None
        self.fail_reason = None
        self.can_host_multiple_sfs = True
        self.is_backup = False
        self.valid_nodes = []
            
        # Cost weights
        self.cpu_factor = 1
        self.cache_factor = 1
        self.band_factor = 1
        self.latency_factor = 5

        self.boot_factor = 0

    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.graph = None
        self.valid_nodes = []
        self.route_info = {}
        self.latency = None
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.single_source_minimum_latency_path = None

    def install_substrate_network(self, substrate_network, sfc_list, shareable_sfs=[]):
        self.substrate_network = substrate_network
        self.graph = copy.deepcopy(substrate_network.graph)
        self.add_mobile_user_to_graph(substrate_network,sfc_list)
        return self.substrate_network

    def add_mobile_user_to_graph(self,substrate_network,sfc_list):
        mobile_device_id = sfc_list[0].dst_node
        closer_router    = sfc_list[0].closer_router
        
        self.valid_nodes = [25, 8, 14, 28, 2, 5, 9, 23, 18, 6, 33, 34]
        self.valid_nodes.append(mobile_device_id)

        # Os recursos do Mobile Device devem estar disponíveis somente para sua SFC
        md_info =  substrate_network.md_graph._node[mobile_device_id]
        self.graph.add_node(mobile_device_id,type='mobile_device',
                                cpu_capacity=md_info['cpu_capacity'],
                                cache_capacity=md_info['cache_capacity'],
                                cpu_used=md_info['cpu_used'],
                                cache_used=md_info['cache_used'],
                                position=md_info['position'],
                                services=md_info['services'])
        router = self.graph._node[closer_router]
        wireless_free = router['w_channel_capacity'] - router['w_channel_used']
        
        # TODO Permitir que o próprio algoritmo escolha o roteador
        # TODO calcular a latência do sinal
        signal_latency = 1
        self.graph.add_edge(mobile_device_id, closer_router, bandwidth_capacity=wireless_free, bandwidth_used=0.00 , latency=signal_latency, services_in_transit={})


    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        self.dst_substrate_node = sfc.dst_node
        return sfc

    def submit_solution(self):
        def allocate_microservice(node_id, service_id, cpu_required, cache_required):
            node = self.graph.nodes[node_id]

            # Verifica se há recursos disponíveis
            if node['cpu_used'] + cpu_required > node['cpu_capacity']:
                raise ValueError(f"CPU excedida no nó {node_id} para serviço {service_id}")
            if node['cache_used'] + cache_required > node['cpu_capacity']:
                raise ValueError(f"Cache excedido no nó {node_id} para serviço {service_id}")

            if service_id in node['services']:
                node['services'][service_id]['copys'] += 1  # Serviço já instanciado
                if not self.is_shareable(service_id):  # Se não for compartilhável
                    node['cpu_used'] += cpu_required
                    node['cache_used'] += cache_required
            else:
                node['services'][service_id] = {'cpu': cpu_required, 'cache': cache_required, 'copys': 1}
                node['cpu_used'] += cpu_required
                node['cache_used'] += cache_required

        def allocate_bandwidth(node1, node2, bw_required, ms_name):
            edge = self.graph.edges[node1, node2]

            # Verifica se há banda disponível
            if edge['bandwidth_used'] + bw_required > edge['bandwidth_capacity']:
                raise ValueError(f"Banda excedida entre os nós {node1} e {node2} para serviço {ms_name}")

            if ms_name in edge['services_in_transit']:
                edge['services_in_transit'][ms_name]['copys'] += 1
                edge['bandwidth_used'] += bw_required
            else:
                edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bw_required}
                edge['bandwidth_used'] += bw_required

        for ms_name, path in self.route_info.items():
            if ms_name in ['src', 'dst']:
                continue

            vnf = self.sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]

            cpu_req = vnf.get_cpu_request()
            cache_req = vnf.get_cache_request()
            allocate_microservice(node_allocated, ms_name, cpu_req, cache_req)

            bw_req = vnf.get_outcome_interface_bandwidth()

            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    allocate_bandwidth(u, v, bw_req, ms_name)

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def handle_failure(self):
        self.route_info = False
        self.latency = None
    
    def get_fail_reason(self):
        return self.fail_reason

    def is_shareable(self,service_name):
        # TODO Mudar para a informação de compartilháveis estar em uma variável separável.
        #if self.shareable_node:
        if True:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False
    
    def check_solution(self):
        print(f"Route Info:{self.route_info}")
        if len(list(self.route_info.keys()))!=6:
            return False
        if not isinstance(self.latency, (int, float)) or self.latency < 0 or self.latency > self.sfc.get_latency_request() or not self.route_info:
            return False
        return True
    
    def set_costs(self, costs_parameters):
        self.cpu_factor, self.cache_factor, self.band_factor = costs_parameters
    
    def start_algorithm(self):
        sfc = self.sfc
        logger.info("Start algorithm")
        self.algorithm(sfc)
        is_success = self.check_solution()
        if is_success:
            try:
                self.submit_solution()
                logger.info("Finished algorithm, success")
                 # self._save_model()
                return True  
            except:
                self.handle_failure() 
                # self._save_model()
                return False
        else:
            self.handle_failure() 
            # self._save_model()
            logger.info("End algorithm, failed")
            return False
    

    def algorithm(self, sfc):
        self.servers_used = []
        dst = sfc.get_substrate_node(sfc.get_dst_vnf())
        nodes_resource = self.set_nodes_resources()
        network_links = copy.deepcopy(self.graph._adj)
        G = self.create_network_graph(network_links)
        services, service_requirements = self.prepare_service_requirements(sfc.vnfs_dict)

        route_info, latency = self.find_best_allocation_for_sfc(
            G,service_requirements, nodes_resource,services, dst
        )
        # print(f"\nRazão Falha: {self.fail_reason}")
        return self.evaluate_result(latency, route_info)

    def create_network_graph(self, network_topology):
        G = nx.Graph()
        for node, edges in network_topology.items():
            for target, attr in edges.items():
                bw_free = attr['bandwidth_capacity']-attr['bandwidth_used']
                G.add_edge(node, target, bandwidth=bw_free, weight=1)
        return G

    def set_nodes_resources(self):
        resources = {}
        for server in list(self.graph.nodes()):
            resources[server] = {
                'cpu_capacity': self.graph.nodes[server]['cpu_capacity'],
                'cache_capacity':self.graph.nodes[server]['cache_capacity'],
                'cpu_used': self.graph.nodes[server]['cpu_used'],
                'cache_used': self.graph.nodes[server]['cache_used'],
                'cpu_free': self.graph.nodes[server]['cpu_capacity'] - self.graph.nodes[server]['cpu_used'],
                'cache_free': self.graph.nodes[server]['cache_capacity'] - self.graph.nodes[server]['cache_used'],
                'reuse': []
                #'position': net_info.nodes[server]['position'],
                #'reuse': [vnf.id for vnf in shareable_sfs.get(server, [])]
            }
        return resources

    def prepare_service_requirements(self, sfs_dict):
        service_requirements = {}
        services = [item['name'] for item in sfs_dict]
        for item in sfs_dict:
            name = item['name']
            service_requirements[name] = {
                'CPU': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw'],
                'latency': item['latency']
            }
        service_requirements['dst'] = {'CPU': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0, 'latency': 0}
        return list(reversed(services)), service_requirements

    def find_best_allocation_for_sfc(self, G,service_requirements, server_resources, services, dst):    
        if not self.env:
            self.env = NetworkEnv(G,server_resources, services ,service_requirements, 
                            self.latency_request, 
                            dst,self.valid_nodes ,
                            pesos={"cpu": self.cpu_factor, "cache": self.cache_factor, "band": self.band_factor, "latency": self.latency_factor})
        else:
            self.env.server_resources, self.env.services ,self.env.service_requirements = server_resources, services, service_requirements
            self.env.latency_request,self.env.dst_node = self.latency_request, dst   
        
        if not self.model:
            self._load_or_create_model(self.env)
        # self.model.learn(total_timesteps=2048)
        state, _ = self.env.reset()
        self.env.is_training = False
        self.env.allocation_results['dst'] = {'allocated_server': dst, 'path': [], 'cost': 0}

        done = False
        self.fail_reason = None
        while not done:
            action, _ = self.model.predict(state, deterministic = True)
            state, _, done, _, _ = self.env.step(action)

            
        if not self.env.success:
            # state, _ = self.env.reset()
            # self.model.learn(total_timesteps=2048)
            # state, _ = self.env.reset()
            # self.env.is_training = False
            # self.env.allocation_results['dst'] = {'allocated_server': dst, 'path': [], 'cost': 0}
            # done = False
            # while not done:
            #     action, _ = self.model.predict(state, deterministic = True)
            #     state, _, done, _, _ = self.env.step(action)             
            # if not self.env.success:
            self.fail_reason = self.env.fail_reason
            return [], None

        print(f"Latencia usada: {self.env.latency_used}")
        route_info = {
            key: list(reversed(value['path']))
            for key, value in self.env.allocation_results.items()
        }

        path_to_src = nx.dijkstra_path(G, int(self.env.current_location), 0, weight='weight')
        self.env.close()
        total_latency = sum(len(p) - 1 for p in route_info.values() if p)
        route_info['src'] = list(reversed(path_to_src))

        return route_info, total_latency

    def evaluate_result(self, latency, route_info):
        # if self.fail_reason == 'resource':
        #     self.route_info = False
        #     self.latency = None
        #     return False
        # if self.fail_reason == 'latency':
        #     self.route_info = False
        #     self.latency = None
        #     return False
        self.latency = latency
        self.route_info = route_info
        return True

    def _load_or_create_model(self,env):
        if self.model_name == "PPO":
            if os.path.exists(self.model_path + ".zip"):
                model = PPO.load(self.model_path)
                self.model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.00003, batch_size=64, n_steps=256, ent_coef=0.3, device=device)
                self.model.policy.load_state_dict(model.policy.state_dict())
            else:
                self.model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.00003, batch_size=64, n_steps=256, ent_coef=0.3, device=device)

    def _save_model(self):
        self.model.save(self.model_path)
 
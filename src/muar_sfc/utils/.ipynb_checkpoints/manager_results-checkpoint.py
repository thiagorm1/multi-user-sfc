import os
import numpy as np
import re
from datetime import datetime
import random
import time
from typing import Dict
from muar_sfc.core.net_v2 import Net2

def format_nodes_to_string(nodes):
    nodes_to_string = np.array2string(nodes, suppress_small=True, precision=3, separator=',')
    return re.sub('[ \n]', '', nodes_to_string)

def format_edges_to_string(edges):
    edges_to_string = ';'.join(map(str, edges))
    return re.sub('[ \n]', '', edges_to_string)

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_output_dir(args,topology):
    ec_servers = topology.get_topology_info()['ec_servers']
    edges = topology.get_topology_info()['edges']
    
    availability = args.ava
    
    number_of_fails = args.number_of_fails
    if availability == '1.0':
        number_of_fails = 0

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f') + str(random.randint(0, 10000))
    base_dir = f'results/'
    #paths = ['cache', 'cpu', 'bandwidth', 'edges_vnf', 'sf']
    paths = ['cache', 'cpu', 'bandwidth', 'sf']

    directories = {}

    for path in paths:
        dir_path = os.path.join(base_dir, f'results_{path}')
        create_directory_if_not_exists(dir_path)
        alg_path = os.path.join(dir_path, f'alg_{args.alg}_s_{args.n_sessions}_p_{args.n_players}_a_{availability}_c_{number_of_fails}')
        create_directory_if_not_exists(alg_path)
        directories[path] = alg_path

    # Prepare file paths
    file_paths = {path: os.path.join(directories[path], timestamp + '.csv') for path in paths}
    
    nodes_string = format_nodes_to_string(np.array(sorted(ec_servers)))
    edges_string = format_edges_to_string(edges)
    
    # Initialize files
    with open(file_paths['cache'], "a") as file:
        file.write(f'timestamp,{nodes_string[1:-1]}\n')
    with open(file_paths['cpu'], "a") as file:
        file.write(f'timestamp,{nodes_string[1:-1]}\n')
    with open(file_paths['sf'], "a") as file:
        file.write(f'timestamp,{nodes_string[1:-1]}\n')
    with open(file_paths['bandwidth'], "a") as file:
        file.write(f'timestamp;{edges_string}\n')
    # with open(file_paths['edges_vnf'], "a") as file:
    #     file.write(f'timestamp;{edges_string}\n')
    
    dir = f'results/results_flows'
    res_dir = f'results/results_resilient'
    directory_path = os.path.join(dir, f'{args.alg}_s_{args.n_sessions}_p_{args.n_players}_a_{availability}_c_{number_of_fails}')
    res_directory_path = os.path.join(res_dir, f'{args.alg}_s_{args.n_sessions}_p_{args.n_players}_a_{availability}_c_{number_of_fails}')
    create_directory_if_not_exists(directory_path)
    create_directory_if_not_exists(res_directory_path)

    flows_path = os.path.join(directory_path, f'{timestamp}.csv')
    res_path = os.path.join(res_directory_path, f'{timestamp}.csv')

    # Define o header como uma lista para facilitar alterações
    header_fields = [
        "No.",
        "timestamp",
        "time_seconds",
        "users",
        "cpu_utilization",
        "bandwidth_utilization",
        "cache_utilization",
        "mobile_cpu_utilization",
        "mobile_cache_utilization", 
        "latency",
        "latency_diff",
        "wait_time",
        "decision_time_ms",
        "success",
        "fail_reason",
        "sfc_id",
        "recovery_time",
        "sfc_recovered",
        "cpu_saved",
        "cache_saved",
        "shared_vnfs",
        "running_sfcs",
        "running_players",
        "running_sessions",
        "trascode_bw",
        "crashing",
        "acceptance_rate",
        "reuse_cpu_rate",
        "reuse_cache_rate",
    ]

    res_fields = ["crash_trial","sfc_id","vnf_id","recover_success","backup_success","backup_efficient","latency_diff","latency_deg","resource_deg","time_to_recover"]
            
    header = ",".join(header_fields) + "\n"
    res_header = ",".join(res_fields) + "\n"

    with open(flows_path, "a") as f:
        f.write(header)

    with open(res_path, "a") as f:
        f.write(res_header)
    return file_paths,flows_path,res_path

class OutputWritter:
    def __init__(self,topology,file_paths,flows_file,res_file):
        
        self.processing_nodes = topology.get_topology_info()['ec_servers']
        self.nodes = topology.get_topology_info()['nodes']
        self.edges = topology.get_topology_info()['edges']

        self.cpu_utilization_file = file_paths['cpu']
        self.cache_utilization_file = file_paths['cache']
        self.bw_utilization_file = file_paths['bandwidth']
        self.sf_utilization_file = file_paths['sf']
        
        self.flows_file = flows_file
        self.resilient_file = res_file
        self.first_time = 0
        self.sfcs_latency_dict = {}
        self.counter_users = 0

    def resilient_output(self,sfc_id,info,crash_trial):
        is_success = info["recover_success"]
        backup_success = info["backup_success"]
        backup_efficient = info['backup_efficient']
        latency_diff = info["latency_diff"]
        time_to_recover = info["time_to_recover"]
        vnf_id = info["vnf_id"]
        latency_deg = info["latency_degrad"]
        resource_deg = info["resource_degrad"] 
             

        with open(self.resilient_file, "a") as file:
            line = str(crash_trial) + ',' + \
                str(sfc_id) + ',' + \
                str(vnf_id) + ',' + \
                str(is_success) + ',' + \
                str(backup_success) + ',' + \
                str(backup_efficient) + ',' + \
                str(latency_diff) + ',' + \
                str(latency_deg) + ',' + \
                str(resource_deg) + ',' + \
                str(time_to_recover) + "\n"
            file.write(line)

    def output_flows(self,substrate_network: Net2,wait_time,running_players_sessions,counter,remaining_time,current_time, sfc_id, 
                     latency, run_duration, is_success,fail_reason,bw_transcode,acceptance_rate, latency_diff=None,crashing=False,alg_name='ga'):
        cpu_utilization = round(substrate_network.get_cpu_utilization_rate(), 4)
        cache_utilization = round(substrate_network.get_cache_utilization_rate(), 4)
        bw_utilization = round(substrate_network.get_bandwidth_utilization_rate(), 4)

        total_cpu_request = round(substrate_network.get_total_cpu_request(), 4)
        reuse_cpu_rate = (substrate_network.get_total_cpu_saved()/total_cpu_request)*100 if total_cpu_request > 0 else 0
        total_cache_request = round(substrate_network.get_total_cache_request(), 4)
        reuse_cache_rate = (substrate_network.get_total_cache_saved()/total_cache_request)*100 if total_cache_request > 0 else 0

        mobile_cpu_utilization = round(substrate_network.mobile_cpu_used, 4)
        mobile_cache_utilization = round(substrate_network.mobile_cache_used, 4)
        
        cpu_resilient = 0 #round(substrate_network.get_resilient_cpu_utilization(), 4)
        cache_resilient = 0 # round(substrate_network.get_resilient_cache_utilization(), 4)
        bw_resilient = 0 #round(substrate_network.get_resilient_bandwidth_utilization(), 4)

        # active_servers_cpu = round(substrate_network.get_active_servers_cpu_rate(), 4)
        # active_servers_cache = round(substrate_network.get_active_servers_cache_rate(), 4)
        # active_links_bw = round(substrate_network.get_active_links_bw_rate(), 4)

        running_sfcs, running_players, running_sessions = running_players_sessions

        cpu_saved =  substrate_network.total_cpu_saved
        cache_saved =  substrate_network.total_cache_saved
        shared_vnfs_count = substrate_network.shared_vnfs_count
        
        sfc_recovery_time = 0
        sfc_recovered = None
        
        # if sfc.id in sfcs_crashed:
        #     sfc_recovered =  0   
        #     sfc_recovery_time = None
        crashed_sfcs = []

        self.update_user_count(sfc_id)
        # if sfc.id in sfcs_crashed and is_success == 1:
        #     sfc_recovery_time = time.time() - sfcs_crashed[sfc.id]
        #     sfc_recovered = 1

        first_loop = (self.first_time == 0)
        time_value = 0  

        if first_loop: 
            self.first_time = current_time
            time_value = 0
        else:        
            time_value = round(current_time - self.first_time,1)
        decision_time =  str(round(run_duration * 1000, 3))

        # if backup_sfc:
        #     # backup_success = is_success
        #     is_success = None
        #     #if alg_name == 'ga':
        #     latency = None
        #     decision_time = None

        line = (
            f"{counter},"
            f"{current_time},"
            f"{time_value},"
            f"{self.counter_users},"
            f"{cpu_utilization},"
            f"{bw_utilization},"
            f"{cache_utilization},"
            f"{mobile_cpu_utilization},"
            f"{mobile_cache_utilization},"
            f"{latency},"
            f"{latency_diff},"
            f"{wait_time},"
            f"{decision_time},"
            f"{is_success},"
            f"{fail_reason},"
            f"{sfc_id},"
            f"{sfc_recovery_time},"
            f"{sfc_recovered},"
            f"{cpu_saved},"
            f"{cache_saved},"
            f"{shared_vnfs_count},"
            f"{running_sfcs},"
            f"{running_players},"
            f"{running_sessions},"
            f"{bw_transcode},"
            f"{crashing},"
            f"{acceptance_rate},"
            f"{reuse_cpu_rate},"
            f"{reuse_cache_rate}\n"
        )

        with open(self.flows_file, "a") as file:
            file.write(line)


    def update_user_count(self, sfc_id):
        # Extrair o número do player e da sessão
        player = int(sfc_id.split("_")[-2][1])
        session = int(sfc_id.split("_")[-1])
        
        # Verificar se a sessão atual é um backup (sessões pares são backups)
        users = self.counter_users

        users = (session - 1) * 5 + player
        
        if users > self.counter_users:
            self.counter_users = users


    def output_cpu_utilization(self, substrate_network, deploy_time, crashed_nodes=[]) -> None:
        """
        Outputs the CPU utilization of nodes to a specified file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
            crashed_nodes (list): List of nodes that are crashed and should not report CPU usage.
        """
        processing_nodes = sorted(self.processing_nodes)
        cpu_nodes_util = [
            None if node in crashed_nodes else round(substrate_network.get_node_cpu_used(node), 2)
            for node in processing_nodes
        ]

        # Format the array to a string
        string_cpu_nodes_util = ','.join(['None' if value is None else f"{value:.2f}" for value in cpu_nodes_util])

        # Write the result to the file
        with open(self.cpu_utilization_file, "a") as file:
            file.write(f"{deploy_time},{string_cpu_nodes_util}\n")

    def output_cache_utilization(self, substrate_network, deploy_time, crashed_nodes=[]) -> None:
        """
        Outputs the cache utilization of nodes to a specified file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
            crashed_nodes (list): List of nodes that are crashed and should not report cache usage.
        """
        processing_nodes = sorted(self.processing_nodes)

        cache_nodes_util = [
            None if node in crashed_nodes else round(substrate_network.get_node_cache_used(node), 2)
            for node in processing_nodes
        ]

        # Format the array to a string
        string_cache_nodes_util = ','.join(['None' if value is None else f"{value:.2f}" for value in cache_nodes_util])

        # Write the result to the file
        with open(self.cache_utilization_file, "a") as file:
            file.write(f"{deploy_time},{string_cache_nodes_util}\n")


    def output_bandwidth_utilization(self, substrate_network, deploy_time: float) -> None:
        """
        Outputs the bandwidth utilization of network edges to a specified file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
        """
        nodes_one = [node_one for node_one, node_two in self.edges]
        nodes_two = [node_two for node_one, node_two in self.edges]

        bw_edges_util = np.array(list(map(substrate_network.get_link_bandwidth_used, nodes_one, nodes_two)))

        string_bw_edges_util = np.array2string(bw_edges_util, suppress_small=True,
                                               precision=3, separator=';', 
                                               formatter={'float_kind': lambda x: "%.2f" % x})

        string_bw_edges_util = re.sub(' ', '', string_bw_edges_util)
        string_bw_edges_util = re.sub('\n', '', string_bw_edges_util)

        with open(self.bw_utilization_file, "a") as file:
            file.write(str(deploy_time) + ';' + string_bw_edges_util[1:-1] + '\n')

    def output_nodes_sf_utilization(self, substrate_network, deploy_time: float) -> None:
        """
        Outputs the service function (SF) utilization of nodes to a specified file.

        This function calculates the number of service functions running on each node in the network.
        The results are formatted as a string and appended to the SF utilization file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
            substrate_network (object): The substrate network object containing node information.

        Returns:
            None
        """
        # sf_nodes_util = np.array([len(substrate_network.get_node_sfc_vnf_list(node)) for node in self.nodes])

        # string_sf_nodes_util = np.array2string(sf_nodes_util, suppress_small=True,
        #                                        precision=3, separator=',', 
        #                                        formatter={'float_kind': lambda x: "%.2f" % x})

        # string_sf_nodes_util = re.sub(' ', '', string_sf_nodes_util)
        # string_sf_nodes_util = re.sub('\n', '', string_sf_nodes_util)

        # with open(self.sf_utilization_file, "a") as file:
        #     file.write(str(deploy_time) + ',' + string_sf_nodes_util[1:-1] + '\n')
        pass

    def output_nodes_information(self, substrate_network,*args) -> None:
        substrate_network.print_out_nodes_information(args[0], args[1])

    def output_edges_information(self,substrate_network,*args) -> None:
        substrate_network.print_out_edges_information(args[0])
    
    def output_acceptance_information(self,substrate_network,success) -> None:
        substrate_network.print_out_acceptance_information(success)

    def print_output_info(self, substrate_network,success) -> None:
        """
        Outputs information about nodes and edges' 
        resource utilization. 
        """
        self.output_nodes_information(substrate_network,None, None)
        self.output_edges_information(substrate_network,None)
        self.output_acceptance_information(substrate_network,success)
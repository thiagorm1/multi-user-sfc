import argparse
import math
import re
import os
import random
import numpy as np
import ast

from muar_sfc.controllers.modules.sfcs_manager import SFCManager
from muar_sfc.controllers.substrate_network_controller import SubstrateNetworkController
from datetime import datetime as dt
from muar_sfc.core.poisson_emitter import PoissonEmitter
from muar_sfc.controllers.sfc_queue import SFCQueue
from muar_sfc.controllers.sfc_generator import SFCGenerator

from muar_sfc.algorithms.instantiator import AlgorithmInstantiator
from muar_sfc.controllers.modules.mobility_manager import MobilityManager
from muar_sfc.controllers.modules.crasher import Crasher
from sumo.tracer_instantiator import TracerInstantiator
from muar_sfc.topology.instantiator import TopologyInstantiator
from muar_sfc.utils.manager_results import create_output_dir, OutputWritter


# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# command line arguments
parser = argparse.ArgumentParser(description='Select MUAR arguments') 
parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
parser.add_argument('--alg',   type=str, help='(str) algorithm name', default='ga')
parser.add_argument('--n_players', type=int, help='(int) number of players', default=4)
#on: quebrar mais em funçoes
#off: monolítico
parser.add_argument('--sfc',   type=str, help='(str) on or off', default='on')
parser.add_argument('--topology', type=str, help='(str) wich topology ex: luxembourg,small luxembourg ,paloalto', default='luxembourg')
parser.add_argument('--share',  type=str, help='(str) whether to share sfs or not', default='y')
parser.add_argument('--shareband',  type=str, help='(str) whether to share sfs or not', default='n')
parser.add_argument('--time',  type=int, help='(int) the total time for the simulation in seconds', default=120)
parser.add_argument('--mobility',  type=str, help='(str) mobility', default='y')

parser.add_argument('--allow_delay', type=str, help='(str) whether to allow delay or not', default='n')
parser.add_argument('--ava', type=str, help='(str) whether to allow delay or not', default='0.95')
parser.add_argument('--number_of_fails', type=str, help='(str) whether to allow delay or not', default='3')

#parser.add_argument('--costs_parameter',   type=str, help='cpu,cache,bandwidht,boot Ex: 1111', default='[1,1,1,1]')
parser.add_argument('--verbose',   type=str, help='verbose log', default='n')

#Coleta dos parâmetros da simulação
args = parser.parse_args()

alg_name = args.alg
top_name = args.topology
n_sessions = int(args.n_sessions)
n_players = int(args.n_players)
mobility_activated = (args.mobility == 'y')
verbose = (args.verbose == 'y')
crasher_activated = (float(args.ava) != 1.0)
number_of_fails = int(args.number_of_fails)
if not crasher_activated:
    number_of_fails = 0
allow_delay =  (args.allow_delay == 'y')
shareable = (args.share == 'y')
shareable_band = (args.shareband == 'y')
fail_interval = 200
#costs_parameters = ast.literal_eval(args.costs_parameter)
SELECTED_ALG = AlgorithmInstantiator().instantiate_algorithm(alg_name)
topology = TopologyInstantiator().instantiate_topology(top_name)
tracer = TracerInstantiator().instantiate_tracer(top_name) if mobility_activated else None

substrate_network = topology.generate_substrate_network()
ec_servers = topology.get_topology_info()['ec_servers']
nodes = topology.get_topology_info()['nodes']
routers = topology.get_topology_info()['routers']
nodes_num = len(nodes)
edges = topology.get_topology_info()['edges']

AVERAGE_TIME_SESSION_ARRIVAL = 20
sfc_poisson_emitter = PoissonEmitter(AVERAGE_TIME_SESSION_ARRIVAL)
sfc_queue = SFCQueue()

SRC_NODE = 0
max_duration = 120
latency = [7,7]
min_latency_acc = max(latency) if allow_delay else min(latency)

fator = 0.33 # 1 players consumes fator*100 percentage of resources of an Edge Server
#https://ieeexplore.ieee.org/document/9417376

cpb = 10e6 #10 cycles per Mbit
#print(cpb, "cycles per Mbit")
IA_bw = 150# * n_players
IA = IA_bw * cpb #
IA = 0
# 400x400 pixels RGB with about 0.48 MB per frame
DET_bw = 0.230 # 480 KB * 8 * 60 fps
DET = DET_bw * cpb
# 4 to 12 feature representations of VO, each have 25KB [100,300] KB per frame
FT_bw = 0.144 # 300 KB * 8 * 60  [48,144] 
FT = FT_bw * cpb
IA_DET_FT_bw = int(IA_bw + DET_bw + FT_bw)
# each AR VO have 2500 KB in average, 2500 * 12
CA_size = 240 # 2500 KB * 8 * 12 objects 240 for 12 VOs

chr = 1/3 # cache hit ratio
MA_bw = int(CA_size * chr)
MA = MA_bw * cpb #
# not in cache
UNI_bw = int(CA_size * (1-chr))
UNI = UNI_bw * cpb # 
# matched and non-matched objects
RE_bw = MA_bw + UNI_bw 
RE = int(RE_bw * cpb) #
#https://ieeexplore.ieee.org/document/9316983
# final out put 
EC_TC_bw = int(IA_bw*0.9*0.9*0.8) 
EC_TC = EC_TC_bw * cpb  #

total = IA+DET+FT+MA+UNI+RE+EC_TC
total_inst = round(total*(1/fator)/ 1000000)
total_inst = total * 3
servers_mips = 96_100 # MIPS SAP 96_100

IA = int(IA/total*fator*100)
DET = DET/total*fator*100
FT = FT/total*fator*100
IA_DET_FT = int(IA + DET + FT) #
MA = int(MA/total*fator*100)#
UNI = int(UNI/total*fator*100)
RE = int(RE/total*fator*100)#
EC_TC = int(EC_TC/total*fator*100)#
MONO = int(DET+FT+MA+UNI+RE+EC_TC)
CA_size = CA_size/CA_size*fator*100

def generate_mono_session(parameter):
    global n_players
    global session_counter
    session_counter = session_counter + 1
    counter = str(session_counter)
    print("Total Number of Monolithic MUAR in session: ", counter)
    dst_node = random.randint(0, len(nodes) -1 )
    while(dst_node == SRC_NODE):
        dst_node = random.randint(0, len(nodes) - 1)
    players_mono_sf_list = []
    for i in range(1,n_players+1):
        mono_sf_list = []
        mono_sf_list.append({"type": 2, "name":"IA_" + counter, 
            "CPU": IA, "cache": 0, "in_bw": IA_bw, "out_bw": IA_bw})
        mono_sf_list.append({"type": 2, "name":"MONO_p" + str(i) + "_"    + counter, 
            "CPU": MONO, "cache": CA_size, "in_bw": IA_bw, "out_bw": EC_TC_bw})
        players_mono_sf_list.append(mono_sf_list)
    lifetime = np.random.poisson(max_duration)
    #lifetime = int(round(np.random.exponential(max_duration)))
    duration = lifetime
    players_mono_dict_list = []
    for i in range(1,n_players+1):
        player_mono_dict = {}
        player_mono_dict['name'] = 'sfc_mono_p' + str(i) + '_' + counter
        player_mono_dict["vnf_list"] = players_mono_sf_list[i-1]
        player_mono_dict["bandwidth"] = EC_TC_bw
        player_mono_dict["src_node"] = SRC_NODE
        player_mono_dict["dst_node"] = dst_node
        player_mono_dict["duration"] = duration
        player_mono_dict["latency"] = min_latency_acc
        players_mono_dict_list.append(player_mono_dict)
    players_sfc_list = []
    for i in range(1,n_players+1):
        players_sfc_list.append([SFCGenerator(players_mono_dict_list[i-1]).generate()])
        #heapq.heappush(sfc_queue, (1, players_sfc_list[i-1]))
        sfc_queue.put_sfc(players_sfc_list[i-1])
    if session_counter >= n_sessions:
        print("MONO MUAR Session ## poisson stop  ##")
        sfc_poisson_emitter.stop()

session_counter = 0
def generate_sfc_session(parameter) -> None:
    global n_players
    global session_counter
    session_counter = session_counter + 1
    counter = str(session_counter)
    print("Total Number of MUAR SFCs in session: ", counter)
    dst_node = random.choice(routers)

    players_cache_sf_list = []
    players_unique_sf_list = []
    for i in range(1,n_players+1):
        caching_sf_list = []
        caching_sf_list.append({"type": 2, "name":"IA_DET_FT_" + counter, 
            "CPU": IA_DET_FT, "cache": 0, "in_bw": IA_bw, "out_bw": IA_DET_FT_bw,"latency":round((IA_DET_FT*total_inst/servers_mips)*10,2)})
        caching_sf_list.append({"type": 2, "name":"MA_region_" + str(dst_node),# + "_" + counter, 
            "CPU": MA, "cache": CA_size, "in_bw": IA_DET_FT_bw, "out_bw": MA_bw,"latency":round((MA*total_inst/servers_mips)*10,2)})
        caching_sf_list.append({"type": 2, "name":"RE_region_" + str(dst_node),# + "_" + counter, 
            "CPU": RE*chr, "cache": 0, "in_bw": MA_bw, "out_bw": RE_bw*chr,"latency":round((RE*chr*total_inst/servers_mips)*10,2)})
        caching_sf_list.append({"type": 2, "name":"EC_TC_p" + str(i) + "_" + counter, 
            "CPU": EC_TC*chr, "cache": 0, "in_bw": RE_bw*chr, "out_bw": EC_TC_bw*chr,"latency":round((EC_TC*chr*total_inst/servers_mips)*10,2)})
        players_cache_sf_list.append(caching_sf_list)
        unique_sf_list = []
        unique_sf_list.append({"type": 2, "name":"IA_DET_FT_" + counter, 
            "CPU": IA_DET_FT, "cache": 0, "in_bw": IA_bw, "out_bw": IA_DET_FT_bw,"latency":round((IA_DET_FT*total_inst/servers_mips)*10,2)})
        unique_sf_list.append({"type": 2, "name":"UNI_p" + str(i) + "_" + counter, 
            "CPU": UNI, "cache": 0, "in_bw": IA_DET_FT_bw, "out_bw": UNI_bw,"latency":round((UNI*total_inst/servers_mips)*10,2)})
        unique_sf_list.append({"type": 2, "name":"RE_p" + str(i) + "_"    + counter, 
            "CPU": RE*(1-chr), "cache": 0, "in_bw": UNI_bw, "out_bw": RE_bw*(1-chr),"latency":round((RE*(1-chr)*total_inst/servers_mips)*10,2)})
        unique_sf_list.append({"type": 2, "name":"EC_TC_p" + str(i) + "_" + counter, 
            "CPU": EC_TC*(1-chr), "cache": 0, "in_bw": RE_bw*(1-chr), "out_bw": EC_TC_bw*(1-chr),"latency":round((EC_TC*(1-chr)*total_inst/servers_mips)*10,2)})
        players_unique_sf_list.append(unique_sf_list)
    lifetime = np.random.poisson(max_duration)
    #lifetime = int(round(np.random.exponential(max_duration)))
    duration = lifetime
    players_sfc_cache_dict_list = []
    players_sfc_unique_dict_list = []
    for i in range(1,n_players+1):
        player_cache_dict = {}
        player_cache_dict['name'] = 'sfc_cache_p' + str(i) + '_' + counter
        player_cache_dict["vnf_list"] = players_cache_sf_list[i-1]
        player_cache_dict["bandwidth"] = EC_TC_bw
        player_cache_dict["src_node"] = SRC_NODE
        player_cache_dict["dst_node"] = dst_node
        player_cache_dict["duration"] = duration
        player_cache_dict["latency"] = min_latency_acc
        players_sfc_cache_dict_list.append(player_cache_dict)
        player_unique_dict = {}
        player_unique_dict['name'] = 'sfc_unique_p' + str(i) + '_' + counter
        player_unique_dict["vnf_list"] = players_unique_sf_list[i-1]
        player_unique_dict["bandwidth"] = EC_TC_bw
        player_unique_dict["src_node"] = SRC_NODE
        player_unique_dict["dst_node"] = dst_node
        player_unique_dict["duration"] = duration
        player_unique_dict["latency"] = min_latency_acc
        players_sfc_unique_dict_list.append(player_unique_dict)
    
    players_sfc_list = []
    for i in range(1,n_players+1):
        players_sfc_list.append([SFCGenerator(players_sfc_cache_dict_list[i-1]).generate(), SFCGenerator(players_sfc_unique_dict_list[i-1]).generate()])
        sfc_queue.put_sfc(players_sfc_list[i-1])
        #heapq.heappush(sfc_queue, (1, counter, i, players_sfc_list[i-1]))
    if session_counter >= n_sessions:
        #print("SFC MUAR Session ## poisson stop  ##")
        sfc_poisson_emitter.stop()

if args.sfc == 'off':
    print('sfc off')
    sfc_poisson_emitter.start(generate_mono_session, (None))
if args.sfc == 'on':
    print('sfc on')
    sfc_poisson_emitter.start(generate_sfc_session, (None))

substrate_network.set_verbose(verbose=verbose)
timestamp,file_paths,flows_path,res_path = create_output_dir(args, topology)

# substrate_network.shareable_band = shareable_band
# substrate_network.shareable_node = shareable

sbn_controller = SubstrateNetworkController()
sbn_controller.substrate_network = substrate_network
sbn_controller.sfc_queue = sfc_queue
sbn_controller.sfc = args.sfc
# sbn_controller.shareable = shareable
sbn_controller.alg = SELECTED_ALG

sbn_controller.crasher_activate = crasher_activated #crasher_activated

sbn_controller.crasher_manager = Crasher(edge_servers=ec_servers,availability=float(args.ava),fail_interval=fail_interval,number_of_fails=number_of_fails,edges_vnf={key: [] for key in edges})
sbn_controller.mobility_manager = MobilityManager(tracer)
sbn_controller.sfc_manager = SFCManager()

sbn_controller.mobility_activated = mobility_activated

sbn_controller.verbose = verbose
# sbn_controller.nodes = nodes
# sbn_controller.ec_servers = ec_servers
# sbn_controller.routers = routers
# sbn_controller.edges = edges

sbn_controller.allow_high_latency = allow_delay
sbn_controller.latency_interval = latency

sbn_controller.output_writter = OutputWritter(nodes,
                                                ec_servers,
                                                edges,
                                                file_paths['cpu'],
                                                file_paths['cache'],
                                                file_paths['bandwidth'],
                                                file_paths['sf'],
                                                flows_path,res_path)
# sbn_controller.shareable_band = False
sbn_controller.flows = n_sessions
sbn_controller.players = n_players

sbn_controller.start()

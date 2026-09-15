
import sys
import time
import re
import copy
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.utils.k_shortest_paths import k_shortest_paths


class SFCManager:
    def __init__(self):
        self.players = 4
        self.flows = 50
        self.sfc_list = [] # sfcs that are running in the simulation
        self.sfcs_routing_info = {}
        self.sfc_id_duration = {}

        self.player_sfc_id_list = {}
        self.sfcs_deployed_history = []
        #self.sfcs_that_crashed = []
        self.sfs_backup = {}
        self.last_sfc_release = False
        self.risk_servers = []
        self.counter = 0 # how many sfcs are running
        self.verbose = True
        self.crashed_servers = []

    def deploy_sfc(self, sfc: object,substrate_network,alg) -> bool:
        if sfc.id in self.sfc_list:
            return False
        
        if sfc.id in list(self.sfs_backup.keys()):
            self.take_off_backup(sfc.id,substrate_network)
        
        alg = copy.deepcopy(alg)
        alg.clear_all()
        shareable_sfs = substrate_network.get_shareable_sfs()
        if alg.name != 'ga':
            alg.install_substrate_network(substrate_network)
            alg.install_SFC(sfc)
        else:
            alg.install_SFC(sfc)
            alg.install_substrate_network(substrate_network,shareable_sfs=shareable_sfs,crashed_servers=self.crashed_servers)
            
        s = time.time() # Start measuring how long it takes to the alg run

        
        is_backup = self.is_Backup(sfc.id)

        match alg.name:
            case 'ga' | 'osfem' | 'goku': # algs with active reuse and cost method
                #alg.set_costs([1,1,1,1])
                alg.start_algorithm(shareable_sfs=shareable_sfs)
            case 'vegeta':
                alg.start_algorithm(shareable_sfs=shareable_sfs,backup=is_backup) # alg that uses backup
            case _: # algs with passive reuse and no cost method
                alg.start_algorithm() 

        s2 = time.time()

        route_info = alg.get_route_info() # Routes choosen by the alg
        
        latency = alg.get_latency() # latency of the solution
        fail_reason = None #alg.get_fail_reason()
        # bit_rate_adjust =  1.0 if self.alg.name != 'osfem' else alg.get_bit_rate_used()
        # bw_transcode =  sfc.vnfs_dict[-1]['out_bw'] if self.alg.name != 'osfem' else alg.get_transcode_bw()

        if (alg.name == 'musfico') and (sfc.id in self.sfcs_routing_info.keys()): # musfico exclusive methodology
            latency,route_info = self.musfico_method(sfc,substrate_network)

        #is_acceptable,wait_time  = self.check_latency_and_bitrate(sfc, route_info,latency)
        #route_info, latency = self.validate_solution(route_info, latency, sfc, self.sfc)
        if latency is not None: # se latencia não é nula e se atende aos requisitos da sfc
            if isinstance(latency, (int, float)):
                if latency > sfc.get_latency_request() or latency < 0:
                    route_info = False
                    latency = None
                else: # everything is okay with the latency request
                    pass
        else: # Latency is None 
            route_info = False
            latency = None
        route_info = self.check_route_info(route_info)

        is_success = False
        current_time = s2
        run_duration = s2 - s
        arrival_time = sfc.arrival_time
        sfc.depart_time = s2
    
        if route_info:
            substrate_network.deploy_sfc(sfc, route_info)
            if is_backup: # Se for uma SFC de Backup apenas coloque na lista de sfcs com backup
                original_sfc = sfc.vnfs_dict[1]['original_sfc']              
                vnfs_dict = sfc.vnfs_dict[1]
                vnf_id = vnfs_dict['name']
                
                if original_sfc not in self.sfs_backup: # Se ela nunca teve backup, então inicializa
                    self.sfs_backup[original_sfc] = []  # Inicializa uma lista para backups

                # Adiciona um backup à lista de backups do original
                self.sfs_backup[original_sfc].append({
                    "sfc_backup_id": sfc.id,
                    "vnf_id": vnf_id,
                    "route_info": route_info
                })
            else: 
                self.sfc_list.append(sfc.id)
                self.sfc_id_duration[sfc.id] = {"duration":sfc.duration,"timer":time.time()}
                if sfc.id in list(self.sfcs_routing_info.keys()):
                    del self.sfcs_routing_info[sfc.id]
                # Adiciona a SFC com o novo route_info -> Importante para o musfico
                self.sfcs_routing_info[sfc.id] = copy.deepcopy(route_info)

            is_success = True
            self.deploy_success(sfc)
        else:
            #self.deploy_failure = 1
            self.deploy_failed(sfc)
            
        substrate_network.update()
        self.counter += 1 # at this time all verifications are done. So we add 1 to counter of sfc
        is_success,fail_reason = self.check_resources_exceed(is_success,sfc.id,substrate_network,fail_reason) # Check if any fees exceed %
        results_dict = {"current_time":current_time,"latency":latency,"run_duration":run_duration,"is_success":is_success,"fail_reason":fail_reason,"backup_sfc":is_backup}
        return results_dict
    

    def check_route_info(self,route_info):
        if route_info != False:
            nodes_crashed = self.crashed_servers
            for key,info in route_info.items():
                if key not in ['src','dst']:
                    server_used = info[0]
                    if server_used in nodes_crashed:
                        return False  
        return route_info

    def undeploy_sfc(self, sfc_id: str,substrate_network) -> None:
        # if sfc_id not in self.sfc_list:
        #     print(sfc_id, "not on the substrate network")
        #     return -1
        # try:

            # except:
            #     pass
        #if self.shareable:
        #    self.check_sf_connections()
        #try:
        substrate_network.undeploy_sfc(sfc_id)
        if not self.is_Backup(sfc_id):
            self.sfc_list.remove(sfc_id)
            del self.sfc_id_duration[sfc_id]
            #del self.sfcs_routing_info[sfc_id]

            # Se a SFC possui backup, retire ela da lista de sfcs com backup e faça undeploy do backup 
            if sfc_id in list(self.sfs_backup.keys()): 
                self.take_off_backup(sfc_id,substrate_network)
        else:
            split = sfc_id.split("_")
            original_sfc_id = split[0] + "_" + split[1] + '_' + split[3] + "_" +split[4]
            if sfc_id in list(self.sfs_backup.keys()): 
                self.take_off_backup(sfc_id,substrate_network)

        # TODO fazer a lógica de remover sfs
        # quando o tempo acaba
        substrate_network.update()
        # except ValueError:
        #     print(f"Error: SFC ID {sfc_id} not found in the list when trying to remove.")
        #     return -1


    def take_off_backup(self,sfc_id,substrate_network):
        # try:
        backups_removed = self.sfs_backup.pop(sfc_id)
        for backup in backups_removed:
            backup_id = backup["sfc_backup_id"]
            substrate_network.undeploy_sfc(backup_id)
        # except:
        #     pass

    def is_Backup(self,sfc_id):
        return True if sfc_id.split("_")[2] == 'backup' else False

    def trigger_sfc_backup(self,sfc,vnf_id, substrate_network, node_id):
        if sfc.id in list(self.sfs_backup.keys()):
            # Ativa o backup especificado
            #vnfs_dict = backup_sfc.vnfs_dict[1]

            substrate_network.reset_vnf_cpu_request(node_id, sfc.id, vnf_id)

            # Desabilita os outros backups
            # backups = self.sfs_backup[sfc.id]
            # for backup in backups:
            #     if backup["sfc_backup_id"] != backup_sfc.id:
            #         backup_id = backup["sfc_backup_id"]
            #         substrate_network.undeploy_sfc(backup_id)
            #         print(f"Backup {backup_id} desinstanciado.")

    def check_sfc_duration(self):
        remove_list = []
        current_time = time.time()  # Obtém o tempo atual uma única vez para evitar múltiplas chamadas
        for sfc_id, info in list(self.sfc_id_duration.items()):  # `info` é o dicionário com 'duration' e 'timer'
            elapsed_time = current_time - info["timer"]  # Calcula o tempo decorrido
            
            if elapsed_time >= info["duration"]:  # Verifica se a duração foi ultrapassada
                remove_list.append(sfc_id)
        return remove_list

    def get_list_sfc_duration(self, threshold=20):
        sfc_list_duration = []
        for sfc_id, info in self.sfc_id_duration.items():  # `info` contém as informações de cada SFC
            if info["duration"] <= threshold:
                sfc_list_duration.append(sfc_id)  # Adiciona o ID à lista
        return sfc_list_duration

    def check_resources_exceed(self,is_success,sfc_id,substrate_network,fail_reason):
        if is_success:
            for node in substrate_network.nodes():
                limit = 100
                cpu_used = substrate_network.get_node_cpu_used(node)
                cache_used = substrate_network.get_node_cache_used(node)
                
                
                if cpu_used > limit or cache_used > limit:
                    self.undeploy_sfc(sfc_id,substrate_network)
                    is_success = False
                    fail_reason = "exceed" 
                    return is_success,fail_reason
                
            for edge in substrate_network.edges():
                bandwidth_used = substrate_network.get_link_bandwidth_used(edge[0], edge[1])
                bandwidth_capacity = substrate_network.get_link_bandwidth_capacity(edge[0], edge[1])
                if bandwidth_used > bandwidth_capacity:
                    is_success = False
                    fail_reason = "band_exceed"
                    self.undeploy_sfc(sfc_id,substrate_network)
                    return is_success,fail_reason
            return is_success,fail_reason
        else:
            return False,fail_reason
    
    def set_risk_sfcs(self,servers,network):
        if len(servers) == 0:
            return []
        #self.risk_servers = servers
        #server_infos = []
        #sfcs_id_backup_made = []        
        backups_mount = []
        sfc_id_duration = copy.deepcopy(self.sfc_id_duration)
        for server in servers:
            server_info = network.get_node_sfc_vnf_list(server)
            sfcs_l = []
            server_infos = []
            current_time = time.time()
            if server_info != []:
                for info in server_info:
                    server_infos.append(info)
                    # TODO acho que dava pra ter mais regras se cri ou não o backup (tempo que ainda resta para aquela SF)
                    sfc_id = info[0]
                    vnf = info[1]
                    vnf_id = vnf.id

                    if sfc_id.split("_")[2]=='backup':
                        continue

                    if sfc_id not in network.sfc_dict or sfc_id not in sfc_id_duration:
                        continue 

                    # sfcs_with_backup = list(self.sfs_backup.keys())
                    # if sfc_id in sfcs_with_backup or sfc_id in sfcs_id_backup_made: # TODO isso deve ser revisto posteriormente (somente um backup por SFC)
                    #     continue
                    
                    split = sfc_id.split("_")
                    name = split[0] + "_" + split[1] + '_backup_' + split[2] + "_" +split[3]
                    
                    try:
                        network.get_sfc_by_id(name)
                        continue
                    except:
                        pass

                    sfc = network.get_sfc_by_id(sfc_id)
                    vnf_info = sfc.vnfs_dict
                    resources_info = [info for info in vnf_info if info['name'] == vnf_id][0]
                    sfc_rf = network.sfc_route_info[sfc_id]
                    
                    src = None
                    location = None
                    dst = None
                    
                    #src_in = 0 
                    src_out = resources_info['in_bw']
                    
                    dst_in = resources_info['out_bw']
                    #dst_out = 0

                    cpu = resources_info['CPU']
                    cache = resources_info['cache']

                    i = 0
                    latency_dismiss = 0
                    for key,value in sfc_rf.items():        
                        if i == 1:
                            src = value[0]
                            if src == 0: # SF IA 
                                src = value[-1]
                            else:
                                src = value[0]
                                latency_dismiss = latency_dismiss + len(value)-1
                                print()
                            break
                        if key == vnf_id:
                            location = value[0]
                            dst = value[-1]
                            latency_dismiss = latency_dismiss + len(value)-1
                            i = i + 1
                    
                    new_sfc_list = []

                    backup_sf_list = []
                    split = sfc_id.split("_")
                    name = split[0] + "_" + split[1] + '_backup_'+vnf_id+ '_' + split[2] + "_" +split[3]
                    src_name = "src_" + name
                    dst_name = "dst_" + name
                    backup_sf_list.append({"type": 2, "name":src_name,"CPU": 0, "cache": 0, "in_bw": 0, "out_bw":src_out ,"latency":0,"location":src})
                    backup_sf_list.append({"type": 2, "name":vnf_id,"CPU": cpu, "cache": cache, "in_bw": src_out, "out_bw": dst_in,"latency":0,"restriction":location,"original_sfc":sfc_id})
                    backup_sf_list.append({"type": 2, "name":dst_name,"CPU": 0, "cache": 0, "in_bw": dst_in, "out_bw":0 ,"latency":0,"location":dst})

                    new_sfc_dict = {}
                    new_sfc_dict["name"] = name
                    new_sfc_dict["vnf_list"] = backup_sf_list
                    new_sfc_dict["bandwidth"] = sfc.input_throughput
                    new_sfc_dict["src_node"] = src
                    new_sfc_dict["dst_node"] = dst

                    duration = sfc_id_duration[sfc_id]["duration"]-(current_time-sfc_id_duration[sfc_id]["timer"])
                    new_sfc_dict["duration"] = duration  + 20 
                    new_sfc_dict["latency"] =  sfc.latency_request - self.calculate_latency(sfc_rf) + latency_dismiss  # TODO deve ter aqui  um cálculo para não passar da latencia da original se implementada
                    # new_sfc_dict["original_sfc"] = sfc_rf
                    # new_sfc_dict["restrictions"] = [location]
                    new_sfc = SFCGenerator(new_sfc_dict).generate()
                    
                    #self.sfs_backup[sfc_id] = {"sfc_backup_id":new_sfc.id,"vnf_id":vnf_id}  # Por enquanto teremos apenas uma SFC de Backup por SFC 
                
                    new_sfc_list.append(new_sfc)
                    backups_mount.append(new_sfc_list)
                    #sfcs_id_backup_made.append(sfc_id)
        return backups_mount
























    # def check_sfc_duration(self):
    #     remove_list = []
    #     for sfc_id, duration in list(self.sfc_id_duration.items()):
    #         if duration <= 1:                    
    #             remove_list.append(sfc_id)
    #             # if duration is over, sfc routing info no longer needed
    #             del self.sfcs_routing_info[sfc_id]
    #             continue
    #         self.sfc_id_duration[sfc_id] = duration - 1
    #     return remove_list


    def get_shareable_sfs(self, sfc):
        """_summary_

        Args:
            route_info (Dict): A provided route info.

        Returns:
            List[Tuple]: A list of SFs candidates.
        """
        shareable_sfs_ar = []
        src_sf = sfc.get_src_vnf()
        dst_sf = sfc.get_dst_vnf()
        sf = src_sf
        while sf != dst_sf:
            for (node, candidate_sf_id, candidate_sf) in self.shareable_list:
                    if candidate_sf_id == sf.id:
                        # if max number of connections reached, cant be used.
                        print("Chegando onde não devia")
                        if len(self.substrate_network.sf_linkeds[candidate_sf]) != self.max_connections:
                            print("shareable sf found.........")
                            shareable_sfs_ar.append((node, candidate_sf_id, candidate_sf))
            sf = sf.get_next_vnf()
        return shareable_sfs_ar

    def calculate_latency(self, route_info):
        total_latency = sum(
            len(path) - 1 for key, path in route_info.items() if path and key not in ["src", "dst"]
        )
        return total_latency


    def deploy_success(self, sfc: object) -> None:
        """Print success message."""
        if self.verbose == True:
            print("deploy succeed, sfc: ", sfc.id)

    def deploy_failed(self, sfc: object) -> None:
        """Print failure message."""
        if self.verbose == True:
            print(" deploy FAILED, sfc: ", sfc.id)


    def get_running_players_sessions(self):
        running_sfcs = self.sfc_id_duration
        players = {}
        sessions = set()

        for key in list(running_sfcs.keys()):
            player = key.split('_')[-2][-1]
            session = key.split('_')[-1]
            sessions.add(session)
            if player not in players:
                players[player] = set()
            players[player].add(session)

        num_players = sum(len(sessions) for sessions in players.values())
        num_sessions = len(sessions)
        return len(running_sfcs.keys()), num_players, num_sessions



    def musfico_method(self, sfc,substrate_network):
        """
        Calculates the latency and obtains the route information for the musfico algorithm.

        Args:
            sfc (object): The service function chain (SFC) object containing the SF details.

        Returns:
            tuple: A tuple containing the latency and route_info.
        """
        route_info = {}
        # get stored route info
        route_info = copy.deepcopy(self.sfcs_routing_info[sfc.id])
        # gets dst vnf
        dst_vnf = sfc.get_dst_vnf()
        previous_vnf = sfc.get_previous_vnf(dst_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)
        prev_vnf_node = route_info[previous_vnf.id][0]
        # applies k shortest to link the last vnf with the previous one
        shortest_path = k_shortest_paths(substrate_network, prev_vnf_node, 
                                        dst_substrate_node, k=1, weight='latency')
        # get the shortest among the k shortest paths
        route_info[previous_vnf.id] = shortest_path[0]
        latency = 0
        for vnf_id in route_info.keys():
            if vnf_id == 'src':
                continue
            path = route_info[vnf_id]
            for i in range(len(path) - 1):
                edge_latency = substrate_network.get_link_latency(
                    path[i], path[i + 1])
                latency += edge_latency

        if latency > sfc.get_latency_request() or latency < 0:
            route_info = False
            latency = None

        return latency, route_info

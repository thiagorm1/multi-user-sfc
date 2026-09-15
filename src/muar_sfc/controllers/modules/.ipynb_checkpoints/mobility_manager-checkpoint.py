# MobilityManager.py
import time
import random
import threading
from typing import Optional

from sumo.luxembourg.luxembourg_trace import Sumo_Luxembourg

class MobilityManager:
    def __init__(self,tracer=None):
        """
        Inicializa o MobilityManager com uma instância de tracer.
        
        Args:
            tracer: Uma instância de um tracer (como Sumo_Luxembourg).
        """
        self.tracer : Optional[Sumo_Luxembourg] = tracer
        self.vehicle_to_service_map = {}
        self.to_remove_vehicles = []
        self.running_sfcs = []
        self.lock = threading.Lock()
        #self.service_to_vehicle_map = {}  # Mapeia serviços para veículos

    def start_simulation(self):
        """Inicia a simulação."""
        self.tracer.start_simulation()

    def add_vehicle(self, sfc_list):
        #try:
        with self.lock:
            for sfc in sfc_list:
                player = sfc.id.split("_")[-2][1]
                p_session = sfc.id.split("_")[-1]
                player_id = int(player + p_session)
                vehicle_id = f"veh_{player_id}"
                
                if vehicle_id in list(self.vehicle_to_service_map.keys()):
                    if self.vehicle_to_service_map[vehicle_id]['connected'] == False:
                        return
                    # Verificando se o SFC já está associado ao veículo
                    if sfc.id in self.vehicle_to_service_map[vehicle_id]['sfcs']:
                        self.vehicle_to_service_map[vehicle_id]['status'] = 'available' # Redeploy bem sucedido
                        #pass  # As associações são feitas no deploy e, como pode ter vindo de um redeploy, não tem erro claro.
                    else:
                        # Adiciona o SFC à lista de SFCS associada ao veículo
                        self.vehicle_to_service_map[vehicle_id]['sfcs'].append(sfc.id)
                        self.running_sfcs.append(sfc.id)
                else:
                    # Caso o veículo não exista no mapeamento, cria o veículo e associa o SFC
                    server_start = sfc.dst_node
                    self.tracer.create_vehicle(vehicle_id, server_start)
                    
                    self.running_sfcs.append(sfc.id)
                    # Adiciona o veículo com o SFC na lista de serviços
                    self.vehicle_to_service_map[vehicle_id] = {
                        'sfcs': [sfc.id],         # Lista com o ID do SFC associado
                        'veh_location': sfc.dst_node,   # A localização inicial do veículo
                        'distance': self.tracer.get_server_distance_from_car(vehicle_id,server_start),
                        'time': time.time(),
                        'connected': True,
                        'status':'available' # active: pode alterar de servidor; 
                    }
        # except Exception as e:
        #     print(f"Erro ao adicionar veículo para o SFC {sfc.id}: {str(e)}")

    def check_all_vehicles_position_changes(self):
        """
        Verifica se algum veículo mudou de posição e retorna os IDs das SFCs associadas a eles.

        Retorna:
            dict: Dicionário onde as chaves são os IDs dos veículos que se moveram e os valores são listas
                de SFCs associadas a cada veículo.
        """
        moved_sfcs = []
        new_locations = []
        with self.lock:
            for vehicle_id in list(self.tracer.vehicles_info.keys()):  # Fazendo uma cópia dos itens
                try:  # Criar log desses casos
                    vehicle_info = self.vehicle_to_service_map[vehicle_id]
                    if vehicle_info['connected'] and vehicle_info['status'] == 'available':
                        current_position, current_distance = self.tracer.get_closest_server(vehicle_id)

                        # Posição registrada do veículo no mapeamento
                        previous_position = int(self.vehicle_to_service_map[vehicle_id]['veh_location'])
                        previous_distance = self.tracer.get_server_distance_from_car(vehicle_id, previous_position)

                        # Verificar se a posição mudou e a nova distância é pelo menos 35% menor
                        if int(current_position) != previous_position:
                            # Calcular a redução percentual da distância
                            reduction_percentage = ((previous_distance - current_distance) / previous_distance) * 100

                            # Verificar se a redução é de pelo menos 35%
                            if reduction_percentage >= 60:
                                # Atualiza a posição e a distância do veículo no mapeamento
                                self.vehicle_to_service_map[vehicle_id]['veh_location'] = int(current_position)
                                self.vehicle_to_service_map[vehicle_id]['status'] = 'moving'
                                
                                # Coleta os SFCs associados ao veículo
                                sfc_ids = self.vehicle_to_service_map[vehicle_id]['sfcs']

                                # Adiciona o veículo e os SFCs à lista de veículos que se moveram
                                moved_sfcs.append(sfc_ids)
                                new_locations.append(current_position)
                        
                        self.vehicle_to_service_map[vehicle_id]['time'] = time.time()
                except Exception as e:
                    # Caso haja exceção, remove o veículo e pode-se adicionar um log aqui
                    self.remove_vehicle(vehicle_id)
                    print(f"Erro ao verificar veículo {vehicle_id}: {e}")  # Log do erro

        return moved_sfcs, new_locations


    def remove_sfc(self, sfc_id,all=False):
        """
        Remove um SFC de todos os veículos que estão associados a ele.

        Args:
            sfc_id: ID do SFC que deve ser removido.
        """
        with self.lock:
            running_sfcs = self.running_sfcs.copy()
            if sfc_id in running_sfcs:
                for vehicle_id, vehicle_info in list(self.vehicle_to_service_map.items()):
                    # Verifica se o veículo está associado ao SFC
                    if all == False:
                        if sfc_id in vehicle_info['sfcs']:
                            # Remove o SFC da lista associada ao veículo
                            self.vehicle_to_service_map[vehicle_id]['sfcs'].remove(sfc_id)
                            self.running_sfcs.remove(sfc_id)

                            # Se o veículo não tiver mais SFCs associados, pode ser removido do mapeamento
                            if not self.vehicle_to_service_map[vehicle_id]['sfcs']:
                                self.remove_vehicle(vehicle_id)
                                #self.running_sfcs.remove(sfc_id)
                    else:
                        #running_sfcs.remove(sfc_id) 
                        sfcs_in_veh = vehicle_info['sfcs']
                        if sfc_id in sfcs_in_veh :
                            self.running_sfcs.remove(sfc_id)
                            self.remove_vehicle(vehicle_id)  
    
    def set_vehicle_status(self,sfc_id,new_status='moving'):
        running_sfcs = self.running_sfcs.copy()
        if sfc_id in running_sfcs:
             for vehicle_id, vehicle_info in list(self.vehicle_to_service_map.items()):
                sfcs_in_veh = vehicle_info['sfcs']
                if sfc_id in sfcs_in_veh :
                    self.vehicle_to_service_map[vehicle_id]['status'] = new_status
                    
    # def set_crashed_servers(self,crashed_servers):
    #     self.crashed_servers = crashed_servers
    #     self.tracer.crashed_servers = crashed_servers
    #     self.tracer.build_kdtree()

    def remove_vehicle(self, vehicle_id):
        """
        Remove um veículo da simulação e dissocia os serviços relacionados.

        Args:
            vehicle_id: Identificação do veículo.
        """
        self.vehicle_to_service_map[vehicle_id] = {'sfcs':[],'connected':False}
        self.tracer.disconnect_vehicle(vehicle_id)

    def stop_simulation(self):
        """Encerra a simulação."""
        self.tracer.stop_simulation()

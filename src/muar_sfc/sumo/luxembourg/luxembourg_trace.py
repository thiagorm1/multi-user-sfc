import datetime
import logging
import math
import random
import sys
import threading
import time

import numpy as np
import pytz
import traci
import traci.step

from muar_sfc.sumo.base.tracer_base import AbstractTracer
from muar_sfc.sumo.luxembourg.config_routes import (
    positions,
    routers,
    server_ids,
    server_tree,
    topology,
)

logger = logging.getLogger(__name__)  # <-- NOVO
# Tracer utilizando SUMO.
# Autor: Rodrigo Flexa
# Data: 17/09/2023

# Tracer utilizando SUMO.
# A simulação é controlada, por meio da biblioteca traci e das funções da classe
# Autor: Rodrigo Flexa
# Data: 17/09/2023


class Sumo_Luxembourg(AbstractTracer):
    def __init__(self, config_file="sumo//luxembourg//luxembourg.sumocfg"):
        super().__init__(config_file)
        self.counter = 0
        self.vehicles_info = {}
        self.topology = topology
        self.positions = positions
        self.lock = threading.Lock()

    def get_datetime(self):
        utc_now = pytz.utc.localize(datetime.datetime.utcnow())
        currentDT = utc_now.astimezone(pytz.timezone("America/Belem"))
        return currentDT.strftime("%Y-%m-%d %H:%M:%S")

    def calculate_kmph(self, m_per_s):
        return round(m_per_s * 3.6, 2)

    def connect_to_sumo(self):
        traci.start(["sumo", "-c", self.config_file])
        self.traci_connected = True
        self.simulation_running = True

    def stop_simulation(self):
        with self.lock:
            if self.traci_connected:
                traci.close()
                self.traci_connected = False
            self.simulation_running = False
        # Fecha o arquivo CSV
        # self.csv_file.close()

    def get_distance(self, server_i, server_j):
        """
        Calculate the Euclidean distance between two edges' coordinates.
        This assumes that self.network_graph stores coordinates for edges.
        """
        coord1 = self.topology[server_i]  # Coordinates of edge1
        coord2 = self.topology[server_j]  # Coordinates of edge2
        return np.linalg.norm(np.array(coord1) - np.array(coord2))  # Euclidean distance

    def check_route(self, server_start, server_end):
        distance = self.get_distance(server_start, server_end)
        # Loop until we find a server_end that is different from the server_start
        while distance < 800:
            # Choose a random server_end, but ensure it's sufficiently distant
            server_end = random.choice(list(self.positions.keys()))
            distance = self.get_distance(server_start, server_end)

        return server_start, server_end

    # Função para encontrar o servidor mais próximo
    def get_closest_server(self, vehicle_id):
        # Posição do veículo
        x, y = traci.vehicle.getPosition(vehicle_id)
        # Encontra o servidor mais próximo entre os disponíveis
        distance, index = server_tree.query([x, y])

        # Recupera o ID do servidor mais próximo
        closest_server_id = server_ids[index]
        return closest_server_id, distance

    def get_server_distance_from_car(self, vehicle_id, server):
        # Posição do veículo (x, y)
        x, y = traci.vehicle.getPosition(vehicle_id)

        # Coordenadas do servidor
        server_coords = self.topology[server]  # Supondo que isso seja uma tupla (x, y)

        # Calculando a distância euclidiana
        ####
        # Sumo não atualiza automaticamente a posição do usuário na simulação quando ele é criado.
        # Caso ele não atualiza, então iremos considerar ou a distância da Edge ou 500m
        ####
        distance = math.sqrt((server_coords[0] - x) ** 2 + (server_coords[1] - y) ** 2)
        if distance > 10000 or distance < -10000:
            traci.simulationStep()
            distance = math.sqrt((server_coords[0] - x) ** 2 + (server_coords[1] - y) ** 2)
            if distance > 10000 or distance < -10000:
                distance = 250
        return distance

        # distance, index = server_tree.query([x, y])

    # def vehicle_is_created(self,vehicle_id):
    #     #try:
    #     if vehicle_id in traci.vehicle.getIDList():
    #         return True  # O veículo existe na simulação
    #     else:
    #         return False  # O veículo não existe na simulação

    # except Exception as e:
    #     print(f"Error in vehicle is created: {str(e)}")
    #     return False

    def create_vehicle(self, vehicle_id, server_start):
        server_end = random.choice(routers)
        server_start, server_end = self.check_route(server_start, server_end)

        start_edge = random.choice(self.positions[server_start])
        end_edge = random.choice(self.positions[server_end])
        way = [start_edge, end_edge]

        traci.route.add(vehicle_id, way)
        traci.vehicle.add(vehicle_id, vehicle_id)
        traci.simulationStep()
        x, y = traci.vehicle.getPosition(vehicle_id)
        self.vehicles_info[vehicle_id] = {
            "closest_server": server_start,
            "server_end": server_end,
            "start_edge": start_edge,
            "end_edge": end_edge,
            "connected": True,
        }
        #'coord': (x,y)}

    def disconnect_vehicle(self, vehicle_id):
        # try:
        # Check if the vehicle exists in the simulation
        # if self.vehicle_is_created(vehicle_id):
        # Remove the vehicle from SUMO
        # traci.vehicle.remove(vehicle_id)

        # Remove the vehicle from the internal tracking dictionary
        if vehicle_id in self.vehicles_info:
            self.vehicles_info[vehicle_id]["connected"] = False
            # traci.simulationStep()
        # except Exception as e:
        #     print(f"Error removing vehicle {vehicle_id}: {str(e)}")

    def connect_vehicle(self, vehicle_id):
        if vehicle_id in list(self.vehicles_info.keys()):
            self.vehicles_info[vehicle_id]["connected"] = True
            # traci.simulationStep()

    # def reroute_vehicle(self, vehicle_id):
    #     #try:
    #         server_end = random.choice(server_ids)
    #         current_server = self.vehicles_info[vehicle_id]['server_end']
    #         current_server, server_end = self.check_route(current_server,server_end)
    #         edge_end = random.choice(self.positions[server_end])

    #         self.vehicles_info[vehicle_id]['end_edge'] = edge_end
    #         self.vehicles_info[vehicle_id]['server_end'] = server_end

    #         print(vehicle_id, " was rerouted")
    #         traci.vehicle.changeTarget(vehicle_id, edge_end)

    def reroute_vehicle(self, vehicle_id):
        # Filtra os servidores disponíveis, removendo os servidores que estão na
        # lista de crashed_servers
        server_end = random.choice(routers)  # Escolhe um servidor disponível
        current_server = self.vehicles_info[vehicle_id]["server_end"]
        current_server, server_end = self.check_route(current_server, server_end)
        edge_end = random.choice(self.positions[server_end])

        self.vehicles_info[vehicle_id]["end_edge"] = edge_end
        self.vehicles_info[vehicle_id]["server_end"] = server_end

        print(vehicle_id, " was rerouted")
        traci.vehicle.changeTarget(vehicle_id, edge_end)

    def check_arrival(self):
        vehicles = list(self.vehicles_info.keys())
        for vehicle_id in vehicles:
            try:
                if self.vehicles_info[vehicle_id][
                    "connected"
                ]:  # Veh desconectados são aqueles que não atualizamos mais
                    # Check if vehicle has reached the last edge of its current route
                    current_route = traci.vehicle.getRoute(vehicle_id)
                    current_route_index = traci.vehicle.getRouteIndex(vehicle_id)

                    # If the vehicle is at the last edge or close to it, reroute it
                    if current_route_index >= len(current_route) - 2:  # A bit before the last edge
                        self.reroute_vehicle(vehicle_id)
            except Exception as e:  # <-- CORRIGIDO: Captura tipada com rastreabilidade
                logger.error(
                    f"Erro no check_arrival para o veículo {vehicle_id}: {e}", exc_info=True
                )
                raise ValueError("Erro no check arrival") from e

    def update_coords(self):
        # try:
        vehicle_ids = list(self.vehicles_info.keys())
        for vehicle_id in vehicle_ids:
            self.vehicles_info[vehicle_id]["closest_server"] = self.get_closest_server(vehicle_id)
            # self.vehicles_info[vehicle_id]['coord'] = traci.vehicle.getPosition(vehicle_id)
        # except:
        #     print("Erro in update vehicle")

    def vehicle_movement_thread(self):
        try:
            while self.simulation_running:
                if self.traci_connected:
                    self.check_arrival()
                    traci.simulationStep()
                    time.sleep(1)
        except Exception as e:  # <-- CORRIGIDO
            logger.error(f"Erro fatal na thread de movimentação do SUMO: {e}", exc_info=True)
            # Tenta fechar a conexão graciosamente antes de matar o processo
            self.stop_simulation()
            sys.exit(
                1
            )  # <-- CORRIGIDO: Código 1 sinaliza falha crítica para o SO/Orquestrador

    def start_simulation(self):
        self.connect_to_sumo()
        # Thread do movimento dos veículos
        movement_thread = threading.Thread(target=self.vehicle_movement_thread)
        movement_thread.start()
        # permite o movimento dos veículos
        self.moving = True


if __name__ == "__main__":
    sim = Sumo_Luxembourg()
    sim.start_simulation()

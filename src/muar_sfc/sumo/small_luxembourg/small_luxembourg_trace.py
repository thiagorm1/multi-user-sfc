import datetime
import logging  # <-- NOVO
import math
import random
import threading
import time
from pathlib import Path

import pytz
import traci

logger = logging.getLogger(__name__)  # <-- NOVO

# Tracer utilizando SUMO.
# A simulação é controlada, por meio da biblioteca traci e das funções da classe
# Autor: Rodrigo Flexa
# Data: 17/09/2023


class Sumo_Small_Luxembourg:
    def __init__(self, config_file=Path("sumo") / "small_luxembourg" / "luxembourg.sumocfg"):

        self.config_file = config_file

        # self.csv_file = open('dados_veiculos.csv', mode='a', newline='')
        # self.csv_writer = csv.writer(self.csv_file)

        # Verifica se o arquivo está vazio; se estiver, escreve o cabeçalho
        # if self.csv_file.tell() == 0:
        #     self.csv_writer.writerow(['Tempo', 'Velocidade (km/h)','Aceleracao'])

        # Se o Traci está conectado
        self.traci_connected = False
        # Diz se a simulação está rodando
        self.simulation_running = False
        self.counting = 0
        self.veiculos_a_excluir = []
        self.mutex = threading.Lock()  # Mutex para garantir acesso seguro à lista
        self.routes_created = []
        self.players_data = {}

        self.topology = {1: (2884, 6892), 2: (4351, 6892), 3: (6446, 6892), 4: (8403, 6500)}

        self.positions = {
            1: ["-34375#3", "-34937", "--34937", "-34375#2", "-34375#1"],
            2: ["-35399#2", "-35075#2", "--35075#2", "--35399#2", "--35075#1"],
            3: ["-34181#2", "-34435#4", "-35240", "-34435#5", "--34435#4"],
            4: ["-34417#2", "-34732#0", "--34732#1", "-34431#3", "-34732#1"],
        }

    def get_datetime(self):
        utc_now = pytz.utc.localize(datetime.datetime.utcnow())
        currentDT = utc_now.astimezone(pytz.timezone("America/Belem"))
        return currentDT.strftime("%Y-%m-%d %H:%M:%S")

    def calculate_kmph(self, m_per_s):
        return round(m_per_s * 3.6, 2)

    def connect_to_sumo(self):
        traci.start(["sumo", "-c", self.config_file])
        self.traci_connected = True
        self.simulation_running = True  # Atualize a flag de simulação em execução

    def stop_simulation(self):
        self.simulation_running = False
        if self.traci_connected:
            traci.close()
            self.traci_connected = False

        # Fecha o arquivo CSV
        self.csv_file.close()

    def check_route(self, server_start, server_end):
        while server_start == server_end:
            server_end = random.choice(list(self.positions.keys()))
        return server_start, server_end

    def create_(self, player, trip_id, vehicle_id, server_start, server_end):
        start_edge = random.choice(self.positions[server_start])
        end_edge = random.choice(self.positions[server_end])

        caminho = [start_edge, end_edge]
        if trip_id not in self.routes_created:
            traci.route.add(trip_id, caminho)
            self.routes_created.append(trip_id)

        traci.vehicle.add(vehicle_id, trip_id)
        x, y = traci.vehicle.getPosition(vehicle_id)
        self.players_data[int(player)] = {
            "server_end": server_end,
            "start_edge": start_edge,
            "end_edge": end_edge,
            "trip_id": trip_id,
            "coord": (x, y),
            "closest_server": server_start,
            "player_on": 1,
        }

    # TODO tem que mudar pra ficar mais veloz
    def update_coords(self):
        vehicles = traci.vehicle.getIDList()
        for vehicle_id in vehicles:
            x, y = traci.vehicle.getPosition(vehicle_id)
            self.players_data[int(vehicle_id.split("_")[-1])]["coord"] = (x, y)

    def get_closest_server(self, player_id):
        closest_server_id = self.players_data[player_id]["closest_server"]
        x, y = self.players_data[player_id]["coord"]
        if x == 0:
            return closest_server_id
        distanciazona = 100000000

        for server_id, (server_x, server_y) in self.topology.items():
            distance = math.sqrt((x - server_x) ** 2 + (y - server_y) ** 2)

            if distance < distanciazona:
                distanciazona = distance
                closest_server_id = server_id
        return closest_server_id

    def vehicle_is_created(self, id):
        try:
            players = list(self.players_data.keys())
            traci.vehicle.getIDList()

            # Verifica se o veículo com o ID desejado está na lista
            return id in players
        except Exception as e:
            logger.error(f"Error in vehicle is created: {e}", exc_info=True)  # <-- CORRIGIDO
            self.stop_simulation()
            return -1

    def create_vehicle(self, player, server_start):
        try:
            server_end = random.choice(list(self.positions.keys()))
            server_start, server_end = self.check_route(server_start, server_end)

            vehicle_id = f"vehicle_{player}"
            trip_id = f"trip_{player}"
            self.create_(player, trip_id, vehicle_id, server_start, server_end)

        except Exception as e:
            logger.error(f"Error in vehicle creation: {e}", exc_info=True)  # <-- CORRIGIDO
            self.stop_simulation()

    def delete_vehicle(self, vehicle_id):
        player = int(vehicle_id.split("_")[-1])
        if self.vehicle_is_created(player):
            # Marcação do veículo para exclusão
            with self.mutex:
                self.veiculos_a_excluir.append(vehicle_id)
            # print(f"{vehicle_id} marked for deletion")
        else:
            print("Trying to delete vehicle, but Not found")

    def reroute_vehicle(self, vehicle_id):
        player = int(vehicle_id.split("_")[-1])

        server_end = random.choice(list(self.positions.keys()))
        new_edge = random.choice(self.positions[server_end])
        current_server = self.players_data[player]["server_end"]

        current_server, server_end = self.check_route(current_server, server_end)

        self.players_data[player]["end_edge"] = new_edge
        self.players_data[player]["server_end"] = server_end

        # print(vehicle_id, " was rerouted")
        traci.vehicle.changeTarget(vehicle_id, new_edge)

    def update_vehicles(self):
        vehicles = traci.vehicle.getIDList()
        for vehicle_id in vehicles:
            try:
                arrived = (
                    traci.vehicle.getRouteIndex(vehicle_id)
                    == len(traci.vehicle.getRoute(vehicle_id)) - 1
                )
                if arrived:
                    self.reroute_vehicle(vehicle_id)
            except Exception as e:  # <-- CORRIGIDO: Captura tipada com EAFP preservado
                logger.error(f"Erro ao redirecionar veículo {vehicle_id}: {e}", exc_info=True)

    def vehicle_movement_thread(self):
        while self.simulation_running:
            # Mova os veículos apenas se traci_conectado
            if self.traci_connected:
                try:
                    traci.simulationStep()
                    self.update_coords()
                    self.update_vehicles()

                    # Coleta as métricas dos veículos
                    # vehicles = traci.vehicle.getIDList()
                    # for vehicle_id in vehicles:
                    #     player = int(vehicle_id.split("_")[-1])
                    #     velocity = self.calculate_kmph(traci.vehicle.getSpeed(vehicle_id))
                    #     acceleration = traci.vehicle.getAcceleration(vehicle_id)

                    #     # Escreve as métricas no arquivo CSV
                    #     with self.mutex:
                    #         self.csv_writer.writerow(
                    #             [self.get_datetime(), velocity, acceleration]
                    #         )

                    time.sleep(1)
                except Exception as e:
                    logger.critical(
                        f"Error in vehicle movement thread: {e}", exc_info=True
                    )  # <-- CORRIGIDO
                    self.stop_simulation()
                    import sys

                    sys.exit(1)

    def start_sumo_simulation(self):
        self.connect_to_sumo()
        # Thread do movimento dos veículos
        movement_thread = threading.Thread(target=self.vehicle_movement_thread)
        movement_thread.start()
        # permite o movimento dos veículos
        self.moving = True


if __name__ == "__main__":
    sim = Sumo_Small_Luxembourg()
    sim.start_sumo_simulation()

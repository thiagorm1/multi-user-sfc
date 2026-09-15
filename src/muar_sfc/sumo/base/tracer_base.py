import datetime
import threading
import time
import traceback
from abc import ABC, abstractmethod

import pytz
import traci


class AbstractTracer(ABC):
    def __init__(self, config_file):
        self.config_file = config_file
        self.traci_connected = False
        self.simulation_running = False

    def get_datetime(self):
        utc_now = pytz.utc.localize(datetime.datetime.utcnow())
        return utc_now.astimezone(pytz.timezone("America/Belem")).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def calculate_kmph(m_per_s):
        return round(m_per_s * 3.6, 2)

    def connect_to_sumo(self):
        traci.start(["sumo", "-c", self.config_file])
        self.traci_connected = True
        self.simulation_running = True

    def stop_simulation(self):
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
        self.simulation_running = False

    @abstractmethod
    def create_vehicle(self, player, server_start):
        """
        Create a vehicle in the SUMO simulation.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def disconnect_vehicle(self, vehicle_id):
        """
        Delete a vehicle from the SUMO simulation.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def reroute_vehicle(self, vehicle_id):
        """
        Reroute a vehicle in the SUMO simulation.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def check_arrival(self):
        """
        Check if a vehicle has reached its destination.
        Must be implemented by subclasses.
        """
        pass

    def vehicle_movement_thread(self):
        while self.simulation_running:
            if self.traci_connected:
                try:
                    traci.simulationStep()
                    self.check_arrival()
                    time.sleep(1)
                except Exception as e:
                    print(f"Unexpected error in vehicle movement: {e}")
                    traceback.print_exc()

    def start_simulation(self):
        self.connect_to_sumo()
        movement_thread = threading.Thread(target=self.vehicle_movement_thread)
        movement_thread.start()

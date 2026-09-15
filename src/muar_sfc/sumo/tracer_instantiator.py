from muar_sfc.sumo.luxembourg.luxembourg_trace import Sumo_Luxembourg
from muar_sfc.sumo.small_luxembourg.small_luxembourg_trace import Sumo_Small_Luxembourg


class TracerInstantiator:
    def instantiate_tracer(self, type):
        if type == "luxembourg" or type == "luxembourgv2":
            tracer = Sumo_Luxembourg()
        elif type == "small luxembourg":
            tracer = Sumo_Small_Luxembourg()
        else:
            raise ValueError("tracer not found")

        return tracer

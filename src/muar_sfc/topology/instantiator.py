from muar_sfc.topology.experimental.nsfnet import NSFNet
from muar_sfc.topology.predefined.luxembourg import Luxembourg
from muar_sfc.topology.predefined.luxembourg_netv2 import LuxembourgV2
from muar_sfc.topology.predefined.paloalto import PaloAlto
from muar_sfc.topology.predefined.sample_topology import SampleTopology
from muar_sfc.topology.predefined.santamonica import SantaMonica


class TopologyInstantiator:
    def __init__(self):
        self.topology_classes = {
            "nsfnet": NSFNet,
            "paloalto": PaloAlto,
            "santamonica": SantaMonica,
            "luxembourg": Luxembourg,
            "luxembourgv2": LuxembourgV2,
            "test": SampleTopology,
        }

    def instantiate_topology(self, type, eco_effi_ratio):
        try:
            topology_class = self.topology_classes[type]
            return topology_class(eco_effi_ratio)
        except KeyError as e:
            raise ValueError(f"Topology '{type}' not found") from e

from muar_sfc.core.vnf import VNF, VNFType


class VNFSRC(VNF):
    """
    This vnf is output the same volumn of input
    """

    def __init__(self):
        VNF.__init__(self, id="src")
        self.type = VNFType.SRC
        self.set_cpu_request(0)
        self.set_cache_request(0)
        self.set_outcome_interface_bandwidth(0)

    def vnf_bw(self, i):
        return i

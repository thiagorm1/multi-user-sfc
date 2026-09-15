from muar_sfc.core.vnf import VNFType
from muar_sfc.vnfs.vnf_type_1 import VNFType1

"""
{type: "TYPE1", "name": "vnf1", "CPU": 100},
"""


class VNFGenerator:
    @classmethod
    def generate(cls, vnf_dict):
        vnf_type = vnf_dict["type"]
        vnf_name = vnf_dict["name"]
        vnf_cpu_request = vnf_dict["CPU"]
        vnf_cache_request = vnf_dict["cache"]
        vnf_in_bw_request = vnf_dict["in_bw"]
        vnf_out_bw_request = vnf_dict["out_bw"]
        if vnf_type == VNFType.TYPE1:
            vnf = VNFType1(vnf_name)
            vnf.set_cpu_request(vnf_cpu_request)
            vnf.set_cache_request(vnf_cache_request)
            vnf.set_income_interface_bandwidth(vnf_in_bw_request)
            vnf.set_outcome_interface_bandwidth(vnf_out_bw_request)
            return vnf
        elif vnf_type == VNFType.TYPE2:
            print("TYPE 2 is not defined. exit")
            exit(1)
        else:
            print("TYPE is existed, exit")
            exit(1)

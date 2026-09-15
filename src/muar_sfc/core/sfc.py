import networkx as nx
from muar_sfc.core.vnf import VNF


class SFC:
    def __init__(self, vnf_src, vnf_dst):
        self.number_of_vnfs = 0  # This is not include src and dst
        self.vnfs = {}  # This is not include src and dst
        self.vnfs_dict = None
        self.src = vnf_src
        self.dst = vnf_dst
        self.dst_node = None
        self.closer_router = None

        self.graph = nx.DiGraph()  # Grafo direcionado nativo da SFC (DAG)
        self.sync_tolerance = 5.0  # Tolerância máxima de sincronização inter-modal (ms)

        self.link_bandwidth_dict = {}
        self.latency_request = 0
        self.id = None
        self.input_throughput = 0
        self.duration = 0
        self.arrival_time = 0
        self.depart_time = 0

        # Adiciona nós base ao grafo interno
        self.graph.add_node(self.src.id, vnf=self.src)
        self.graph.add_node(self.dst.id, vnf=self.dst)

    def __str__(self):
        attrs = vars(self)
        return ", ".join(f"{k}: {v}" for k, v in attrs.items())

    def __hash__(self):
        return hash(str(self))

    def add_vnf(self, vnf):
        self.vnfs[vnf.id] = vnf
        self.number_of_vnfs += 1
        self.graph.add_node(vnf.id, vnf=vnf)

    def set_src_substrate_node(self, substrate_node):
        self.src.assign_substrate_node(substrate_node)

    def set_dst_substrate_node(self, substrate_node):
        self.dst_node = substrate_node
        self.dst.assign_substrate_node(substrate_node)

    def is_dag(self) -> bool:
        """Verifica se a SFC forma um Grafo Acíclico Dirigido (DAG)."""
        return nx.is_directed_acyclic_graph(self.graph)

    def add_dependency(self, vnf1, vnf2, bandwidth=None):
        """Conecta duas VNFs em arco direcionado vnf1 -> vnf2 no DAG."""
        if bandwidth is None:
            bandwidth = vnf1.get_outcome_interface_bandwidth() or 0
        self.graph.add_edge(vnf1.id, vnf2.id, bandwidth=bandwidth)
        vnf1.add_next_vnf(vnf2)
        vnf2.add_previous_vnf(vnf1)
        self.link_bandwidth_dict[(vnf1.id, vnf2.id)] = bandwidth

    def connect_two_vnfs(self, vnf1, vnf2):
        self.add_dependency(vnf1, vnf2)

    def change_link_bandwidth_request_to(self, vnf_id, bw):
        vnf = self.get_vnf_by_id(vnf_id)
        if not vnf.next_vnf:
            return
        next_vnf = vnf.next_vnf
        vnf.set_outcome_interface_bandwidth(bw)
        next_vnf.set_income_interface_bandwidth(bw)
        self.link_bandwidth_dict[(vnf.id, next_vnf.id)] = bw

    def change_node_cpu_request_to(self, vnf_id, cpu):
        vnf = self.get_vnf_by_id(vnf_id)
        vnf.set_cpu_request(cpu)

    def get_number_of_vnfs(self):
        return self.number_of_vnfs

    def get_vnf_cpu_request(self, vnf):
        return vnf.get_cpu_request()

    def get_vnf_cache_request(self, vnf):
        return vnf.get_cache_request()

    def get_link_bandwidth_request(self, vnf1_id, vnf2_id):
        if not vnf1_id or not vnf2_id:
            return 0
        return self.link_bandwidth_dict[(vnf1_id, vnf2_id)]

    def get_next_vnf(self, vnf):
        return vnf.get_next_vnf()

    def get_previous_vnf(self, vnf):
        return vnf.get_previous_vnf()

    def get_substrate_node(self, vnf):
        # print('vnf sfc.py')
        # print(vnf)
        return vnf.get_substrate_node()

    def get_src_vnf(self):
        return self.src

    def get_dst_vnf(self):
        return self.dst

    def get_vnf_by_id(self, vnf_id):
        # This method could return the src and dst vnf
        if vnf_id == "src":
            return self.src
        if vnf_id == "dst":
            return self.dst
        return self.vnfs[vnf_id]

    def set_latency_request(self, latency_request):
        self.latency_request = latency_request

    def get_latency_request(self):
        return self.latency_request

    def set_input_throughput(self, tp):
        self.input_throughput = tp
        self.update()

    def update(self):
        # if not self.input_throughput:
        #    return
        # tp = self.input_throughput
        vnf = self.src
        self.link_bandwidth_dict[(vnf.id, vnf.next_vnf.id)] = vnf.get_outcome_interface_bandwidth()
        vnf = vnf.next_vnf

        while vnf.next_vnf:
            # vnf.vnf_function()
            next_vnf = vnf.next_vnf
            link_bw = vnf.get_outcome_interface_bandwidth()
            # if not link_bw:
            #    print("outcome bandwith not set")
            #    return
            next_vnf.set_income_interface_bandwidth(link_bw)
            self.link_bandwidth_dict[(vnf.id, next_vnf.id)] = link_bw
            vnf = next_vnf

    def get_all_paths(self) -> list[list[str]]:
        """Retorna todos os caminhos direcionados de src até dst no DAG."""
        if self.src.id in self.graph and self.dst.id in self.graph:
            return list(nx.all_simple_paths(self.graph, source=self.src.id, target=self.dst.id))
        return []

    def get_critical_path_latency(self, path_latencies: dict[tuple[str, str], float], comp_latencies: dict[str, float]) -> float:
        """
        Calcula a latência ponta a ponta pelo caminho mais longo (caminho crítico do DAG):
        T^{e2e} = max_{p in paths(src -> dst)} sum_{u -> v in p} (lat_comm(u, v) + lat_comp(v))
        """
        paths = self.get_all_paths()
        if not paths:
            # Fallback para soma simples se não houver caminhos no grafo
            return sum(comp_latencies.values()) + sum(path_latencies.values())

        max_latency = 0.0
        for path in paths:
            path_lat = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_lat += path_latencies.get((u, v), 0.0)
                path_lat += comp_latencies.get(v, 0.0)
            max_latency = max(max_latency, path_lat)

        return max_latency

    def get_sync_differential(
        self,
        branch_a_vnfs: list[str],
        branch_b_vnfs: list[str],
        path_latencies: dict[tuple[str, str], float],
        comp_latencies: dict[str, float],
    ) -> float:
        """
        Calcula o diferencial de sincronização inter-modal delta_sync:
        delta_sync = | Latencia(Ramo A) - Latencia(Ramo B) |
        """
        lat_a = sum(comp_latencies.get(v, 0.0) for v in branch_a_vnfs)
        lat_b = sum(comp_latencies.get(v, 0.0) for v in branch_b_vnfs)
        for (u, v), lat in path_latencies.items():
            if u in branch_a_vnfs or v in branch_a_vnfs:
                lat_a += lat
            if u in branch_b_vnfs or v in branch_b_vnfs:
                lat_b += lat

        return abs(lat_a - lat_b)


if __name__ == "__main__":
    src_vnf = VNF("src")
    src_vnf.set_cpu_request(0)
    src_vnf.set_outcome_interface_bandwidth(20)
    dst_vnf = VNF("dst")
    dst_vnf.set_cpu_request(0)
    sfc = SFC(src_vnf, dst_vnf)
    print(sfc.get_number_of_vnfs())
    vnf1 = VNF(1)
    vnf1.set_cpu_request(10)
    vnf1.set_outcome_interface_bandwidth(10)
    vnf2 = VNF(2)
    vnf2.set_cpu_request(20)
    vnf2.set_outcome_interface_bandwidth(20)
    vnf3 = VNF(3)
    vnf3.set_cpu_request(30)
    vnf3.set_outcome_interface_bandwidth(30)
    sfc.add_vnf(vnf1)
    sfc.add_vnf(vnf2)
    sfc.add_vnf(vnf3)
    sfc.connect_two_vnfs(src_vnf, vnf1)
    sfc.connect_two_vnfs(vnf1, vnf2)
    sfc.connect_two_vnfs(vnf2, vnf3)
    sfc.connect_two_vnfs(vnf3, dst_vnf)
    print(sfc.get_number_of_vnfs())
    print(sfc.get_previous_vnf(src_vnf))

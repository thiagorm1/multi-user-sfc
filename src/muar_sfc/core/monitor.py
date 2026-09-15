class Monitor:
    def __init__(self, nw):
        pass
        self.substrate_network = nw
        # self.scheduler = sched.scheduler(time.time, time.sleep)
        self.node_info = {}

    def start(self):

        pass

    def stop(self):
        pass

    def collect_information(self):
        pass

    def get_nodes_information(self):
        for node in self.substrate_network.nodes():
            self.node_info[node] = self.get_node_information(node)

    def output_node_information(self):
        # for node, info in self.node_info.items():
        #     print node
        #     print info
        self.substrate_network.print_out_nodes_information()

    def get_node_information(self, node_id):
        cpu_used = self.substrate_network.get_node_cpu_used(node_id)
        cpu_free = self.substrate_network.get_node_cpu_free(node_id)
        cpu_capacity = self.substrate_network.get_node_cpu_capacity(node_id)
        sfc_vnf_list = self.substrate_network.get_node_sfc_vnf_list(node_id)
        return (cpu_used, cpu_free, cpu_capacity, sfc_vnf_list)

    def update(self):
        self.substrate_network.update()
        self.get_nodes_information()

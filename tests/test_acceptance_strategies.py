import copy
from types import SimpleNamespace

import pytest

from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.admission_queue import AdmissionQueue
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.capacity import apply_capacity_profile
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.orchestration.placement import reserve_node
from muar_sfc.core.vnf import VNF


@pytest.fixture
def network():
    net = Net2()
    net.add_node(0, "router", w_channel_capacity=2000)
    net.add_node(6, "router", w_channel_capacity=2000)
    for node in range(1, 6):
        net.add_node(node, "server", cpu_capacity=100, cache_capacity=500, ips=1)
    for u, v, lat in [(0, 1, 1), (1, 2, 2), (1, 3, 3), (2, 4, 2), (3, 4, 1), (4, 5, 1), (5, 6, 1)]:
        net.add_edge(u, v, bandwidth_capacity=1000, latency=lat)
    net.set_sharing_params(True)
    net.set_bandwidth_sharing(True)
    return net


def dag_request(player=1):
    names = [
        "IA_DET_FT_1",
        "MA_region_1",
        f"UNI_p{player}_1",
        f"RE_p{player}_1",
        f"EC_TC_p{player}_1",
    ]
    dependencies = [
        ("src", names[0], 150),
        (names[0], names[1], 80),
        (names[0], names[2], 160),
        (names[1], names[3], 80),
        (names[2], names[3], 160),
        (names[3], names[4], 97),
        (names[4], "dst", 97),
    ]
    return SFCGenerator(
        {
            "name": f"sfc_dag_p{player}_1",
            "vnf_list": [
                {
                    "type": 2,
                    "name": name,
                    "CPU": 10,
                    "cache": 33 if i == 1 else 0,
                    "in_bw": 150,
                    "out_bw": 150,
                }
                for i, name in enumerate(names)
            ],
            "dependencies": dependencies,
            "bandwidth": 97,
            "closer_router": 1,
            "src_node": 0,
            "dst_node": 6,
            "latency": 50,
            "duration": 120,
        }
    ).generate()


def placed_plan(network, player=1):
    sfc = dag_request(player)
    env = MAPPO_SFC_Env(
        network.graph,
        sfc,
        n_regions=2,
        cloud_node=1,
        share=True,
        shareband=network.allocator.shareable_band,
    )
    env.excluded_nodes = {
        v.id: set(range(1, 6)) - {i} for i, v in enumerate(env.vnf_sequence, start=1)
    }
    done = False
    while not done:
        _, _, _, done, _ = env.step({0: env.max_nodes_per_region})
    assert env.success
    sfc.edge_routes = env.edge_routes
    sfc.edge_latencies = env.edge_latencies
    sfc.comp_latencies = env.comp_latencies
    return sfc, env


@pytest.mark.parametrize("release_order", [(1, 2), (2, 1)])
def test_shared_instances_and_streams_release_only_last_reference(network, release_order):
    requests = {}
    for player in (1, 2):
        sfc, env = placed_plan(network, player)
        network.deploy_sfc(sfc, env.route_info)
        requests[player] = sfc
    assert network.graph.nodes[1]["cpu_used"] == 10
    assert network.graph.nodes[2]["cache_used"] == 33
    assert network.graph.edges[0, 1]["bandwidth_used"] == 150
    assert network.graph.edges[1, 2]["bandwidth_used"] == 80
    assert network.graph.edges[1, 3]["bandwidth_used"] == 320
    network.undeploy_sfc(requests[release_order[0]].id)
    assert network.graph.nodes[1]["cpu_used"] == 10
    assert network.graph.edges[0, 1]["bandwidth_used"] == 150
    network.undeploy_sfc(requests[release_order[1]].id)
    assert all(
        d["cpu_used"] == 0 and d["cache_used"] == 0 for _, d in network.graph.nodes(data=True)
    )
    assert all(
        d["bandwidth_used"] == 0 and not d["services_in_transit"]
        for _, _, d in network.graph.edges(data=True)
    )
    assert network.metrics.total_bandwidth_used == 0
    assert network.metrics.total_cpu_saved == 0
    assert network.metrics.total_cache_saved == 0


def test_dag_reserves_every_branch_and_terminal_link(network):
    sfc, env = placed_plan(network)
    assert set(sfc.edge_routes) == set(sfc.graph.edges)
    assert sfc.vnfs["IA_DET_FT_1"].get_income_interface_bandwidth() == 150
    assert sfc.vnfs["RE_p1_1"].get_income_interface_bandwidth() == 240
    assert env.total_latency == pytest.approx(8.078333333333333)
    assert env.sync_differential < sfc.sync_tolerance
    network.deploy_sfc(sfc, env.route_info)
    assert network.graph.edges[1, 3]["bandwidth_used"] == 160
    assert network.graph.edges[3, 4]["bandwidth_used"] == 160
    assert network.graph.edges[5, 6]["bandwidth_used"] == 97
    assert network.calculate_sfc_total_latency(sfc.id) == pytest.approx(env.total_latency)


def test_failed_deployment_rolls_back_all_reservations(network):
    sfc, env = placed_plan(network)
    network.graph.edges[5, 6]["bandwidth_capacity"] = 0
    with pytest.raises(ValueError):
        network.deploy_sfc(sfc, env.route_info)
    assert sfc.id not in network.sfc_dict
    assert all(
        d["cpu_used"] == 0 and d["cache_used"] == 0 for _, d in network.graph.nodes(data=True)
    )
    assert all(d["bandwidth_used"] == 0 for _, _, d in network.graph.edges(data=True))


def test_planning_does_not_mutate_committed_nested_services(network):
    original = copy.deepcopy(network.graph.nodes[1]["services"])
    placed_plan(network)
    assert network.graph.nodes[1]["services"] == original


def test_same_service_id_with_different_requirements_is_rejected(network):
    vnf = VNF("MA_region_1")
    vnf.set_cpu_request(10)
    vnf.set_cache_request(33)
    sfc = SimpleNamespace(id="sfc_dag_p1_1")
    reserve_node(network.graph, sfc, vnf, 1, True)
    vnf.set_cpu_request(20)
    with pytest.raises(ValueError, match="incompat"):
        reserve_node(network.graph, sfc, vnf, 1, True)


def test_queue_never_counts_pending_requests_as_accepted():
    queue = AdmissionQueue(True, 5, 1)
    request = queue.register([SimpleNamespace(id="req")], 0)
    assert queue.next_ready(0) is request
    assert not queue.attempt(request, 0, lambda _: (False, []))
    assert queue.snapshot()["accepted"] == 0
    assert queue.snapshot()["pending"] == 1
    assert queue.next_ready(1) is request
    assert queue.attempt(request, 1, lambda _: (True, [12.0]))
    assert queue.snapshot()["accepted_after_wait"] == 1
    assert queue.snapshot()["mean_wait_seconds"] == 1


def test_queue_expires_without_retry_past_deadline():
    queue = AdmissionQueue(True, 1, 1)
    request = queue.register([SimpleNamespace(id="req")], 0)
    queue.next_ready(0)
    queue.attempt(request, 0, lambda _: (False, []))
    queue.next_ready(1)
    assert queue.attempt(request, 1, lambda _: (False, []))
    assert request.status == "expired"
    assert queue.pending == 0
    assert queue.snapshot()["acceptance_pct"] == 0


def test_successful_deploy_after_deadline_is_rolled_back():
    queue = AdmissionQueue(True, 1, 1, clock=lambda: 2)
    request = queue.register([SimpleNamespace(id="req")], 0)
    queue.next_ready(0)
    rolled_back = []
    queue.attempt(request, 0, lambda _: (True, [10]), lambda batch: rolled_back.extend(batch))
    assert request.status == "expired"
    assert len(rolled_back) == 1
    assert queue.snapshot()["accepted"] == 0


def test_capacity_profile_has_finite_budgets_and_cannot_be_applied_twice(network):
    settings = SimulationSettings(
        cpu_capacity_factor=3.6,
        cache_capacity_factor=3.6,
        link_capacity_factor=8,
        wireless_capacity_factor=4,
        source_bandwidth_mbps=40000,
    )
    apply_capacity_profile(network, settings)
    assert network.graph.nodes[1]["cpu_capacity"] == 360
    assert network.graph.edges[0, 1]["bandwidth_capacity"] == 40000
    assert network.graph.edges[1, 2]["bandwidth_capacity"] == 8000
    assert network.graph.nodes[0]["w_channel_capacity"] == 8000
    with pytest.raises(ValueError, match="já aplicado"):
        apply_capacity_profile(network, settings)


def test_queue_admits_after_physical_session_expiry(network):
    from muar_sfc.algorithms.mappo import MAPPOAlgorithm

    network.set_bandwidth_sharing(False)
    network.graph.edges[0, 1]["bandwidth_capacity"] = 150
    first, env = placed_plan(network)
    network.deploy_sfc(first, env.route_info)
    second = dag_request(2)
    algorithm = MAPPOAlgorithm(n_regions=2, cloud_node=1)
    settings = SimulationSettings(share=True, shareband=False, mappo_search_attempts=2)
    queue = AdmissionQueue(True, 2, 1)
    request = queue.register([second], 0)

    def deploy(batch):
        sfc = batch[0]
        algorithm.install_substrate_network(network.graph)
        algorithm.install_SFC(sfc)
        success = algorithm.start_algorithm(args=settings)
        if success:
            network.deploy_sfc(sfc, algorithm.get_route_info())
        return success, [algorithm.get_latency()] if success else []

    queue.next_ready(0)
    assert not queue.attempt(request, 0, deploy)
    assert second.id not in network.sfc_dict
    network.undeploy_sfc(first.id)
    queue.next_ready(1)
    assert queue.attempt(request, 1, deploy)
    assert queue.snapshot()["accepted_after_wait"] == 1
    assert second.id in network.sfc_dict
    assert network.graph.edges[0, 1]["bandwidth_used"] == 150
    network.undeploy_sfc(second.id)
    assert network.graph.edges[0, 1]["bandwidth_used"] == 0


def test_generated_muar_dag_reserves_renderer_output_before_transcoding():
    from muar_sfc.controllers.sfc_queue import SFCQueue
    from muar_sfc.core.poisson_emitter import PoissonEmitter
    from muar_sfc.core.scenarios.muar import MuarScenario

    settings = SimulationSettings(dag=True, n_players=1, n_sessions=2)
    queue = SFCQueue()
    topology = SimpleNamespace(get_topology_info=lambda: {"routers": [1]})
    scene = MuarScenario(settings, queue, topology, PoissonEmitter(3))
    scene.generate_sfc_session(None)
    sfc = queue.peek_sfc()[0]
    assert sfc.get_latency_request() == settings.latency
    assert sfc.graph.edges["RE_p1_1", "EC_TC_p1_1"]["bandwidth"] == 240
    assert sfc.vnfs["EC_TC_p1_1"].get_income_interface_bandwidth() == 240
    assert sfc.graph.edges["EC_TC_p1_1", "dst"]["bandwidth"] == 97


def test_backup_never_reuses_primary_processing_or_reserved_streams(network):
    primary, env = placed_plan(network)
    network.deploy_sfc(primary, env.route_info)
    backup = dag_request()
    backup.id += "_b"
    backup.is_backup = True
    backup.edge_routes = primary.edge_routes
    backup.edge_latencies = primary.edge_latencies
    backup.comp_latencies = primary.comp_latencies
    network.deploy_sfc(backup, env.route_info)
    assert network.graph.nodes[1]["cpu_used"] == 20
    assert network.graph.edges[0, 1]["bandwidth_used"] == 150
    assert network.graph.edges[0, 1]["bandwidth_reserved"] == 150
    network.undeploy_sfc(backup.id)
    assert network.graph.nodes[1]["cpu_used"] == 10
    assert network.graph.edges[0, 1]["bandwidth_reserved"] == 0
    network.undeploy_sfc(primary.id)
    assert network.graph.edges[0, 1]["bandwidth_used"] == 0


def test_duplicate_deployment_does_not_charge_resources_twice(network):
    sfc, env = placed_plan(network)
    network.deploy_sfc(sfc, env.route_info)
    with pytest.raises(ValueError, match="já está implantada"):
        network.deploy_sfc(sfc, env.route_info)
    assert network.graph.nodes[1]["cpu_used"] == 10
    assert network.graph.edges[0, 1]["bandwidth_used"] == 150

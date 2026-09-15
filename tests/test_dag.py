import pytest
import networkx as nx

from muar_sfc.core.vnf import VNF
from muar_sfc.core.sfc import SFC
from muar_sfc.vnfs.vnf_type_src import VNFSRC
from muar_sfc.vnfs.vnf_type_dst import VNFDST
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.algorithms.dag_greedy import DAGGreedyAlgorithm


def test_vnf_dag_links():
    v1 = VNF("IA")
    v2 = VNF("MA")
    v3 = VNF("UNI")
    v4 = VNF("RE")

    # IA -> MA e IA -> UNI (Fork)
    v1.add_next_vnf(v2)
    v1.add_next_vnf(v3)
    v2.add_previous_vnf(v1)
    v3.add_previous_vnf(v1)

    # MA -> RE e UNI -> RE (Join)
    v2.add_next_vnf(v4)
    v3.add_next_vnf(v4)
    v4.add_previous_vnf(v2)
    v4.add_previous_vnf(v3)

    assert len(v1.next_vnfs) == 2
    assert len(v4.previous_vnfs) == 2
    # Retrocompatibilidade de propriedades
    assert v1.next_vnf == v2
    assert v4.previous_vnf == v2


def test_sfc_dag_properties():
    src = VNFSRC()
    dst = VNFDST()
    sfc = SFC(src, dst)

    ia = VNF("IA")
    ma = VNF("MA")
    uni = VNF("UNI")
    re = VNF("RE")

    sfc.add_vnf(ia)
    sfc.add_vnf(ma)
    sfc.add_vnf(uni)
    sfc.add_vnf(re)

    sfc.add_dependency(src, ia, bandwidth=150)
    sfc.add_dependency(ia, ma, bandwidth=80)
    sfc.add_dependency(ia, uni, bandwidth=160)
    sfc.add_dependency(ma, re, bandwidth=80)
    sfc.add_dependency(uni, re, bandwidth=160)
    sfc.add_dependency(re, dst, bandwidth=100)

    assert sfc.is_dag() is True

    paths = sfc.get_all_paths()
    assert len(paths) == 2
    assert ["src", "IA", "MA", "RE", "dst"] in paths
    assert ["src", "IA", "UNI", "RE", "dst"] in paths


def test_sfc_generator_dag():
    vnf_list = [
        {"type": 2, "name": "IA", "CPU": 10, "cache": 0, "in_bw": 150, "out_bw": 150, "latency": 1.0},
        {"type": 2, "name": "MA", "CPU": 20, "cache": 240, "in_bw": 80, "out_bw": 80, "latency": 2.0},
        {"type": 2, "name": "UNI", "CPU": 30, "cache": 0, "in_bw": 160, "out_bw": 160, "latency": 3.0},
        {"type": 2, "name": "RE", "CPU": 25, "cache": 0, "in_bw": 240, "out_bw": 240, "latency": 2.5},
    ]

    dependencies = [
        ("src", "IA", 150),
        ("IA", "MA", 80),
        ("IA", "UNI", 160),
        ("MA", "RE", 80),
        ("UNI", "RE", 160),
        ("RE", "dst", 100),
    ]

    sfc_dict = {
        "name": "sfc_dag_test",
        "vnf_list": vnf_list,
        "dependencies": dependencies,
        "bandwidth": 100,
        "closer_router": 1,
        "src_node": 0,
        "dst_node": 10,
        "latency": 50,
        "duration": 60,
    }

    sfc = SFCGenerator(sfc_dict).generate()
    assert sfc.is_dag() is True
    assert sfc.get_number_of_vnfs() == 4
    assert len(sfc.get_all_paths()) == 2


def test_dag_greedy_algorithm():
    # Cria substrato simples para teste
    graph = nx.Graph()
    graph.add_node(0, type="server", cpu_capacity=1000, cpu_used=0, cache_capacity=1000, cache_used=0, servers_mips=100000)
    graph.add_node(1, type="server", cpu_capacity=1000, cpu_used=0, cache_capacity=1000, cache_used=0, servers_mips=100000)
    graph.add_node(2, type="server", cpu_capacity=1000, cpu_used=0, cache_capacity=1000, cache_used=0, servers_mips=100000)
    graph.add_node(10, type="server", cpu_capacity=1000, cpu_used=0, cache_capacity=1000, cache_used=0, servers_mips=100000)

    graph.add_edge(0, 1, latency=1.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(1, 2, latency=1.0, bandwidth_capacity=1000, bandwidth_used=0)
    graph.add_edge(2, 10, latency=1.0, bandwidth_capacity=1000, bandwidth_used=0)

    vnf_list = [
        {"type": 2, "name": "IA_DET_FT_1", "CPU": 10, "cache": 0, "in_bw": 150, "out_bw": 150, "latency": 1.0},
        {"type": 2, "name": "MA_region_1", "CPU": 20, "cache": 100, "in_bw": 80, "out_bw": 80, "latency": 2.0},
        {"type": 2, "name": "UNI_p1_1", "CPU": 30, "cache": 0, "in_bw": 160, "out_bw": 160, "latency": 2.0},
        {"type": 2, "name": "RE_p1_1", "CPU": 25, "cache": 0, "in_bw": 240, "out_bw": 240, "latency": 2.5},
        {"type": 2, "name": "EC_TC_p1_1", "CPU": 15, "cache": 0, "in_bw": 100, "out_bw": 100, "latency": 1.5},
    ]

    dependencies = [
        ("src", "IA_DET_FT_1", 150),
        ("IA_DET_FT_1", "MA_region_1", 80),
        ("IA_DET_FT_1", "UNI_p1_1", 160),
        ("MA_region_1", "RE_p1_1", 80),
        ("UNI_p1_1", "RE_p1_1", 160),
        ("RE_p1_1", "EC_TC_p1_1", 100),
        ("EC_TC_p1_1", "dst", 100),
    ]

    sfc_dict = {
        "name": "sfc_dag_p1_1",
        "vnf_list": vnf_list,
        "dependencies": dependencies,
        "bandwidth": 100,
        "closer_router": 1,
        "src_node": 0,
        "dst_node": 10,
        "latency": 100,
        "duration": 60,
    }

    sfc = SFCGenerator(sfc_dict).generate()
    alg = DAGGreedyAlgorithm()
    alg.install_substrate_network(graph)
    alg.install_SFC(sfc)

    success = alg.start_algorithm()
    assert success is True
    assert alg.get_latency() > 0
    assert alg.sync_differential <= 5.0  # Atende ao SLA de sincronização

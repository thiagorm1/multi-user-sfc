from itertools import islice

import networkx as nx


def k_shortest_paths(G, source, target, k, weight=None):
    """Calculate k shortest paths
    :param G: networkx topology
           source: source node
           target: target node
           k: the number of shortest paths
           weight: weight for calculation of shortest path
    :return: a list of k shortest paths
    """
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

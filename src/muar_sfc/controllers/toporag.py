"""Execution-local structural trace retrieval (a simulator assumption, not TopoRAG replication)."""

import copy
import hashlib
import json
import re
from collections import defaultdict

import networkx as nx


def sf_type(vnf):
    """Semantic MUVR role, excluding player/session identity; generic VNF type otherwise."""
    name = str(vnf.id)
    shared = re.fullmatch(r"MA_region_MUVR_(.+)_\d+", name)
    private = re.fullmatch(r"(FI|BE_pred|BE_view)_p\d+_\d+", name)
    if shared:
        return shared[1] + ":shared"
    if private:
        return private[1] + ":private"
    return str(vnf.type)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def canonical_dag(dag):
    """Exact canonical adjacency via refinement/individualisation, pruning automorphisms."""
    def visit(colors):
        while True:
            labels = {
                n: (colors[n], tuple(sorted(colors[p] for p in dag.predecessors(n))),
                    tuple(sorted(colors[p] for p in dag.successors(n))))
                for n in dag
            }
            palette = {label: i for i, label in enumerate(sorted(set(labels.values())))}
            refined = {n: palette[label] for n, label in labels.items()}
            if len(set(refined.values())) == len(set(colors.values())):
                colors = refined
                break
            colors = refined
        cells = defaultdict(list)
        for n, color in colors.items():
            cells[color].append(n)
        unresolved = [cell for _, cell in sorted(cells.items()) if len(cell) > 1]
        if not unresolved:
            order = sorted(dag, key=colors.get)
            return (
                tuple(dag.nodes[n]["sf"] for n in order),
                tuple(int(dag.has_edge(u, v)) for u in order for v in order),
            )
        representatives, results = [], []
        for n in unresolved[0]:
            marked = dag.copy()
            nx.set_node_attributes(marked, {v: (colors[v], v == n) for v in dag}, "color")
            if any(nx.is_isomorphic(
                marked, previous, node_match=nx.algorithms.isomorphism.categorical_node_match(
                    "color", None
                )
            ) for previous in representatives):
                continue
            representatives.append(marked)
            individualized = dict(colors)
            individualized[n] = max(colors.values()) + 1
            results.append(visit(individualized))
        return min(results)

    types = sorted({d["sf"] for _, d in dag.nodes(data=True)})
    return visit({n: types.index(d["sf"]) for n, d in dag.nodes(data=True)})


def structural_signature(sfc, graph):
    """Canonical typed DAG plus SF set and quartile of free physical bandwidth."""
    # NetworkX cached views are attributes too; materialise them on owned copies.
    graph = copy.deepcopy(graph)
    service_graph = copy.deepcopy(sfc.graph)
    dag = nx.DiGraph()
    for name in service_graph:
        dag.add_node(name, sf=sf_type(sfc.get_vnf_by_id(name)))
    dag.add_edges_from(service_graph.edges)
    capacity = sum(d.get("bandwidth_capacity", 0) for _, _, d in graph.edges(data=True))
    used = sum(
        d.get("bandwidth_used", 0) + d.get("bandwidth_reserved", 0)
        for _, _, d in graph.edges(data=True)
    )
    return {
        "dag": digest(canonical_dag(dag)),
        "sf_types": sorted({d["sf"] for _, d in dag.nodes(data=True)}),
        "free_band": min(3, max(0, int((1 - used / capacity) * 4))) if capacity else 0,
    }


class TopoRAGMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self._traces = []

    def __len__(self):
        return len(self._traces)

    def record(self, trace):
        # Own both inputs and outputs: callers cannot mutate memory through aliases.
        self._traces.append(copy.deepcopy(trace))

    def state_hash(self):
        return digest(self._traces)

    def retrieve(self, signature, k):
        if k < 1:
            raise ValueError("top_k must be positive")
        ranked = []
        for index, trace in enumerate(self._traces):
            other = trace["signature"]
            a, b = set(signature["sf_types"]), set(other["sf_types"])
            similarity = (
                len(a & b) / max(1, len(a | b))
                + float(signature["dag"] == other["dag"])
                + float(signature["free_band"] == other["free_band"])
            ) / 3
            ranked.append((other == signature, similarity, index, trace))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [
            {"similarity": score, "trace": copy.deepcopy(trace)}
            for _, score, _, trace in ranked[:k]
        ]


def node_prior(retrieved):
    """Unnormalised similarity sums by SF type/host, using only accepted traces."""
    scores = defaultdict(lambda: defaultdict(float))
    for item in retrieved:
        trace = item["trace"]
        if not trace["accepted"] or not trace["sla_met"]:
            continue
        for sf_type, hosts in trace["hosts_by_type"].items():
            for host in set(hosts):
                scores[str(sf_type)][host] += item["similarity"]
    return {sf_type: dict(hosts) for sf_type, hosts in scores.items()}

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Modernização do I/O com Pathlib
PATH_TRACE = Path("traces") / "sumoTraceVehicle.xml"


def generate_trace(substrate_network):
    # CORREÇÃO DO ERRO F821: Construindo edges_position dinamicamente
    edges_position = []
    for node in substrate_network.nodes():
        if node == 0:  # Ignora a nuvem (cloud)
            continue
        position = substrate_network.get_node_position(node)
        edges_position.append(np.asarray(position))

    edges_position = np.vstack(edges_position)

    voronoi_kdtree = cKDTree(edges_position)
    vehicles_dict = {}

    tree = ET.parse(PATH_TRACE)
    root = tree.getroot()
    timesteps = root.findall("timestep")

    for time_obj in timesteps:
        timestamp = time_obj.attrib["time"]
        if timestamp == "100.00":
            break

        vehicles_attr = {"speed": [], "id": [], "position": [], "region": []}

        # CORREÇÃO: getchildren() não existe mais no Python moderno.
        # Iteramos diretamente sobre o elemento XML.
        for vehicle in time_obj:
            vehicle_id = int(vehicle.attrib["id"])
            if vehicle_id > 400:
                break
            vehicles_attr["id"].append(vehicle_id)
            vehicles_attr["speed"].append(float(vehicle.attrib["speed"]))
            vehicles_attr["position"].append(
                [float(vehicle.attrib["x"]), float(vehicle.attrib["y"])]
            )

        # Proteção: só faz a query se a lista de posições não estiver vazia
        if vehicles_attr["position"]:
            test_point_dist, test_point_regions = voronoi_kdtree.query(vehicles_attr["position"])
            vehicles_attr["region"] = list(test_point_regions)

        vehicles_dict[timestamp] = vehicles_attr

    if "99.00" in vehicles_dict:
        print(vehicles_dict["99.00"]["region"])

    return vehicles_dict

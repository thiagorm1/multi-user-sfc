# import logging
# import xml.etree.ElementTree as ET
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.spatial import Voronoi, cKDTree, voronoi_plot_2d

# # Atualizando importação para a nova arquitetura src-layout
# from muar_sfc.topology.predefined.sample_topology import generate_substrate_network

# # Modernização Orientada a Objetos Multiplataforma (pathlib)
# PATH_TRACE = Path("traces") / "sumoTraceVehicle.xml"

# # Configuração de Logs Estruturados
# logger = logging.getLogger(__name__)


# def generate_trace(edges_position: np.ndarray) -> dict:
#     """
#     Processa o XML do SUMO Trace e extrai posições e regiões baseadas em Voronoi.
#     """
#     voronoi_kdtree = cKDTree(edges_position)
#     vehicles_dict = {}

#     try:
#         tree = ET.parse(PATH_TRACE)
#     except FileNotFoundError:
#         logger.error(f"Arquivo de trace não encontrado: {PATH_TRACE.resolve()}")
#         raise

#     root = tree.getroot()
#     timesteps = root.findall("timestep")

#     for time_obj in timesteps:
#         timestamp = time_obj.attrib["time"]
#         if timestamp == "100.00":
#             break

#         vehicles = list(time_obj)
#         vehicles_attr = {"speed": [], "id": [], "position": [], "region": []}

#         for vehicle in vehicles:
#             vehicle_id = int(vehicle.attrib["id"])
#             if vehicle_id > 400:
#                 break
#             vehicles_attr["id"].append(vehicle_id)
#             vehicles_attr["speed"].append(float(vehicle.attrib["speed"]))
#             vehicles_attr["position"].append(
#                 [float(vehicle.attrib["x"]), float(vehicle.attrib["y"])]
#             )

#         # Proteção: só consulta a árvore se existirem veículos mapeados
#         if vehicles_attr["position"]:
#             test_point_dist, test_point_regions = voronoi_kdtree.query(vehicles_attr["position"])
#             vehicles_attr["region"] = list(test_point_regions)

#         vehicles_dict[timestamp] = vehicles_attr

#     if "99.00" in vehicles_dict:
#         logger.info(f"Regiões processadas no timestep 99.00: {vehicles_dict['99.00']['region']}")

#     return vehicles_dict


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     substrate_network = generate_substrate_network()
#     edges_position_list = []

#     for node in substrate_network.nodes():
#         # skip cloud
#         if node == 0:
#             continue
#         position = substrate_network.get_node_position(node)
#         arr = np.asarray(position)
#         edges_position_list.append(arr)

#     edges_position_array = np.vstack(edges_position_list)
#     vor = Voronoi(edges_position_array)

#     fig = voronoi_plot_2d(
#         vor, show_vertices=False, line_colors="orange", line_width=2, line_alpha=0.6, point_size=2
#     )
#     plt.show()

#     # Executa a geração de rotas
#     generate_trace(edges_position_array)

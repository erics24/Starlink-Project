import json
import pickle
import numpy as np

import topology

"""
Simulation Configurations
"""
# Use following setup if not loading from saved topology file
SATELLITE_ALTITUDE = 1000 * 1000
INCLINATIONS = [30, 50, 70, -30, -50, -70]
N_PLANES = 10
N_NODES_PER_PLANE = 30
GATEWAY_FILEPATH = "gateways.json"
INITIAL_AREA_PLANE_WIDTH = 1                                  # 1, 2
INITIAL_AREA_SATELLITE_CNT_PER_PLANE = 5                      # 3, 5, 6, 10
INITIAL_AREA_STATIC_RATIO, INITIAL_AREA_DYNAMIC_RATIO = 1, 1  # 1:0, 3:1, 1:1, 1:3

USE_SAVED_TOPOLOGY_FILE = True
SAVE_TOPOLOGY_AFTER_FINISH = False
TOPOLOGY_FILEPATH = "saved_topology_{}_{}.pkl".format(INITIAL_AREA_STATIC_RATIO, INITIAL_AREA_DYNAMIC_RATIO)

SIMULATION_TIME_STEP = 1                                      # in seconds
SIMULATION_TIME_LENGTH = 600                                  # in seconds  # a full period = 6298


def initialize(topology_file=None):
    global SATELLITE_ALTITUDE, INCLINATIONS, N_PLANES, N_NODES_PER_PLANE, GATEWAY_FILEPATH
    if topology_file is not None:
        with open(topology_file, "rb") as topology_in_file:
            saved_topology = pickle.load(topology_in_file)
        SATELLITE_ALTITUDE = saved_topology["satellite_altitude"]
        INCLINATIONS = saved_topology["inclinations"]
        N_PLANES = saved_topology["n_planes"]
        N_NODES_PER_PLANE = saved_topology["n_nodes_per_plane"]
        GATEWAY_FILEPATH = saved_topology["gateway_filepath"]

    # Add constellations, planes, and satellites
    for ci, incl in enumerate(INCLINATIONS):
        topology.Constellation(num_planes=N_PLANES, num_nodes_per_plane=N_NODES_PER_PLANE,
                               inclination=incl, altitude=SATELLITE_ALTITUDE,
                               plane_offset_shift=360. / N_PLANES / len(INCLINATIONS) * ci)
    # Add gateways
    # Retrieved from https://starlink.sx/gateways.json
    with open(GATEWAY_FILEPATH, "r") as gateway_file:
        gateway_data = json.load(gateway_file)
        for gateway in gateway_data:
            topology.Gateway(latitude=gateway['lat'], longitude=gateway['lng'], altitude=gateway['minElevation'])
    # Initialize node positions from scratch at time 0
    topology.NODE_POSITION_MATRIX = np.zeros((len(topology.SATELLITE_LIST) + len(topology.GATEWAY_LIST), 3),
                                             dtype=np.float64)
    # Initialize links and areas
    if topology_file is None:
        topology.ISL_LINK_MATRIX = np.full((len(topology.SATELLITE_LIST), len(topology.SATELLITE_LIST)),
                                           np.inf, dtype=np.float64)
        topology.STG_LINK_MATRIX = np.full((len(topology.SATELLITE_LIST), len(topology.GATEWAY_LIST)),
                                           np.inf, dtype=np.float64)
        n_areas = topology.initialize_area(n_planes_per_area=INITIAL_AREA_PLANE_WIDTH,
                                           n_nodes_per_plane_per_area=INITIAL_AREA_SATELLITE_CNT_PER_PLANE,
                                           static_area_ratio=INITIAL_AREA_STATIC_RATIO,
                                           dynamic_area_ratio=INITIAL_AREA_DYNAMIC_RATIO)
    else:
        topology.ISL_LINK_MATRIX = saved_topology["isl_link_matrix"]
        topology.STG_LINK_MATRIX = saved_topology["stg_link_matrix"]
        topology.SATELLITE_INITIAL_AREA = saved_topology["sat_init_area"]
        topology.NODE_AREA_ASSIGNMENT = saved_topology["node_area_matrix"]
        topology.AREA_CONNECTIVITY_MATRIX = saved_topology["area_matrix"]
        topology.SHORTEST_AREA_PATH_DIST_MATRIX = saved_topology["spf_area_dist_matrix"]
        topology.SHORTEST_AREA_PATH_PREDECESSOR_MATRIX = saved_topology["spf_area_path_matrix"]
        topology.SHORTEST_NODE_PATH_PREDECESSOR_MATRIX = saved_topology["spf_node_path_matrix"]
        n_areas = topology.AREA_CONNECTIVITY_MATRIX.shape[0]
    print("# Total Areas: {}".format(n_areas))


def run_simulation():
    initialize(topology_file=TOPOLOGY_FILEPATH if USE_SAVED_TOPOLOGY_FILE else None)
    print("# Total Satellites: {}".format(len(topology.SATELLITE_LIST)))
    print("# Total Gateways: {}".format(len(topology.GATEWAY_LIST)))
    print("Period: {}".format(topology.SATELLITE_LIST[0].plane.period))
    sim_time = 0
    while sim_time < SIMULATION_TIME_LENGTH:
        print("Time: {}".format(sim_time))
        topology.update_pos(sim_time)
        isl_link_broken, isl_link_established, _, _ = topology.update_links()
        _, _, = topology.update_area(isl_link_broken, isl_link_established)

        # Observation period = ?
        # [0] Comparison:
        # [0.1] All satellites have an initial static area assignment (1:0)
        # [0.2] Some satellites have an initial static area assignment, some are dynamically assigned (3:1, 1:1, 1:3)
        # Evaluation:
        # [1] Stability
        # [1.1] Averages: link count, link/area graph connectivity and diameter; distribution of node/area degrees TODO
        # [1.2] Averages: link/area topology change rate
        # [2] Performance
        # Traffic patterns:
        # all satellites to nearest gateways (all-to-all, then pick nearest) TODO
        # all satellites to all satellites TODO
        # [2.1] distribution of #hops (nodes or areas) TODO
        # [2.2] distribution of latency, compare with optimal (no area assignment, global optimal path) TODO
        # Finally, record all data, dump to files, and plot figures TODO

        sim_time += SIMULATION_TIME_STEP

    # Save links and area assignments to file
    if SAVE_TOPOLOGY_AFTER_FINISH:
        topology_dict = {"satellite_altitude": SATELLITE_ALTITUDE,
                         "inclinations": INCLINATIONS,
                         "n_planes": N_PLANES,
                         "n_nodes_per_plane": N_NODES_PER_PLANE,
                         "gateway_filepath": GATEWAY_FILEPATH,
                         "isl_link_matrix": topology.ISL_LINK_MATRIX,
                         "stg_link_matrix": topology.STG_LINK_MATRIX,
                         "sat_init_area": topology.SATELLITE_INITIAL_AREA,
                         "node_area_matrix": topology.NODE_AREA_ASSIGNMENT,
                         "area_matrix": topology.AREA_CONNECTIVITY_MATRIX,
                         "spf_area_dist_matrix": topology.SHORTEST_AREA_PATH_DIST_MATRIX,
                         "spf_area_path_matrix": topology.SHORTEST_AREA_PATH_PREDECESSOR_MATRIX,
                         "spf_node_path_matrix": topology.SHORTEST_NODE_PATH_PREDECESSOR_MATRIX}
        with open(TOPOLOGY_FILEPATH, "wb") as topology_out_file:
            pickle.dump(topology_dict, topology_out_file)


if __name__ == "__main__":
    run_simulation()

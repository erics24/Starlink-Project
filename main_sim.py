import json
import pickle
import numpy as np

import topology

"""
Configurable Parameters
"""
SIMULATION_TIME_STEP = 1  # in seconds
SIMULATION_TIME_LENGTH = 6298  # in seconds  # a full period
USE_SAVED_TOPOLOGY_FILE = True
SAVE_TOPOLOGY_AFTER_FINISH = False
TOPOLOGY_FILEPATH = "saved_topology.pkl"

SATELLITE_ALTITUDE = 1000 * 1000
INCLINATIONS = [30, 50, 70, -30, -50, -70]
N_PLANES = 10
N_NODES_PER_PLANE = 30
GATEWAY_FILEPATH = "gateways.json"


def initialize(topology_file=None):
    global SATELLITE_ALTITUDE, INCLINATIONS, N_PLANES, N_NODES_PER_PLANE, GATEWAY_FILEPATH
    if topology_file is not None:
        # Only load links and area assignments, initialize node positions from scratch at time 0
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

    if topology_file is None:
        topology.initialize_area()
    else:
        topology.ISL_LINK_MATRIX = saved_topology["isl_link_matrix"]
        topology.STG_LINK_MATRIX = saved_topology["stg_link_matrix"]
        # TODO: Also load area assignment and area connectivity matrix
        # TODO: Also load predecessor_matrix and shortest_dist_matrix
        pass


def run_simulation():
    initialize(topology_file=TOPOLOGY_FILEPATH if USE_SAVED_TOPOLOGY_FILE else None)
    print("# Total Satellites: {}".format(len(topology.SATELLITE_LIST)))
    print("# Total Gateways: {}".format(len(topology.GATEWAY_LIST)))
    print("Period: {}".format(topology.SATELLITE_LIST[0].plane.period))
    sim_time = 0
    while sim_time < SIMULATION_TIME_LENGTH:
        print("Time: {}".format(sim_time))
        topology.update_pos(sim_time)
        print("#{}\t{}".format(topology.SATELLITE_LIST[0].id,
                               topology.NODE_POSITION_MATRIX[topology.SATELLITE_LIST[0].id, :]))
        isl_link_broken, isl_link_established, stg_link_broken, stg_link_established = topology.update_links()
        topology.update_area(isl_link_broken, isl_link_established, stg_link_broken, stg_link_established)

        # print(topology.ISL_LINK_MATRIX)
        # print(topology.STG_LINK_MATRIX)
        # The closest satellite to each gateway
        # print(np.argmin(topology.STG_LINK_MATRIX, axis=0)[0:165])
        # print(np.min(topology.STG_LINK_MATRIX, axis=0)[0:165])
        sim_time += SIMULATION_TIME_STEP

    # Calculate average #links at stable state
    # Measure link change rate at stable state
    # Measure area change rate at stable state

    # Save links and area assignments to file
    if SAVE_TOPOLOGY_AFTER_FINISH:
        topology_dict = {"satellite_altitude": SATELLITE_ALTITUDE,
                         "inclinations": INCLINATIONS,
                         "n_planes": N_PLANES,
                         "n_nodes_per_plane": N_NODES_PER_PLANE,
                         "gateway_filepath": GATEWAY_FILEPATH,
                         "isl_link_matrix": topology.ISL_LINK_MATRIX,
                         "stg_link_matrix": topology.STG_LINK_MATRIX}
        with open(TOPOLOGY_FILEPATH, "wb") as topology_out_file:
            pickle.dump(topology_dict, topology_out_file)


if __name__ == "__main__":
    run_simulation()

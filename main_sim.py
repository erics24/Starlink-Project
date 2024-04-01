import json
import numpy as np

import topology

"""
Configurable Parameters
"""
SIMULATION_TIME_STEP = 10  # in seconds
SIMULATION_TIME_LENGTH = 100  # in seconds


def initialize():
    satellite_altitude = 1000 * 1000
    inclinations = [30, 45, 60, -30, -45, -60]
    # Add constellations, planes, and satellites
    for ci, incl in enumerate(inclinations):
        n_planes = 10
        n_nodes_per_plane = 50
        topology.Constellation(num_planes=n_planes, num_nodes_per_plane=n_nodes_per_plane,
                               inclination=incl, altitude=satellite_altitude,
                               plane_offset_shift=360. / n_planes / len(inclinations) * ci)
    # Add gateways
    # Retrieved from https://starlink.sx/gateways.json
    with open("gateways.json", "r") as gateway_file:
        gateway_data = json.load(gateway_file)
        for gateway in gateway_data:
            topology.Gateway(latitude=gateway['lat'], longitude=gateway['lng'], altitude=gateway['minElevation'])

    topology.initialize_area()


def run_simulation():
    initialize()
    print("# Total Satellites: {}".format(len(topology.SATELLITE_LIST)))
    print("# Total Gateways: {}".format(len(topology.GATEWAY_LIST)))
    print("Period: {}".format(topology.SATELLITE_LIST[0].plane.period))
    sim_time = 0
    while sim_time < SIMULATION_TIME_LENGTH:
        topology.update_pos(sim_time)
        topology.update_links()
        print("Time: {}".format(sim_time))
        print("#{}\t{}".format(topology.SATELLITE_LIST[0].id,
                               topology.NODE_POSITION_MATRIX[topology.SATELLITE_LIST[0].id, :]))
        print(topology.ISL_LINK_MATRIX)
        print(topology.STG_LINK_MATRIX)
        # The closest satellite to each gateway
        print(np.argmin(topology.STG_LINK_MATRIX, axis=0)[0:165])
        sim_time += SIMULATION_TIME_STEP


if __name__ == "__main__":
    run_simulation()

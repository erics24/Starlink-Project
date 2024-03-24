from topology import *

"""
Configurable Parameters
"""
SIMULATION_TIME_STEP = 10  # in seconds
SIMULATION_TIME_LENGTH = 1000  # in seconds


def initialize():
    c0 = Constellation(num_planes=10, num_nodes_per_plane=50, inclination=60, altitude=1000 * 1000)


def run_simulation():
    initialize()
    sim_time = 0
    print(SATELLITE_LIST[0].plane.period)
    while sim_time < SIMULATION_TIME_LENGTH:
        update_pos(sim_time)
        print("Time: {}".format(sim_time))
        print("#{}\t{}".format(SATELLITE_LIST[0].id, SATELLITE_LIST[0].pos))
        sim_time += SIMULATION_TIME_STEP


if __name__ == "__main__":
    run_simulation()

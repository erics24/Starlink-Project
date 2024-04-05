import numpy as np

"""
Constants
"""
EARTH_RADIUS = 6371000
EARTH_ROTATION_AXIS = (0, 0, 1)
STD_GRAVITATIONAL_PARAMETER_EARTH = 3.986004418e14
SECONDS_PER_DAY = 86400
EARTH_ROTATION_PER_SECONDS = 360. / SECONDS_PER_DAY
SPEED_OF_LIGHT = 299792458


def lat_lon_to_cartesian(latitude, longitude, altitude):
    radius = EARTH_RADIUS + altitude
    latitude = np.radians(latitude)
    longitude = np.radians(longitude)
    return np.array([radius * np.cos(latitude) * np.cos(longitude),
                     radius * np.cos(latitude) * np.sin(longitude),
                     radius * np.sin(latitude)],
                    dtype=np.float64)


def calculate_orbit_period(semi_major_axis):
    return 2. * np.pi * np.sqrt(np.power(float(semi_major_axis), 3) / STD_GRAVITATIONAL_PARAMETER_EARTH)


def gen_node_init_time_offset(n_planes, n_nodes_per_plane):
    phase_offset_inc = 1. / n_nodes_per_plane / n_planes
    phase_offsets = []
    toggle = False
    # Put offsets in this order [...8,6,4,2,0,1,3,5,7,9...]
    # so that offsets in adjacent planes are close to each other
    for i in range(n_planes):
        if toggle:
            phase_offsets.append(phase_offset_inc * i)
        else:
            phase_offsets.insert(0, phase_offset_inc * i)
        toggle = not toggle
    return phase_offsets


def get_earth_rotation_matrix(degree):
    theta = np.radians(degree)
    axis = np.asarray(EARTH_ROTATION_AXIS)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.)
    b, c, d = -axis * np.sin(theta / 2.)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

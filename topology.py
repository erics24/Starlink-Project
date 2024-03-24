# Reference: https://github.com/Ben-Kempton/SILLEO-SCNS/blob/master/source/constellation.py

# Qifei: https://github.com/abx13/Hypatia seems closer to what we want to achieve.
#        They also use Python for setting up satellite topology, but their work is more
#        complicated with packet-level simulations using ns-3.
#        Currently I'm writing code based on the SILLEO-SCNS one.

import numpy as np
from PyAstronomy import pyasl

"""
Constants
"""
EARTH_RADIUS = 6371000
EARTH_ROTATION_AXIS = (0, 0, 1)
STD_GRAVITATIONAL_PARAMETER_EARTH = 3.986004418e14
SECONDS_PER_DAY = 86400
EARTH_ROTATION_PER_SECONDS = 360. / SECONDS_PER_DAY

"""
Configurable Parameters
"""
GROUND_STATION_DEFAULT_ALTITUDE = 100
MAX_ISL_DISTANCE = 5000 * 1000
MAX_STG_DISTANCE = 1000 * 1000

# TODO: Make numpy arrays of satellite/gnd coordinates for faster distance computation?
SATELLITE_LIST = []
GROUND_STATION_LIST = []


class Constellation:
    constellation_id = 0

    def __init__(self,
                 num_planes=1,
                 num_nodes_per_plane=4,
                 inclination=45.,
                 altitude=1000 * 1000):
        self.id = Constellation.constellation_id
        Constellation.constellation_id += 1
        self.num_nodes_per_plane = num_nodes_per_plane
        self.total_nodes = num_planes * num_nodes_per_plane
        self.semi_major_axis = EARTH_RADIUS + altitude
        self.inclination = inclination

        self.planes = []
        for plane_i in range(num_planes):
            plane_offset = 360. / num_planes * plane_i
            self.planes.append(Plane(constellation=self, plane_offset=plane_offset))
        self.add_inter_plane_links()

    def add_inter_plane_links(self):
        pass


class Plane:
    plane_id = 0

    def __init__(self, constellation, plane_offset=0.):
        self.id = Plane.plane_id
        Plane.plane_id += 1
        self.constellation = constellation
        num_nodes = constellation.num_nodes_per_plane
        self.semi_major_axis = constellation.semi_major_axis
        self.inclination = constellation.inclination
        self.period = calculate_orbit_period(self.semi_major_axis)
        self.plane_solver = pyasl.KeplerEllipse(a=self.semi_major_axis,
                                                per=self.period,
                                                e=0.,  # eccentricity of 0 for leo constillations
                                                Omega=plane_offset,  # right ascension (longitude)
                                                w=0.0,  # initial time offset / mean anamoly
                                                i=self.inclination)
        self.satellites = []
        for node_i in range(num_nodes):
            time_offset = self.period / num_nodes * node_i
            # TODO: Shift the starting offset of each plane by a bit so they won't all start from the equator?
            self.satellites.append(Satellite(plane=self, time_offset=time_offset))
        self.add_intra_plane_links()

    def add_intra_plane_links(self):
        pass


class Satellite:
    satellite_id = 0

    def __init__(self, plane, time_offset=0.):
        self.id = Satellite.satellite_id
        Satellite.satellite_id += 1
        self.plane = plane
        self.init_time_offset = time_offset
        pos = plane.plane_solver.xyzPos(time_offset)
        self.pos = np.array(pos, dtype=np.float32)  # (x, y, z)
        # Fixed links
        self.intra_plane_links = []  # Adjacency list
        self.inter_plane_links = []
        # Dynamic links
        # We search for the nearest satellite only in the next plane id
        # from each constellation
        # when trying to find the next intra constellation link
        # during the simulation
        self.inter_constellation_links = []
        self.satellite_to_gnd_links = []

        SATELLITE_LIST.append(self)


class GroundStation:
    gnd_id = -1  # use negative integers for ground station IDs

    def __init__(self, latitude, longitude, altitude=GROUND_STATION_DEFAULT_ALTITUDE):
        self.id = GroundStation.gnd_id
        GroundStation.gnd_id -= 1
        self.init_pos = lat_lon_to_cartesian(latitude, longitude, altitude)
        self.pos = self.init_pos.copy()
        # Fixed links
        self.gnd_to_gnd_links = []
        # Dynamic links
        self.gnd_to_satellite_links = []

        GROUND_STATION_LIST.append(self)


def add_gnd_to_gnd_links():
    pass


def lat_lon_to_cartesian(latitude, longitude, altitude):
    radius = EARTH_RADIUS + altitude
    latitude = np.radians(latitude)
    longitude = np.radians(longitude)
    return np.array([radius * np.cos(latitude) * np.cos(longitude),
                     radius * np.cos(latitude) * np.sin(longitude),
                     radius * np.sin(latitude)],
                    dtype=np.float32)


def calculate_orbit_period(semi_major_axis):
    return 2. * np.pi * np.sqrt(np.power(float(semi_major_axis), 3) / STD_GRAVITATIONAL_PARAMETER_EARTH)


def update_pos(cur_time=0.):
    # Update all satellite positions
    for satellite in SATELLITE_LIST:
        satellite_pos = satellite.plane.plane_solver.xyzPos(satellite.init_time_offset + cur_time)
        satellite.pos = np.array(satellite_pos, dtype=np.float32)
    # Update all ground station positions
    rotation_degree = EARTH_ROTATION_PER_SECONDS * np.fmod(cur_time, SECONDS_PER_DAY)
    rotation_matrix = get_rotation_matrix(rotation_degree)
    for gnd in GROUND_STATION_LIST:
        gnd.pos = np.dot(rotation_matrix, gnd.init_pos)


def get_rotation_matrix(degree):
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


def update_links(cur_time=0.):
    # TODO: Check existing inter-constellation/satellite-to-gnd links
    # TODO: Search for new available links
    #       TODO: Implement 3D Nearest Neighbor Search
    pass

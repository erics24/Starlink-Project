# Reference: https://github.com/Ben-Kempton/SILLEO-SCNS/blob/master/source/constellation.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
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
GATEWAY_DEFAULT_ELEVATION = 25
MAX_ISL_DISTANCE = 3000 * 1000
MAX_STG_DISTANCE = 1500 * 1000

CONSTELLATION_LIST = []
PLANE_LIST = []
SATELLITE_LIST = []
GATEWAY_LIST = []
NODE_POSITION_MATRIX = None
ISL_LINK_MATRIX = None
STG_LINK_MATRIX = None
AREA_CONNECTIVITY_DICT = None


class Constellation:
    cons_id = 0

    def __init__(self,
                 num_planes=1,
                 num_nodes_per_plane=4,
                 inclination=45.,
                 altitude=1000 * 1000,
                 plane_offset_shift=0.):
        self.id = Constellation.cons_id
        Constellation.cons_id += 1
        self.num_nodes_per_plane = num_nodes_per_plane
        self.total_nodes = num_planes * num_nodes_per_plane
        self.semi_major_axis = EARTH_RADIUS + altitude
        self.inclination = inclination

        self.planes = []
        plane_node_init_time_offsets = gen_node_init_time_offset(n_planes=num_planes,
                                                                 n_nodes_per_plane=num_nodes_per_plane)
        for plane_i in range(num_planes):
            # Shift the right ascension of each plane by a bit so planes from different constellations
            # won't intersect at the equator
            plane_offset = 360. / num_planes * plane_i + plane_offset_shift
            # Shift the initial position (time offset) of each satellite on the plane by a bit so
            # the first satellite from different planes won't all start from the equator
            node_init_time_offset = plane_node_init_time_offsets[plane_i]
            self.planes.append(Plane(constellation=self, plane_offset=plane_offset,
                                     node_init_time_offset=node_init_time_offset))
        CONSTELLATION_LIST.append(self)


class Plane:
    plane_id = 0

    def __init__(self, constellation, plane_offset=0., node_init_time_offset=0.):
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
            # Shift the initial position (time offset) of each satellite on the plane by a bit so
            # the first satellite from different planes won't all start from the equator
            time_offset = self.period * (node_i / num_nodes + node_init_time_offset)
            self.satellites.append(Satellite(plane=self, time_offset=time_offset))
        PLANE_LIST.append(self)


class Satellite:
    sat_id = 0

    def __init__(self, plane, time_offset=0.):
        self.id = Satellite.sat_id
        Satellite.sat_id += 1
        self.plane = plane
        self.init_time_offset = time_offset
        self.init_area = None
        self.area = None
        SATELLITE_LIST.append(self)


class Gateway:
    gate_id = 0

    def __init__(self, latitude, longitude, altitude=GATEWAY_DEFAULT_ELEVATION, init_pos=None):
        self.id = Gateway.gate_id
        Gateway.gate_id += 1
        if init_pos is not None:
            self.init_pos = init_pos.copy()
        else:
            self.init_pos = lat_lon_to_cartesian(latitude, longitude, altitude)
        self.area = None
        GATEWAY_LIST.append(self)


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


def update_pos(cur_time=0.):
    global NODE_POSITION_MATRIX
    if NODE_POSITION_MATRIX is None:
        NODE_POSITION_MATRIX = np.zeros((len(SATELLITE_LIST) + len(GATEWAY_LIST), 3), dtype=np.float32)
    # Update all satellite positions
    for satellite in SATELLITE_LIST:
        satellite_pos = satellite.plane.plane_solver.xyzPos(satellite.init_time_offset + cur_time)
        NODE_POSITION_MATRIX[satellite.id, :] = satellite_pos
    # Update all gateway positions
    rotation_degree = EARTH_ROTATION_PER_SECONDS * np.fmod(cur_time, SECONDS_PER_DAY)
    rotation_matrix = get_rotation_matrix(rotation_degree)
    for gate in GATEWAY_LIST:
        NODE_POSITION_MATRIX[gate.id + len(SATELLITE_LIST), :] = np.dot(rotation_matrix, gate.init_pos)


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


def initialize_area():
    pass


def update_links():
    global ISL_LINK_MATRIX
    if ISL_LINK_MATRIX is None:
        ISL_LINK_MATRIX = np.full((len(SATELLITE_LIST), len(SATELLITE_LIST)), np.inf, dtype=np.float32)
    global STG_LINK_MATRIX
    if STG_LINK_MATRIX is None:
        STG_LINK_MATRIX = np.full((len(SATELLITE_LIST), len(GATEWAY_LIST)), np.inf, dtype=np.float32)
    new_links = squareform(pdist(NODE_POSITION_MATRIX).astype(np.float32))

    # Filter links within reachable distances
    new_isl_links = new_links[:len(SATELLITE_LIST), :len(SATELLITE_LIST)]
    new_isl_links[new_isl_links > MAX_ISL_DISTANCE] = np.inf
    new_isl_links[new_isl_links == 0] = np.inf
    new_stg_links = new_links[:len(SATELLITE_LIST), len(SATELLITE_LIST):]
    new_stg_links[new_stg_links > MAX_STG_DISTANCE] = np.inf
    new_stg_links[new_stg_links == 0] = np.inf

    isl_link_mask = np.full((len(SATELLITE_LIST), len(SATELLITE_LIST)), np.inf, dtype=np.float32)
    stg_link_mask = np.full((len(SATELLITE_LIST), len(GATEWAY_LIST)), np.inf, dtype=np.float32)
    for sat in SATELLITE_LIST:
        # Always and only allow links between adjacent satellites on the same plane
        plane_sat_start_id = sat.plane.satellites[0].id
        plane_sat_ind = sat.id - plane_sat_start_id
        plane_sat_cnt = sat.plane.constellation.num_nodes_per_plane
        isl_link_mask[sat.id, plane_sat_start_id + (plane_sat_ind - 1) % plane_sat_cnt] = 1.
        isl_link_mask[sat.id, plane_sat_start_id + (plane_sat_ind + 1) % plane_sat_cnt] = 1.

        # Allow links between a satellite and another on adjacent planes
        # (use existing nearest or pick new nearest)
        cons_plane_start_id = sat.plane.constellation.planes[0].id
        cons_plane_ind = sat.plane.id - cons_plane_start_id
        cons_plane_cnt = len(sat.plane.constellation.planes)
        adj_planes = (PLANE_LIST[(cons_plane_ind - 1) % cons_plane_cnt + cons_plane_start_id],
                      PLANE_LIST[(cons_plane_ind + 1) % cons_plane_cnt + cons_plane_start_id])
        for adj_plane in adj_planes:
            adj_plane_sat_start_id = adj_plane.satellites[0].id
            existing_sat_id = np.argmin(ISL_LINK_MATRIX[
                                        sat.id, adj_plane_sat_start_id:adj_plane_sat_start_id + plane_sat_cnt])
            if (ISL_LINK_MATRIX[sat.id, adj_plane_sat_start_id + existing_sat_id] == np.inf or
                    new_isl_links[sat.id, adj_plane_sat_start_id + existing_sat_id] == np.inf):
                # Find new nearest satellite on the adjacent plane if not existed or broken
                nearest_sat_id = np.argmin(new_isl_links[
                                           sat.id, adj_plane_sat_start_id:adj_plane_sat_start_id + plane_sat_cnt])
                if new_isl_links[sat.id, adj_plane_sat_start_id + nearest_sat_id] != np.inf:
                    isl_link_mask[sat.id, adj_plane_sat_start_id + nearest_sat_id] = 1.
            else:
                # Use existing nearest satellite link
                isl_link_mask[sat.id, adj_plane_sat_start_id + existing_sat_id] = 1.

        # Allow links between a satellite and another on a different constellation
        # (use existing nearest or pick new nearest)
        for other_cons in CONSTELLATION_LIST:
            if other_cons.id != sat.plane.constellation.id:
                cons_sat_start_id = other_cons.planes[0].satellites[0].id
                cons_sat_end_id = other_cons.planes[-1].satellites[-1].id
                existing_sat_id = np.argmin(ISL_LINK_MATRIX[sat.id, cons_sat_start_id:cons_sat_end_id + 1])
                if (ISL_LINK_MATRIX[sat.id, cons_sat_start_id + existing_sat_id] == np.inf or
                        new_isl_links[sat.id, cons_sat_start_id + existing_sat_id] == np.inf):
                    # Find new nearest satellite in the constellation if not existed or broken
                    nearest_sat_id = np.argmin(new_isl_links[sat.id, cons_sat_start_id:cons_sat_end_id + 1])
                    if new_isl_links[sat.id, cons_sat_start_id + nearest_sat_id] != np.inf:
                        isl_link_mask[sat.id, cons_sat_start_id + nearest_sat_id] = 1.
                else:
                    # Use existing nearest satellite link
                    isl_link_mask[sat.id, cons_sat_start_id + existing_sat_id] = 1.

        # Allow links from any satellite to a gateway
        # (use existing or pick new nearest)
        existing_gate_id = np.argmin(STG_LINK_MATRIX[sat.id, :])
        if STG_LINK_MATRIX[sat.id, existing_gate_id] == np.inf or new_stg_links[sat.id, existing_gate_id] == np.inf:
            nearest_gate_id = np.argmin(new_stg_links[sat.id, :])
            if new_stg_links[sat.id, nearest_gate_id] != np.inf:
                stg_link_mask[sat.id, nearest_gate_id] = 1.
        else:
            stg_link_mask[sat.id, existing_gate_id] = 1.

    # Ensure that any gateway is connected to at least one satellite if possible
    # (use existing or pick new nearest)
    for gate in GATEWAY_LIST:
        if np.sum(stg_link_mask[:, gate.id] != np.inf) < 1:
            existing_sat_id = np.argmin(STG_LINK_MATRIX[:, gate.id])
            if STG_LINK_MATRIX[existing_sat_id, gate.id] == np.inf or new_stg_links[existing_sat_id, gate.id] == np.inf:
                nearest_sat_id = np.argmin(new_stg_links[:, gate.id])
                if new_stg_links[nearest_sat_id, gate.id] != np.inf:
                    stg_link_mask[nearest_sat_id, gate.id] = 1.
            else:
                stg_link_mask[existing_sat_id, gate.id] = 1.

    new_isl_links = new_isl_links * isl_link_mask
    new_stg_links = new_stg_links * stg_link_mask

    # Make ISL link matrix symmetric
    # i.e. making temporary one-way links two-way
    new_isl_links = np.minimum(new_isl_links, new_isl_links.T)

    # Check ISL connectivity
    sat_degree_row = np.sum(new_isl_links < np.inf, axis=1)
    sat_degree_col = np.sum(new_isl_links < np.inf, axis=0)
    assert np.array_equal(sat_degree_row, sat_degree_col)
    if np.min(sat_degree_row) < 1:
        print("Warning: At least one satellite is unreachable: #{}".format(np.argmin(sat_degree_row)))

    # Check STG link connectivity
    stg_sat_degree = np.sum(new_stg_links < np.inf, axis=1)
    stg_gate_degree = np.sum(new_stg_links < np.inf, axis=0)
    if np.min(stg_gate_degree) < 1:
        print("Warning: At least one gateway is unreachable: #{}".format(np.argmin(stg_gate_degree)))

    # Compare changes, measure change frequency
    isl_link_changes = (new_isl_links < np.inf).astype(np.int8) - (ISL_LINK_MATRIX < np.inf).astype(np.int8)
    isl_link_broken = np.nonzero(isl_link_changes == -1)
    isl_link_established = np.nonzero(isl_link_changes == 1)
    stg_link_changes = (new_stg_links < np.inf).astype(np.int8) - (STG_LINK_MATRIX < np.inf).astype(np.int8)
    stg_link_broken = np.nonzero(stg_link_changes == -1)
    stg_link_established = np.nonzero(stg_link_changes == 1)
    print("{} ISL links broken, {} ISL links established, {} in total".format(len(isl_link_broken[0]) // 2,
                                                                              len(isl_link_established[0]) // 2,
                                                                              np.sum(sat_degree_row) // 2))
    print("{} STG links broken, {} STG links established, {} in total".format(len(stg_link_broken[0]),
                                                                              len(stg_link_established[0]),
                                                                              np.sum(stg_gate_degree)))
    ISL_LINK_MATRIX = new_isl_links
    STG_LINK_MATRIX = new_stg_links

    return isl_link_broken, isl_link_established, stg_link_broken, stg_link_established


def update_area(isl_link_broken, isl_link_established, stg_link_broken, stg_link_established):
    # TODO:
    # If matrices not exist, assign dynamic satellites to areas, mark ABRs for each area,
    # get area connectivity matrix (do not include gateway here),
    # then assign gateways to areas based on stg links, then run OSPF within each area

    # Otherwise, update area if changed
    # Update area level connectivity, etc.

    # Note: Never use a gateway as ABR when calculating routes -
    # only compare satellite ABRs and find the closest one locally
    pass

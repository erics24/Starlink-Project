# Reference: https://github.com/Ben-Kempton/SILLEO-SCNS/blob/master/source/constellation.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from PyAstronomy import pyasl

from utils import *

"""
Topology Configurations
"""
GATEWAY_DEFAULT_ELEVATION = 25
MAX_ISL_DISTANCE = 3000 * 1000
MAX_STG_DISTANCE = 1500 * 1000

CONSTELLATION_LIST = []
PLANE_LIST = []
SATELLITE_LIST = []
GATEWAY_LIST = []

global NODE_POSITION_MATRIX  # #node by 3, (x, y, z)
global ISL_LINK_MATRIX  # #sat by #sat, elements are distances, inf means no paths
global STG_LINK_MATRIX  # #sat by #gate, elements are distances, inf means no paths

global SATELLITE_INITIAL_AREA  # elements are area ids, -1 means no assignment
global NODE_AREA_ASSIGNMENT  # #node by #area, 0 and 1 elements

global AREA_CONNECTIVITY_MATRIX  # #area by #area, 0 and 1 elements
global SHORTEST_AREA_PATH_DIST_MATRIX  # #area by #area, 0 means no paths
global SHORTEST_AREA_PATH_PREDECESSOR_MATRIX  # same, elements are area ids, -9999 means no paths

global SHORTEST_NODE_PATH_DIST_MATRIX  # #node by #node, 0 means no paths
global SHORTEST_NODE_PATH_PREDECESSOR_MATRIX  # same, elements are sat ids or (gate ids + #sat), -9999 means no paths


class Constellation:
    cons_id = 0

    def __init__(self, num_planes, num_nodes_per_plane, inclination,
                 altitude=1000 * 1000, plane_offset_shift=0.):
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
        SATELLITE_LIST.append(self)


class Gateway:
    gate_id = 0

    def __init__(self, latitude, longitude, altitude=GATEWAY_DEFAULT_ELEVATION):
        self.id = Gateway.gate_id
        Gateway.gate_id += 1
        self.init_pos = lat_lon_to_cartesian(latitude, longitude, altitude)
        GATEWAY_LIST.append(self)


def initialize_area(n_planes_per_area, n_nodes_per_plane_per_area, static_area_ratio, dynamic_area_ratio, random=False):
    assert isinstance(static_area_ratio, int) and isinstance(dynamic_area_ratio, int)
    assert static_area_ratio > 0
    assert dynamic_area_ratio >= 0
    global SATELLITE_INITIAL_AREA, NODE_AREA_ASSIGNMENT, AREA_CONNECTIVITY_MATRIX
    global SHORTEST_AREA_PATH_DIST_MATRIX, SHORTEST_AREA_PATH_PREDECESSOR_MATRIX
    global SHORTEST_NODE_PATH_DIST_MATRIX, SHORTEST_NODE_PATH_PREDECESSOR_MATRIX
    SATELLITE_INITIAL_AREA = np.full((len(SATELLITE_LIST)), -1, dtype=np.int32)

    area_id = 0
    assignment_counter = static_area_ratio
    static_toggle = True
    for cons in CONSTELLATION_LIST:
        n_planes = len(cons.planes)
        n_nodes_per_plane = cons.num_nodes_per_plane
        assert n_planes % n_planes_per_area == 0
        assert n_nodes_per_plane % n_nodes_per_plane_per_area == 0
        # plane 0: sat 0-4 -> plane 1: sat 0-4 -> ...
        cur_sat_ind = 0
        while cur_sat_ind < n_nodes_per_plane:
            cur_plane_ind = 0
            while cur_plane_ind < n_planes:
                for plane_ind in range(cur_plane_ind, cur_plane_ind + n_planes_per_area):
                    assignment_counter -= 1
                    if static_toggle:
                        SATELLITE_INITIAL_AREA[
                        cons.planes[plane_ind].satellites[cur_sat_ind].id:
                        cons.planes[plane_ind].satellites[cur_sat_ind + n_nodes_per_plane_per_area - 1].id + 1
                        ] = area_id
                        area_id += 1
                    if assignment_counter <= 0:
                        if dynamic_area_ratio > 0:
                            static_toggle = not static_toggle
                            assignment_counter = static_area_ratio if static_toggle else dynamic_area_ratio
                        else:
                            assignment_counter = static_area_ratio
                cur_plane_ind += n_planes_per_area
            cur_sat_ind += n_nodes_per_plane_per_area

    NODE_AREA_ASSIGNMENT = np.zeros((len(SATELLITE_LIST) + len(GATEWAY_LIST), area_id), dtype=np.int32)
    static_satellite_ind = np.nonzero(SATELLITE_INITIAL_AREA >= 0)
    NODE_AREA_ASSIGNMENT[static_satellite_ind, SATELLITE_INITIAL_AREA[static_satellite_ind]] = 1
    AREA_CONNECTIVITY_MATRIX = np.zeros((area_id, area_id), dtype=np.int32)
    SHORTEST_AREA_PATH_DIST_MATRIX = np.zeros((area_id, area_id), dtype=np.float32)
    SHORTEST_AREA_PATH_PREDECESSOR_MATRIX = np.full((area_id, area_id), -9999, dtype=np.int32)
    SHORTEST_NODE_PATH_DIST_MATRIX = np.zeros((len(SATELLITE_LIST) + len(GATEWAY_LIST),
                                               len(SATELLITE_LIST) + len(GATEWAY_LIST)), dtype=np.float32)
    SHORTEST_NODE_PATH_PREDECESSOR_MATRIX = np.full((len(SATELLITE_LIST) + len(GATEWAY_LIST),
                                                     len(SATELLITE_LIST) + len(GATEWAY_LIST)), -9999, dtype=np.int32)
    return area_id


def update_pos(cur_time=0.):
    global NODE_POSITION_MATRIX
    # Update all satellite positions
    for satellite in SATELLITE_LIST:
        satellite_pos = satellite.plane.plane_solver.xyzPos(satellite.init_time_offset + cur_time)
        NODE_POSITION_MATRIX[satellite.id, :] = satellite_pos
    # Update all gateway positions
    rotation_degree = EARTH_ROTATION_PER_SECONDS * np.fmod(cur_time, SECONDS_PER_DAY)
    rotation_matrix = get_earth_rotation_matrix(rotation_degree)
    for gate in GATEWAY_LIST:
        NODE_POSITION_MATRIX[gate.id + len(SATELLITE_LIST), :] = np.dot(rotation_matrix, gate.init_pos)


def update_links():
    global NODE_POSITION_MATRIX, ISL_LINK_MATRIX, STG_LINK_MATRIX
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
    # TODO: Also return link count, the diameter of the ISL graph (the whole graph should be that + 1)
    return isl_link_broken, isl_link_established, stg_link_broken, stg_link_established


def get_node_area(node_id):
    global NODE_AREA_ASSIGNMENT
    return np.nonzero(NODE_AREA_ASSIGNMENT[node_id, :] > 0)[0]


def get_sat_neighbors(sat_id):
    global ISL_LINK_MATRIX
    return np.nonzero(ISL_LINK_MATRIX[sat_id, :] < np.inf)[0]


def update_area(isl_link_broken, isl_link_established, stg_link_broken, stg_link_established):
    global ISL_LINK_MATRIX, STG_LINK_MATRIX
    global SATELLITE_INITIAL_AREA, NODE_AREA_ASSIGNMENT, AREA_CONNECTIVITY_MATRIX
    global SHORTEST_AREA_PATH_DIST_MATRIX, SHORTEST_AREA_PATH_PREDECESSOR_MATRIX
    global SHORTEST_NODE_PATH_DIST_MATRIX, SHORTEST_NODE_PATH_PREDECESSOR_MATRIX

    # Remove duplicate undirected links
    isl_broken = np.asarray([isl_link_broken[0][isl_link_broken[0] < isl_link_broken[1]],
                             isl_link_broken[1][isl_link_broken[0] < isl_link_broken[1]]]).T
    isl_established = np.asarray([isl_link_established[0][isl_link_established[0] < isl_link_established[1]],
                                  isl_link_established[1][isl_link_established[0] < isl_link_established[1]]]).T
    area_with_nodes_added = set()
    area_with_nodes_removed = set()

    # Deal with broken ISL links first: check if each end node's last area assignments still hold
    # 1. If the end node has a static area assignment, keep using it
    # 2. If the end node has other area assignments, search for its neighbor nodes which:
    #    have the same area assignment but static, or did not have ISL link broken in the last timestep
    #    if found, the end node can also keep those area assignments
    # 3. If both 1 and 2 do not satisfy, treat all the node's links as newly established and re-initialize its area
    isl_broken_sat = np.unique(isl_broken.flatten())
    for broken_pairs in isl_broken:
        for end_node in broken_pairs:
            end_node_area_prev = get_node_area(end_node)
            neighbors = get_sat_neighbors(end_node)
            neighbors_without_broken_links = np.setdiff1d(neighbors, isl_broken_sat)
            keep_prev_area = False
            for prev_area in end_node_area_prev:
                if (prev_area == SATELLITE_INITIAL_AREA[end_node] or
                        prev_area in [SATELLITE_INITIAL_AREA[neighbor] for neighbor in neighbors] or
                        prev_area in sum([list(get_node_area(neighbor)) for neighbor in neighbors_without_broken_links],
                                         [])):
                    keep_prev_area = True
                else:
                    NODE_AREA_ASSIGNMENT[end_node, prev_area] = 0
                    area_with_nodes_removed.add(prev_area)
                    # Note: If the end node loses an area assignment, this effect may propagate to its neighbors
                    # whose areas are dynamically assigned, so there is no guarantee that an area is still fully
                    # connected without iterative checking
            if not keep_prev_area:
                print("Warning: Removed all prev area for sat {}".format(end_node))
                for neighbor in neighbors:
                    isl_established = np.append(isl_established, [[end_node, neighbor]], axis=0)

    # Deal with newly established ISL links:
    # 1. If both link have the same static area assignment, nothing happens
    # 2. If both link have area assignments, decide which node is an ABR:
    #    [1] The node with the most neighbor nodes (plus itself) that have different static area
    #    assignments will be an ABR;
    #    [2] The node with the most neighbor nodes (plus itself) with static area assignment will
    #    be an ABR;
    #    [3] The node with more links will be an ABR;
    #    [4] Lastly, break ties using satellite ids (prefer smaller)
    # 3. Do nothing to the non-ABR node; For the ABR node, add an area assignment to it:
    #    [1] If the non-ABR node has a static area assignment, pick that and add it to the ABR node;
    #    [2] Otherwise, pick an area from the non-ABR node so that the most non-ABR node's neighbors
    #        also have that same area assignment as their [i] static/dynamic area, or [ii] static area,
    #        or [iii] break ties using area ids (prefer smaller)
    # 4. If one end node does not have an area assignment, instead of setting the other end node as an ABR, assign an
    #    area to this node by picking an area from the other node in the same way as 3
    # 5. If both nodes have no area assignment, mark them as unsolved, and loop again to solve them at the end
    #    This should not happen often during the stable state, and could be largely avoided during initialization
    #    by aligning static/dynamic-area satellites interleaved
    while isl_established.size > 0:
        new_isl_unresolved = np.empty((0, 2), dtype=np.int64)
        for established_pairs in isl_established:
            if (SATELLITE_INITIAL_AREA[established_pairs[0]] == SATELLITE_INITIAL_AREA[established_pairs[1]] and
                    SATELLITE_INITIAL_AREA[established_pairs[0]] >= 0):
                continue
            node_0_prev_area = get_node_area(established_pairs[0])
            node_1_prev_area = get_node_area(established_pairs[1])
            if node_0_prev_area.size == 0 and node_1_prev_area.size == 0:
                print("Warning: Unresolved links between no-area nodes {} and {}".format(established_pairs[0],
                                                                                         established_pairs[1]))
                new_isl_unresolved = np.append(new_isl_unresolved,
                                               [[established_pairs[0], established_pairs[1]]], axis=0)
                continue
            if node_0_prev_area.size == 0:
                abr, non_abr = established_pairs[0], established_pairs[1]
            elif node_1_prev_area.size == 0:
                abr, non_abr = established_pairs[1], established_pairs[0]
            else:
                node_0_neighbors = get_sat_neighbors(established_pairs[0])
                node_1_neighbors = get_sat_neighbors(established_pairs[1])
                node_0_neighbors_static_areas = [SATELLITE_INITIAL_AREA[neighbor] for neighbor
                                                 in list(node_0_neighbors) + [established_pairs[0]]
                                                 if SATELLITE_INITIAL_AREA[neighbor] != -1]
                node_1_neighbors_static_areas = [SATELLITE_INITIAL_AREA[neighbor] for neighbor
                                                 in list(node_1_neighbors) + [established_pairs[1]]
                                                 if SATELLITE_INITIAL_AREA[neighbor] != -1]
                node_0_neighbors_unique_static_areas = len(set(node_0_neighbors_static_areas))
                node_1_neighbors_unique_static_areas = len(set(node_1_neighbors_static_areas))
                if node_0_neighbors_unique_static_areas > node_1_neighbors_unique_static_areas:
                    abr, non_abr = established_pairs[0], established_pairs[1]
                elif node_0_neighbors_unique_static_areas < node_1_neighbors_unique_static_areas:
                    abr, non_abr = established_pairs[1], established_pairs[0]
                else:
                    if len(node_0_neighbors_static_areas) > len(node_1_neighbors_static_areas):
                        abr, non_abr = established_pairs[0], established_pairs[1]
                    elif len(node_0_neighbors_static_areas) < len(node_1_neighbors_static_areas):
                        abr, non_abr = established_pairs[1], established_pairs[0]
                    else:
                        if len(node_0_neighbors) > len(node_1_neighbors):
                            abr, non_abr = established_pairs[0], established_pairs[1]
                        elif len(node_0_neighbors) < len(node_1_neighbors):
                            abr, non_abr = established_pairs[1], established_pairs[0]
                        else:
                            # Break ties
                            sorted_nodes = sorted([established_pairs[0], established_pairs[1]])
                            abr, non_abr = sorted_nodes[0], sorted_nodes[1]
            abr_prev_area = get_node_area(abr)
            if SATELLITE_INITIAL_AREA[non_abr] >= 0:
                area_to_add = SATELLITE_INITIAL_AREA[non_abr]
            else:
                non_abr_prev_area = get_node_area(non_abr)
                non_abr_neighbors = get_sat_neighbors(non_abr)
                non_abr_neighbors_areas = sum([list(get_node_area(neighbor)) for neighbor in non_abr_neighbors], [])
                non_abr_neighbors_area_counter = np.asarray([non_abr_neighbors_areas.count(prev_area)
                                                             for prev_area in non_abr_prev_area])
                non_abr_prev_area_selected = non_abr_prev_area[non_abr_neighbors_area_counter ==
                                                               np.max(non_abr_neighbors_area_counter)]
                if non_abr_prev_area_selected.size == 1:
                    area_to_add = non_abr_prev_area_selected[0]
                else:
                    non_abr_neighbors_static_areas = [SATELLITE_INITIAL_AREA[neighbor] for neighbor in non_abr_neighbors
                                                      if SATELLITE_INITIAL_AREA[neighbor] != -1]
                    non_abr_neighbors_area_counter = np.asarray([non_abr_neighbors_static_areas.count(prev_area)
                                                                 for prev_area in non_abr_prev_area_selected])
                    non_abr_prev_area_selected = non_abr_prev_area_selected[non_abr_neighbors_area_counter ==
                                                                            np.max(non_abr_neighbors_area_counter)]
                    if non_abr_prev_area_selected.size == 1:
                        area_to_add = non_abr_prev_area_selected[0]
                    else:
                        # Break ties
                        area_to_add = np.min(non_abr_prev_area_selected)

            if area_to_add not in abr_prev_area:
                NODE_AREA_ASSIGNMENT[abr, area_to_add] = 1
                area_with_nodes_added.add(area_to_add)

        # Check if each area is fully connected
        # If not, only keep the connected component of that area which contains nodes statically assigned to it
        # Remove this area assignment from other nodes previously in this area
        for area_id in range(NODE_AREA_ASSIGNMENT.shape[1]):
            area_sat_list = np.nonzero(NODE_AREA_ASSIGNMENT[:len(SATELLITE_LIST), area_id] > 0)[0]
            area_isl_link_matrix = ISL_LINK_MATRIX[np.ix_(area_sat_list, area_sat_list)]
            area_isl_link_matrix[area_isl_link_matrix == np.inf] = 0
            area_graph = csr_matrix(area_isl_link_matrix)
            n_comp, labels = connected_components(csgraph=area_graph, directed=False, return_labels=True)
            if n_comp > 1:
                static_area_sat = np.argmax(SATELLITE_INITIAL_AREA == area_id)
                static_area_label = labels[np.argmax(area_sat_list == static_area_sat)]
                nodes_to_remove_area = area_sat_list[labels != static_area_label]
                for node_to_remove in nodes_to_remove_area:
                    NODE_AREA_ASSIGNMENT[node_to_remove, area_id] = 0
                    area_with_nodes_removed.add(area_id)

        # Double check that each node belongs to an area
        # If not, add links of those no-area nodes to new_isl_unresolved
        no_area_nodes = np.nonzero(np.sum(NODE_AREA_ASSIGNMENT[:len(SATELLITE_LIST), :], axis=1) == 0)[0]
        for no_area_node in no_area_nodes:
            for neighbor in get_sat_neighbors(no_area_node):
                no_area_pair = sorted([no_area_node, neighbor])
                new_isl_unresolved = np.append(new_isl_unresolved,
                                               [[no_area_pair[0], no_area_pair[1]]], axis=0)
        # Remove duplicate undirected links
        new_isl_unresolved = np.unique(new_isl_unresolved, axis=0)

        # Repeat the process above if there are unresolved links
        if np.array_equal(isl_established, new_isl_unresolved):
            print("Warning: Unable to resolve area assignments for the following links:\n{}".format(isl_established))
            break
        isl_established = new_isl_unresolved

    return update_fixed_area(stg_link_broken, stg_link_established)


def update_fixed_area(stg_link_broken, stg_link_established):
    global ISL_LINK_MATRIX, STG_LINK_MATRIX
    global SATELLITE_INITIAL_AREA, NODE_AREA_ASSIGNMENT, AREA_CONNECTIVITY_MATRIX
    global SHORTEST_AREA_PATH_DIST_MATRIX, SHORTEST_AREA_PATH_PREDECESSOR_MATRIX
    global SHORTEST_NODE_PATH_DIST_MATRIX, SHORTEST_NODE_PATH_PREDECESSOR_MATRIX
    # Deal with broken STG links:
    # Check the area assignment of the gateway node, delete or re-assign areas accordingly
    # TODO

    # Deal with newly established STG links:
    # Check the area assignment of the gateway node, add areas accordingly in the same way as 3 above
    # A gateway is like an ABR in the way that it always belongs to all possible areas of its neighbor satellite
    # while area propagation won't pass across it
    # TODO

    # Update area-level connectivity
    # Only if there is a change compared to the last one:
    #   TODO: Assert the area graph is still fully connected
    #   Calculate area-level shortest path:
    #   Run Dijkstra to update SHORTEST_AREA_PATH_DIST_MATRIX and SHORTEST_AREA_PATH_PREDECESSOR_MATRIX

    # Run OSPF (Dijkstra) within each updated area;
    # Compare with previous routing table in SHORTEST_NODE_PATH_PREDECESSOR_MATRIX;
    # Then update SHORTEST_NODE_PATH_DIST_MATRIX and SHORTEST_NODE_PATH_PREDECESSOR_MATRIX

    # Record the areas with changes in SHORTEST_NODE_PATH_PREDECESSOR_MATRIX

    # Get some statistics here: # node per area, # assigned area per node,
    # # of area-level changes; areas with internal routing table changes and diameter of the area graph
    # return these
    pass


def compute_node_to_node_latency():
    # Note: Run this only at stable state

    # define src nodes
    # define dest nodes
    # create matrix storing all latencies
    # for each pair of src -> dest, compute route and distance (latency) and fill in the matrix
    # also store the hop count to

    # Cross-area routing:
    # First find the area-level path using SHORTEST_AREA_PATH_PREDECESSOR_MATRIX
    # Then, within each area along the path, search for ABRs that connect two areas from NODE_AREA_ASSIGNMENT
    # (including gateway nodes), pick an ABR with the shortest path locally, then move on to the next area

    # TODO: Special baseline case: No areas, ideal route; directly run dijkstra from ISL_LINK_MATRIX (could be slow...)
    # return latency matrix
    pass

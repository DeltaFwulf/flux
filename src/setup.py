"""Sets up objects to be simulated in mesh_2d."""

from os import getcwd
from os.path import join
from math import copysign, pi
import numpy as np
import yaml

from util import update_properties



def setup_2d(defname:str):
    """Sets up simulation data for the mesh_2d script."""

    sim_inputs = {}

    # load runtime settings and mesh definitions
    with open(join(getcwd(), defname), 'r', encoding='utf-8') as cfg:
        cfg = yaml.load(cfg, Loader=yaml.SafeLoader)

    with open(join(getcwd(), 'src', 'data', 'materials.yaml'), encoding='utf-8') as f:
        materials = yaml.load(stream=f, Loader=yaml.SafeLoader)

    # Runtime global settings
    sim_inputs.update({'tf':cfg['tf']})
    sim_inputs.update({'dt_storage':cfg['dt_storage']})
    sim_inputs.update({'theta':cfg['theta']})
    sim_inputs.update({'max_courant':cfg['max_courant']})

    # Meshes
    meshes = {}
    for key, m in cfg['meshes'].items():

        # snap lines to the grid and convert to mesh indices
        m['lines'] = [tuple((round(p[0] / m['dx']), round(p[1] / m['dy'])) for p in l)\
                 for l in m['lines']]

        i_min = min(p[0] for l in m['lines'] for p in l)
        i_max = max(p[0] for l in m['lines'] for p in l)
        j_min = min(p[1] for l in m['lines'] for p in l)
        j_max = max(p[1] for l in m['lines'] for p in l)

        m.update({'i_arr':np.arange(i_min, i_max + 1, 1)})
        m.update({'j_arr':np.arange(j_min, j_max + 1, 1)})

        m.update({'regions_x':[find_regions_2d(direction='x', ind_n=j, ind_p=m['i_arr'],\
            lines=m['lines']) for j in m['j_arr']]})
        m.update({'regions_y':[find_regions_2d(direction='y', ind_n=i, ind_p=m['j_arr'],\
            lines=m['lines']) for i in m['i_arr']]})

        # replace material string with material definition
        mat = materials.get(m['material'])
        if mat is None:
            print(f"Material {m['material']} is not found at /src/data/materials.yaml.")
            raise ValueError

        m.update({'material':mat})
        find_edges(m)
        calc_bc_relations(m)

        m.update({'x':m['dx']*m['i_arr']})
        m.update({'y':m['dy']*m['j_arr']})

        # Meshes store 'u' for final results, u_latest for use in next timestep, u_last
        # for reference by other meshes.
        m.update({'u':np.zeros((m['i_arr'].size, m['j_arr'].size, 1), float) + m['u0']})
        m.update({'u_latest':m['u'][:, :, -1]})
        m.update({'u_last':m['u'][:, :, -1]})

        update_properties(m)

        meshes.update({key:m})

    sim_inputs.update({'meshes':meshes})

    # Environment
    environment = {}
    sim_inputs.update({'environment':environment})

    return sim_inputs



def find_regions_2d(direction:str, ind_n:int, ind_p:np.ndarray, lines:list) -> list[list]:
    """
    Uses a transition method to identify all regions in this slice.

    Regions are either 'i', internal or 'b', bounding.

    Internal regions are solved for in the TDMA steps.
    Bounding regions have boundary conditions applied in the BC step, in the correct direction.
    
    boundary lines have a direction, 'l' or 'r' in which to apply the boundary.
    'l' is descending index, 'r' is ascending index.
    
    ind_n is the slice's index in the normal axis
    ind_p is the array of indices along the parallel direction

    each region has a type and bounding points. If edge it has a direction and boundary number
    {'type':'internal', 'bounds':(pa, pb)}
    {'type':'edge', 'bounds':(pa, pb), 'direction':1, 'bc_index':5}
    direction is +1 for ascending order, -1 for descending. (towards next internal region)
    """

    a = 0 if direction == 'x' else 1 # gives line coordinate to inspect for normality
    transitions = []
    lefts = 0
    rights = 0

    for p in ind_p:

        for line_index, line in enumerate(lines):

            na = line[0][1 - a] - ind_n
            nb = line[1][1 - a] - ind_n
            is_normal = line[0][a] == line[1][a]
            spans = copysign(1, na) != copysign(1, nb) or na*nb == 0
            touches = p in (line[0][a], line[1][a])

            if is_normal and spans and touches:

                if na*nb != 0:                      # both
                    d_normal = 2
                    lefts += 1
                    rights += 1
                    d_parallel = 1 if (rights % 2 == lefts % 2 == 1) else -1
                elif na > 0 or nb > 0:      # right
                    d_normal = 1
                    rights += 1
                    d_parallel = 1 if rights % 2 == 1 else -1
                else:                               # left
                    d_normal = -1
                    lefts += 1
                    d_parallel = 1 if lefts % 2 == 1 else -1

                transitions.append([p, line_index, d_normal, d_parallel])
                break

    regions = []
    for i, t in enumerate(transitions):
        if i == 0:
            continue

        t_prev = transitions[i-1]

        # region bounds
        na = (t_prev[0], ind_n) if a == 0 else (ind_n, t_prev[0])
        nb = (t[0], ind_n) if a == 0 else (ind_n, t[0])
        reg = {'bounds':(na[a], nb[a])}

        # edge region
        is_edge = False
        l = 0
        for l, line in enumerate(lines):
            if na in line and nb in line:
                is_edge = True
                break

        if is_edge:
            reg.update({'type':'edge', 'direction':-t[2]*t[3], 'bc':l})
            regions.append(reg)
            continue

        # internal region
        if t[3] == -1:
            reg.update({'type':'internal', 'bc_s':t_prev[1], 'bc_e':t[1]})
            regions.append(reg)
            continue

    return regions



def find_edges(mesh:dict) -> list:
    """Find all edges in region set, store necessary info for BC calculations."""

    edges = []

    for k, regs in enumerate(mesh['regions_x']):
        for reg in regs:
            if reg['type'] == 'edge':

                s = int(np.where(mesh['i_arr'] == reg['bounds'][0])[0])
                e = int(np.where(mesh['i_arr'] == reg['bounds'][1])[0])

                edge = {}
                edge.update({'indices':(s, e, k)})  # start, end, normal
                edge.update({'line_index': reg['bc']})
                edge.update({'direction':(0, reg['direction'])})
                edge.update({'h_norm':mesh['dy']})
                edge.update({'emissivity':mesh['material']['emissivity']})

                if mesh['curvature'] == 0:
                    areas = mesh['depth']*mesh['dx']*np.r_[1, 2*np.ones(e - s - 1, float), 1]
                    edge.update({'areas':areas})
                elif mesh['curvature'] == 1:
                    r = mesh['dx']*np.arange(reg['bounds'][0], reg['bounds'][1] + 1)
                    areas = pi*np.r_[((r[1] + r[0])**2 / 4 - r[0]**2),
                                     ((r[2:]+r[1:-1])**2 - (r[1:-1] + r[:-2])**2) / 4,
                                     (r[-1]**2 - (r[-1] + r[-2])**2 / 4)]
                    edge.update({'areas':areas})

                edges.append(edge)

    for k, regs in enumerate(mesh['regions_y']):
        for reg in regs:
            if reg['type'] == 'edge':

                s = int(np.where(mesh['j_arr'] == reg['bounds'][0])[0])
                e = int(np.where(mesh['j_arr'] == reg['bounds'][1])[0])

                edge = {}
                edge.update({'indices': (s, e, k)})
                edge.update({'line_index':reg['bc']})
                edge.update({'direction':(reg['direction'], 0)})
                edge.update({'h_norm':mesh['dx']})
                edge.update({'emissivity':mesh['material']['emissivity']})

                areas = np.zeros_like(edge['indices'])
                if mesh['curvature'] == 0:
                    areas = mesh['depth']*mesh['dy']*np.r_[1, 2*np.ones(e - s - 1, float), 1]
                    edge.update({'areas':areas})
                elif mesh['curvature'] == 1:
                    r = mesh['dx']*mesh['i_arr'][k]
                    areas = 2*pi*r*mesh['dy']*np.r_[1, 2*np.ones(e - s - 1, float), 1]
                    edge.update({'areas':areas})

                edges.append(edge)

    # sort edges by line index
    edges = [edges[p] for p in np.argsort([edge['line_index'] for edge in edges])]
    mesh.update({'edges':edges})


def calc_bc_relations(mesh:dict):
    """Returns a list of boundary condition indices relevant to each edge in a mesh."""

    edge_bcs = []
    for l in range(len(mesh['edges'])):

        # iterate through all boundary conditions, add relevant entries to list
        relevant = [i for i, bc in enumerate(mesh['boundary_conditions']) if bc['edge'] == l]
        edge_bcs.append(relevant)

    mesh.update({'edge_bcs':edge_bcs})

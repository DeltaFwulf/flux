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
    sim_inputs.update({'force_finer':cfg['force_finer']})

    # Meshes
    meshes = {}
    for key, m in cfg['meshes'].items():

        z = 0
        while z < 2:
            # snap lines to the grid and convert to mesh indices
            m['line_indices'] = [tuple((round(p[0] / m['dx']), round(p[1] / m['dy'])) for p in l)\
                    for l in m['lines']]

            i_min = min(p[0] for l in m['line_indices'] for p in l)
            i_max = max(p[0] for l in m['line_indices'] for p in l)
            j_min = min(p[1] for l in m['line_indices'] for p in l)
            j_max = max(p[1] for l in m['line_indices'] for p in l)

            m.update({'i_arr':np.arange(i_min, i_max + 1, 1)})
            m.update({'j_arr':np.arange(j_min, j_max + 1, 1)})

            # get all x slice regions
            m.update({'regions_x':[slice_regions(m, direction='x', n=j) for j in m['j_arr']]})

            # get all y slice regions
            m.update({'regions_y':[slice_regions(m, direction='y', n=i) for i in m['i_arr']]})

            # rescale mesh
            if z == 0:
                fit_mesh_resolution(m, cfg['force_finer'])

            z += 1

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
    sim_inputs.update({'environment':cfg['environment']})

    return sim_inputs



def slice_regions(mesh:dict, direction:str, n:int) -> list[dict]:
    """Calculates regions within a mesh slice.

    Regions are returned in ascending index order (+x or +y direction). These regions
    are used by the mesh ADI solver to identify where to apply boundary conditions
    and when to apply edge states.

    The following variables are used:
    - direction; 'x' or 'y'. This is the direction parallel to the slice.
    - ind_n; this is the index of the slice, in the normal direction.
    - ind_p; this is the array of indices within the slice, in the parallel direction.
    - line_inds; this is the array of line boundary indices, snapped to the mesh grid.
    - line_pts; this is the array of line boundary point locations, not snapped to mesh grid.
    
    Ouptutted regions contain the following:
    - type; either 'edge' or 'internal'.
    - direction; +- 1. This is the direction, normal to the edge line to the mesh interior.
    - if internal, line_s and line_e; give the line indices of the bounding mesh edges.
    
    The output is a list of all region dictionaries in the slice, in ascending order.
    """

    a = 0 if direction == 'x' else 1 # gives line coordinate to inspect for normality
    transitions = []
    lefts = 0
    rights = 0

    for p in (mesh['i_arr'] if direction == 'x' else mesh['j_arr']):

        for l, line in enumerate(mesh['line_indices']):

            na = line[0][1 - a] - n
            nb = line[1][1 - a] - n
            is_normal = line[0][a] == line[1][a]
            spans = copysign(1, na) != copysign(1, nb) or na*nb == 0
            touches = p in (line[0][a], line[1][a])

            if is_normal and spans and touches:

                if na*nb != 0:                      # both
                    dn = 2
                    lefts += 1
                    rights += 1
                    dp = 1 if (rights % 2 == lefts % 2 == 1) else -1
                elif na > 0 or nb > 0:      # right
                    dn = 1
                    rights += 1
                    dp = 1 if rights % 2 == 1 else -1
                else:                               # left
                    dn = -1
                    lefts += 1
                    dp = 1 if lefts % 2 == 1 else -1

                transitions.append({'ind_parallel':p,
                                    'line_index':l,
                                    'normal':dn,
                                    'parallel':dp})

                break

    regions = []
    for i, t in enumerate(transitions):
        if i == 0:
            continue

        t_prev = transitions[i-1]

        # region bounds
        na = (t_prev['ind_parallel'], n) if a == 0 else (n, t_prev['ind_parallel'])
        nb = (t['ind_parallel'], n) if a == 0 else (n, t['ind_parallel'])
        reg = {'bounds':(na[a], nb[a])}

        is_edge = False
        for l, line in enumerate(mesh['line_indices']):
            if na in line and nb in line:
                is_edge = True
                break

        # edge region
        if is_edge:
            reg.update({'length':abs(mesh['lines'][l][1][a] - mesh['lines'][l][0][a])})
            reg.update({'type':'edge', 'direction':-t['normal']*t['parallel'], 'line':l})
            regions.append(reg)

        # internal region
        elif t['parallel'] == -1:
            reg.update({'length':abs(mesh['lines'][t['line_index']][0][a] -
                                 mesh['lines'][t_prev['line_index']][0][a])})

            reg.update({'type':'internal', 'line_s':t_prev['line_index'], 'line_e':t['line_index']})
            regions.append(reg)
            
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
                edge.update({'line_index': reg['line']})
                edge.update({'direction':(0, reg['direction'])})
                edge.update({'hp':mesh['dx']})
                edge.update({'hn':mesh['dy']})
                edge.update({'emissivity':mesh['material']['emissivity']})

                if mesh['curvature'] == 0:
                    areas = mesh['depth']*mesh['dx']*np.r_[1, 2*np.ones(e - s - 1, float), 1]
                    edge.update({'areas':areas})
                    edge.update({'perimeter':2*(mesh['depth'] + mesh['dx']*(e - s))})
                elif mesh['curvature'] == 1:
                    r = mesh['dx']*np.arange(reg['bounds'][0], reg['bounds'][1] + 1)
                    areas = pi*np.r_[((r[1] + r[0])**2 / 4 - r[0]**2),
                                     ((r[2:]+r[1:-1])**2 - (r[1:-1] + r[:-2])**2) / 4,
                                     (r[-1]**2 - (r[-1] + r[-2])**2 / 4)]
                    edge.update({'areas':areas})
                    edge.update({'perimeter':2*pi*(r[-1] + r[0])})

                edges.append(edge)

    for k, regs in enumerate(mesh['regions_y']):
        for reg in regs:
            if reg['type'] == 'edge':

                s = int(np.where(mesh['j_arr'] == reg['bounds'][0])[0])
                e = int(np.where(mesh['j_arr'] == reg['bounds'][1])[0])

                edge = {}
                edge.update({'indices': (s, e, k)})
                edge.update({'line_index':reg['line']})
                edge.update({'direction':(reg['direction'], 0)})
                edge.update({'hp':mesh['dy']})
                edge.update({'hn':mesh['dx']})
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



def fit_mesh_resolution(mesh:dict, force_finer:bool=True) -> tuple[float, float]:
    """Calculates a mesh resolution (dx, dy) that tiles the mesh with integer elements."""

    widths_x = [reg['length'] for slc in mesh['regions_x'] for reg in slc]
    widths_y = [reg['length'] for slc in mesh['regions_y'] for reg in slc]

    power_x = max(len(str(pt[0]).split(".")[1]) for line in mesh['lines'] for pt in line)
    power_y = max(len(str(pt[1]).split(".")[1]) for line in mesh['lines'] for pt in line)

    w_scaled_x = [round(w*10**power_x) for w in widths_x]
    w_scaled_y = [round(w*10**power_y) for w in widths_y]

    dx = float(np.gcd.reduce(w_scaled_x)) / 10**power_x
    dy = float(np.gcd.reduce(w_scaled_y)) / 10**power_y

    if dx > mesh['dx'] and force_finer:
        dx /= np.ceil(dx / mesh['dx'])

    if dy > mesh['dy'] and force_finer:
        dy /= np.ceil(dy / mesh['dy'])

    print(f"new mesh resolution (dx, dy): {dx, dy}")
    mesh.update({'dx':dx})
    mesh.update({'dy':dy})

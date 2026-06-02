"""Setup script for 2D mesh dictionaries.

This contains all the functions required to build a useable mesh
dictionary in flux.

Meshes contain:
- edges
- regions
- resolutions (x, y)
- boundary conditions
- curvature
- material

"""

from os import getcwd
from os.path import join
from copy import deepcopy
from math import copysign, pi
import numpy as np
import yaml

from util import update_properties, calc_bc_relations, get_link_data
from heat_transfer import conduction, convection, radiation


# TODO: preserve global x, y array values

def init_mesh(mesh_def:dict, force_finer:bool):
    """Prepares a mesh dictionary for simulation."""

    with open(join(getcwd(), 'src', 'data', 'materials.yaml'), encoding='utf-8') as f:
        materials = yaml.load(stream=f, Loader=yaml.SafeLoader)

    m = deepcopy(mesh_def)

    x_min = min(pt[0] for line in m['lines'] for pt in line)
    y_min = min(pt[1] for line in m['lines'] for pt in line)

    z = 0
    while z < 2:
        # mesh indices for line points
        m['line_indices'] = [tuple((round((p[0] - x_min) / m['dx']),
                                    round((p[1] - y_min) / m['dy']))
                             for p in l) for l in m['lines']]

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
            fit_mesh_resolution(m, force_finer)

        z += 1

    mat = materials.get(m['material'])
    if mat is None:
        print(f"Material {m['material']} is not found at /src/data/materials.yaml.")
        raise ValueError

    m.update({'material':mat})

    find_edges(m)
    calc_bc_relations(m)
    mask_void_regions(m)

    m.update({'x':x_min + m['dx']*m['i_arr']})
    m.update({'y':y_min + m['dy']*m['j_arr']})

    # Meshes store 'u' for final results, u_latest for use in next timestep, u_last
    # for reference by other meshes.
    m.update({'u':np.zeros((m['i_arr'].size, m['j_arr'].size, 1), float) + m['u0']})
    m.update({'u_latest':m['u'][:, :, -1]})
    m.update({'u_last':m['u'][:, :, -1]})
    m.update({'edge_fluxes':[np.zeros(1, float) for e in range(len(m['edges']))]})
    m.update({'edge_powers':[np.zeros(1, float) for e in range(len(m['edges']))]})

    update_properties(m)

    return m



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



def fit_mesh_resolution(mesh:dict, force_finer:bool=True) -> tuple[float, float]:
    """Calculates a mesh resolution (dx, dy) that tiles the mesh with integer elements."""

    widths_x = [reg['length'] for slc in mesh['regions_x'] for reg in slc]
    widths_y = [reg['length'] for slc in mesh['regions_y'] for reg in slc]

    # calculate scaling order of magnitudes, x and y
    pow_x = 0
    pow_y = 0
    for line in mesh['lines']:
        for pt in line:
            if round(pt[0]) != pt[0]:
                pow_x = max(pow_x, len(str(pt[0]).split(".")[1]))

            if round(pt[1]) != pt[1]:
                pow_y = max(pow_y, len(str(pt[1]).split(".")[1]))

    # power_x = max(len(str(float(pt[0])).split(".")[1]) for line in mesh['lines'] for pt in line)
    # power_y = max(len(str(float(pt[1])).split(".")[1]) for line in mesh['lines'] for pt in line)

    w_scaled_x = [round(w*10**pow_x) for w in widths_x]
    w_scaled_y = [round(w*10**pow_y) for w in widths_y]

    dx = float(np.gcd.reduce(w_scaled_x)) / 10**pow_x
    dy = float(np.gcd.reduce(w_scaled_y)) / 10**pow_y

    if dx > mesh['dx'] and force_finer:
        dx /= np.ceil(dx / mesh['dx'])

    if dy > mesh['dy'] and force_finer:
        dy /= np.ceil(dy / mesh['dy'])

    print(f"{mesh['label']}, new resolution (dx, dy): {dx, dy}")
    mesh.update({'dx':dx})
    mesh.update({'dy':dy})



def calc_scaling_power(num) -> int:
    """Calculates the order of magnitude required to make an integer."""



def mask_void_regions(mesh:dict) -> np.ndarray:
    """Masks a mesh's void regions so plotters ignore them.
    
    By scanning through all regions in one direction, the 'voids'
    between different regions can be located and a mask array created
    for plotters, so that meshes are plotted with clean boundaries.
    """

    mask = np.zeros((mesh['i_arr'].size, mesh['j_arr'].size,), bool)

    # iterate through all x slices and locate all void regions
    for j, row in enumerate(mesh['regions_x']):
        for reg in row:
            mask[:, j] = [not (reg['bounds'][0] <= i <= reg['bounds'][1]) for i in mesh['i_arr']]

    # update the mesh's 'mask' term
    mesh.update({'mask':mask})



def update_mesh(*, mesh:dict, dt:float, curv:int, theta:float) -> np.ndarray:
    """Updates the state of a single mesh over a single timestep via the ADI method."""

    i_arr = mesh['i_arr']
    j_arr = mesh['j_arr']
    dx = mesh['dx']
    dy = mesh['dy']
    alpha = mesh['diffusivity']

    # calculate mesh coefficients
    bxx_c = alpha*dt*(1 - theta) / dx**2
    bxx_n = alpha*dt*theta / dx**2
    byy_c = 0 if curv > 1 else (alpha*(1 - theta)*dt / dy**2)
    byy_n = 0 if curv > 1 else (alpha*theta*dt / dy**2)
    bx_c = curv*alpha*(1 - theta)*dt / (2*dx)
    bx_n = curv*alpha*theta*dt / (2*dx)

    u_in = mesh['u_last']
    u_mid = np.zeros_like(u_in, float)

    # row slices (across x)
    for j in range(j_arr.size):
        for reg in mesh['regions_x'][j]:

            s = np.where(i_arr == reg['bounds'][0])[0][0]
            e = np.where(i_arr == reg['bounds'][1])[0][0]

            if reg['type'] == 'edge':

                edge = mesh['edges'][reg['line']]
                edge_state = mesh['edge_states'][reg['line']]

                if edge_state['type'] == 'direct':
                    u_mid[s:e+1, j] = edge_state['values']
                else:
                    d_edge = sum(edge['direction'])
                    u_mid[s:e+1, j] = u_in[s:e+1, j+d_edge] - dy*edge_state['values']*d_edge

                continue

            line_s = reg['line_s']
            line_e = reg['line_e']
            type_s = mesh['edge_states'][line_s]['type']
            type_e = mesh['edge_states'][line_e]['type']
            val_s = mesh['edge_states'][line_s]['values'][j - mesh['edges'][line_s]['indices'][0]]
            val_e = mesh['edge_states'][line_e]['values'][j - mesh['edges'][line_e]['indices'][0]]

            a = np.r_[0.0, -bxx_n[s:e-1, j] + bx_n[s:e-1, j] / (dx*i_arr[s+1:e]), 0.0 if\
                        type_e == 'direct' else -1.0]

            b = np.r_[1.0 if type_s == 'direct' else -1.0, 1 + 2*bxx_n[s+1:e,j], 1.0]

            c = np.r_[0.0 if type_s == 'direct' else 1.0, -bxx_n[s+2:e+1,j] -\
                        bx_n[s+2:e+1,j] / (dx*i_arr[s+1:e]), 0.0]

            d = np.r_[val_s if type_s == 'direct' else dx*val_s,\

                        byy_c[s+1:e,j-1]*u_in[s+1:e,j-1] +\
                        (1 - 2*byy_c[s+1:e,j])*u_in[s+1:e,j] +\
                        byy_c[s+1:e,j+1]*u_in[s+1:e,j+1],\

                        val_e if type_e == 'direct' else dx*val_e]

            u_mid[s:e+1,j] = tdma(u_in[s:e+1,j], a, b, c, d)

    # column slices (across y)
    u_out = u_mid
    for i in range(i_arr.size):
        for reg in mesh['regions_y'][i]:

            s = np.where(j_arr == reg['bounds'][0])[0][0]
            e = np.where(j_arr == reg['bounds'][1])[0][0]

            if reg['type'] == 'edge':

                edge = mesh['edges'][reg['line']]
                edge_state = mesh['edge_states'][reg['line']]

                if edge_state['type'] == 'direct':
                    u_mid[i, s:e+1] = edge_state['values']
                else:
                    d_edge = sum(edge['direction'])
                    u_mid[i, s:e+1] = u_in[i+d_edge, s:e+1] - dx*edge_state['values']*d_edge

                continue

            line_s = reg['line_s']
            line_e = reg['line_e']
            type_s = mesh['edge_states'][line_s]['type']
            type_e = mesh['edge_states'][line_e]['type']
            val_s = mesh['edge_states'][line_s]['values'][i - mesh['edges'][line_s]['indices'][0]]
            val_e = mesh['edge_states'][line_e]['values'][i - mesh['edges'][line_e]['indices'][0]]

            a = np.r_[0.0, -byy_n[i, s:e-1], 0.0 if type_e == 'direct' else -1.0]

            b = np.r_[1.0 if type_s == 'direct' else -1.0, 1 + 2*byy_n[i, s+1:e], 1.0]

            c = np.r_[0.0 if type_s == 'direct' else 1.0, -byy_n[i,s+2:e+1], 0.0]

            d = np.r_[val_s if type_s == 'direct' else dy*val_s,\

                (bxx_c[i+1,s+1:e] + bx_c[i+1,s+1:e] / (dx*i_arr[i]))*u_mid[i+1,s+1:e] +\
                (1 - 2*bxx_c[i,s+1:e])*u_mid[i,s+1:e] +\
                (bxx_c[i-1,s+1:e] - bx_c[i-1,s+1:e] / (dx*i_arr[i]))*u_mid[i-1,s+1:e],

                val_e if type_e == 'direct' else dy*val_e]

            u_out[i,s:e+1] = tdma(u_mid[i,s:e+1], a, b, c, d)

    return u_out



def tdma(x, a, b, c, d) -> np.ndarray:
    """Solves for x given a tridiagonal matrix, following Thomas' Algorithm."""

    # forward substitution
    p = np.zeros_like(x, float)
    q = np.zeros_like(x, float)

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, x.size):
        dn = b[i] + a[i]*p[i - 1]
        p[i] = - c[i] / dn
        q[i] = (d[i] - a[i]*q[i - 1]) / dn

    # back substitution for x
    x[-1] = q[-1]
    for i in range(x.size - 2, -1, -1):
        x[i] = p[i]*x[i + 1] + q[i]

    return x



def calc_edge_states(cfg:dict) -> None:
    """Updates all mesh edge state arrays in the simulation and calculates edge fluxes.
    
    This function has three main tasks:
    - calculate the thermal flux at each edge
    - calculate the new edge state (gradient / value)
    - integrate edge fluxes for edge powers

    q is calculated relative to the edge direction, so positive q means flux into the mesh.
    power, likewise will be relative to the mesh.

    gradients are relative to the grid, so a positive gradient means temperature increases
    in ascending x or y.
    """

    for mesh in cfg['meshes'].values():

        edge_states = []
        fluxes = []
        powers = []

        for l, edge in enumerate(mesh['edges']):

            s, e, n = edge['indices']
            d = sum(edge['direction'])

            edge_link = edge | {'emissivity':mesh['material']['emissivity']}

            # get appropriate slice of temperature, conductivity, etc.
            if edge['direction'][0] == 0: # slice along x axis
                edge_link.update({'u':mesh['u_last'][s:e+1, n]})
                edge_link.update({'u_in':mesh['u_last'][s:e+1, n+d]})
                edge_link.update({'k_bar':0.5*(mesh['k'][s:e+1, n] + mesh['k'][s:e+1, n+d])})

            else:
                edge_link.update({'u':mesh['u_last'][n, s:e+1]})
                edge_link.update({'u_in':mesh['u_last'][n+d, s:e+1]})
                edge_link.update({'k_bar':0.5*(mesh['k'][n, s:e+1] + mesh['k'][n+d, s:e+1])})

            state_val = np.zeros((e - s + 1), float)
            flux = 0.0

            for bc in mesh['edge_bcs'][l]:

                boundary_condition = mesh['boundary_conditions'][bc]

                match boundary_condition['mode']:
                    case 'dirichlet':
                        state_type = 'direct'
                        state_val += boundary_condition['value']
                        du = edge_link['u_in'] - boundary_condition['value']
                        q = sum(edge['direction'])*edge_link['k_bar']*du / edge_link['h_norm']
                        flux = q
                        break

                    case 'neumann':
                        state_type = 'gradient'
                        state_val += boundary_condition['value']
                        q = -sum(edge_link['direction'])*edge_link['k_bar']*state_val
                        flux = q
                        break

                    case 'conduction':
                        state_type = 'direct'
                        pair_link = get_link_data(cfg, edge, boundary_condition)
                        edge_state = conduction(edge_link, pair_link)
                        q = edge_state['q']
                        state_val = edge_state['u_int']
                        flux = q
                        break

                    case 'convection':
                        state_type = 'gradient'
                        pair_link = get_link_data(cfg, edge, boundary_condition)
                        q = convection(edge_link, pair_link)
                        state_val -= sum(edge['direction'])*q / edge_link['k_bar']
                        flux += q

                    case 'radiation':
                        state_type = 'gradient'
                        pair_link = get_link_data(cfg, edge, boundary_condition)
                        q = radiation(edge_link, pair_link)
                        state_val -= sum(edge['direction'])*q / edge_link['k_bar']
                        flux += q

            edge_states.append({'type':state_type, 'values':state_val})
            fluxes.append(np.sum(flux))
            powers.append(np.sum(flux*edge['areas']))

        mesh.update({'edge_states':edge_states})
        mesh.update({'edge_fluxes_latest':fluxes})
        mesh.update({'edge_powers_latest':powers})

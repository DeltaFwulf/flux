"""Contains functions used to manipulate meshes."""

from os import getcwd
from os.path import join
from copy import deepcopy
from math import copysign, pi
import numpy as np
import yaml

from .util import update_properties, calc_bc_relations, get_decimal_resolution, calc_face_perimeter
from .heat_transfer import conduction, convection, radiation



def create_mesh(mesh_def:dict, force_finer:bool):
    """Prepares a mesh dictionary for simulation."""

    with open(join(getcwd(), 'src', 'data', 'materials.yaml'), encoding='utf-8') as f:
        materials = yaml.load(stream=f, Loader=yaml.SafeLoader)

    m = deepcopy(mesh_def)
    m.pop('u0')

    x_min = min(pt[0] for line in m['lines'] for pt in line)
    y_min = min(pt[1] for line in m['lines'] for pt in line)

    z = 0
    while z < 2:
        m.update({'i':np.arange(0, max((p[0] - x_min) / m['dx'] for l in m['lines'] for p in l) + 1, dtype=int)})
        m.update({'j':np.arange(0, max((p[1] - y_min) / m['dy'] for l in m['lines'] for p in l) + 1, dtype=int)})

        x_res = max(get_decimal_resolution(n) for n in (m['dx'], x_min))
        y_res = max(get_decimal_resolution(n) for n in (m['dy'], y_min))
        m.update({'x':np.round(x_min + m['dx']*m['i'], x_res)})
        m.update({'y':np.round(y_min + m['dy']*m['j'], y_res)})

        m.update({'regions_x':[slice_regions('x', m['x'], y, m['lines']) for y in m['y']]})
        m.update({'regions_y':[slice_regions('y', m['y'], x, m['lines']) for x in m['x']]})

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

    # Meshes store 'u' for final results, u_latest for use in next timestep, u_prev
    # for use in current timestep.
    m.update({'u':np.zeros((m['i'].size, m['j'].size, 1), float) + mesh_def['u0']})
    m.update({'u_latest':m['u'][:, :, -1]})
    m.update({'u_prev':m['u'][:, :, -1]})
    m.update({'edge_fluxes':[np.zeros(1, float) for e in range(len(m['edges']))]})
    m.update({'edge_powers':[np.zeros(1, float) for e in range(len(m['edges']))]})

    update_properties(m)

    m.update({'net_energy':np.zeros(1, float)})

    return m



def slice_regions(direction:str, xp:np.ndarray, xn:float, lines:list) -> list[dict]:
    """Calculates regions within a mesh slice.

    Regions are returned in ascending index order (+x or +y direction). These regions
    are used by the mesh ADI solver to identify where to apply boundary conditions
    and when to apply edge states.

    - direction is either 'x' or 'y', giving the orientation of the slice
    - xp is the array of parallel mesh node locations in the grid.
    - xn is the coordinate of the slice in the normal axis.
    - lines is the list of all line endpoint coordinates in the grid.

    Ouptutted regions contain the following:
    - type; either 'edge' or 'internal'.
    - direction; +- 1. This is the direction, normal to the edge line to the mesh interior.
    - if internal, line_s and line_e; give the line indices of the bounding mesh edges.
    
    The output is a list of all region dictionaries in the slice, in ascending order.
    """

    a = 0 if direction == 'x' else 1 # gives line coordinate to inspect for normality
    transitions, regions = [], []
    lefts, rights = 0, 0

    for p in xp:
        for l, line in enumerate(lines):

            is_normal = line[0][a] == line[1][a]
            na = line[0][1 - a] - xn
            nb = line[1][1 - a] - xn
            spans = copysign(1, na) != copysign(1, nb) or na + nb in (na, nb)
            touches = p == line[0][a]

            if not (is_normal and spans and touches):
                continue

            if na*nb != 0:  # both directions
                dn = 2
                lefts += 1
                rights += 1
                dp = 1 if (rights % 2 == lefts % 2 == 1) else -1
            elif na + nb > 0:  # right
                dn = 1
                rights += 1
                dp = 1 if rights % 2 == 1 else -1
            else:  # left
                dn = -1
                lefts += 1
                dp = 1 if lefts % 2 == 1 else -1

            transitions.append({'p':p,
                                'line_index':l,
                                'sign_n':dn,
                                'sign_p':dp})

            break

    for t, trans in enumerate(transitions[1:], start=1):
        trans_prev = transitions[t - 1]
        bnd_a = int((trans_prev['p'] - xp[0]) / abs(xp[1] - xp[0]))
        bnd_b = int((trans['p'] - xp[0]) / abs(xp[1] - xp[0]))

        reg = {'bounds':(bnd_a, bnd_b)}
        reg.update({'length':abs(trans['p'] - trans_prev['p'])})

        is_edge = False
        for l, line in enumerate(lines):
            aligned = set([line[0][a], line[1][a]]) == set([trans['p'], trans_prev['p']])
            if aligned and line[0][1 - a] == xn:
                is_edge = True
                break

        if not is_edge and trans['sign_p'] != -1:
            continue

        if is_edge:
            reg.update({'type':'edge', 'line':l})
            reg.update({'direction':-trans['sign_n']*trans['sign_p']})

        else:
            reg.update({'type':'internal'})
            reg.update({'line_s':trans_prev['line_index'], 'line_e':trans['line_index']})

        regions.append(reg)

    return regions



def find_edges(mesh:dict) -> list:
    """Find all edges in region set, store necessary info for BC calculations."""

    edges = []

    for k, regs in enumerate(mesh['regions_x']):
        for reg in regs:
            if reg['type'] == 'edge':

                s = int(np.where(mesh['i'] == reg['bounds'][0])[0][0])
                e = int(np.where(mesh['i'] == reg['bounds'][1])[0][0])

                edge = {}
                edge.update({'indices':(s, e, k)})  # start, end, normal
                edge.update({'bounds':(mesh['x'][s], mesh['x'][e], mesh['y'][k])})
                edge.update({'line_index': reg['line']})
                edge.update({'direction':(0, reg['direction'])})
                edge.update({'hp':mesh['dx']})
                edge.update({'hn':mesh['dy']})

                edge.update({'areas':calc_areas(edge['bounds'],
                                                edge['hp'],
                                                edge['direction'],
                                                mesh['curvature'],
                                                mesh.get('depth'))})

                edge.update({'perimeter':calc_face_perimeter(edge['bounds'],
                                           edge['direction'],
                                           mesh['curvature'],
                                           mesh.get('depth'))})

                edges.append(edge)

    for k, regs in enumerate(mesh['regions_y']):
        for reg in regs:
            if reg['type'] == 'edge':

                s = int(np.where(mesh['j'] == reg['bounds'][0])[0][0])
                e = int(np.where(mesh['j'] == reg['bounds'][1])[0][0])

                edge = {}
                edge.update({'indices':(s, e, k)})
                edge.update({'bounds':(mesh['y'][s], mesh['y'][e], mesh['x'][k])})
                edge.update({'line_index':reg['line']})
                edge.update({'direction':(reg['direction'], 0)})
                edge.update({'hp':mesh['dy']})
                edge.update({'hn':mesh['dx']})

                edge.update({'areas':calc_areas(edge['bounds'],
                                                edge['hp'],
                                                edge['direction'],
                                                mesh['curvature'],
                                                mesh.get('depth'))})

                edge.update({'perimeter':calc_face_perimeter(edge['bounds'],
                                           edge['direction'],
                                           mesh['curvature'],
                                           mesh.get('depth'))})

                edges.append(edge)

    # sort edges by line index
    edges = [edges[p] for p in np.argsort([edge['line_index'] for edge in edges])]
    mesh.update({'edges':edges})



def fit_mesh_resolution(mesh:dict, force_finer:bool=True) -> tuple[float, float]:
    """Calculates a mesh resolution (dx, dy) that tiles the mesh with integer elements."""

    # Does region actually need a 'length' term or can we just reconstruct it with dx or dy?
    widths_x = [reg['length'] for slc in mesh['regions_x'] for reg in slc]
    widths_y = [reg['length'] for slc in mesh['regions_y'] for reg in slc]

    # calculate scaling order of magnitudes, x and y
    pow_x = max(get_decimal_resolution(pt[0]) for line in mesh['lines'] for pt in line)
    pow_y = max(get_decimal_resolution(pt[1]) for line in mesh['lines'] for pt in line)

    w_scaled_x = [round(w*10**pow_x) for w in widths_x]
    w_scaled_y = [round(w*10**pow_y) for w in widths_y]

    dx = float(np.gcd.reduce(w_scaled_x)) / 10**pow_x
    dy = float(np.gcd.reduce(w_scaled_y)) / 10**pow_y

    while dx > mesh['dx'] and force_finer:
        dx /= 2

    while dy > mesh['dy'] and force_finer:
        dy /= 2

    print(f"{mesh['label']}, new resolution (dx, dy): {dx, dy}")
    mesh.update({'dx':dx})
    mesh.update({'dy':dy})



def mask_void_regions(mesh:dict) -> np.ndarray:
    """Masks a mesh's void regions so plotters ignore them.
    
    By scanning through all regions in one direction, the 'voids'
    between different regions can be located and a mask array created
    for plotters, so that meshes are plotted with clean boundaries.
    """

    mask = np.zeros((mesh['i'].size, mesh['j'].size,), bool)

    # iterate through all x slices and locate all void regions
    for j, row in enumerate(mesh['regions_x']):
        for reg in row:
            mask[:, j] = [not (reg['bounds'][0] <= i <= reg['bounds'][1]) for i in mesh['i']]

    # update the mesh's 'mask' term
    mesh.update({'mask':mask})



def update_mesh(*, mesh:dict, dt:float, curv:int, theta:float) -> np.ndarray:
    """Updates the state of a single mesh over a single timestep via the ADI method."""

    # TODO: shorten name to m, directly reference instead of making all these 
    #       pointless local variables

    i_arr = mesh['i']
    j_arr = mesh['j']
    dx = mesh['dx']
    dy = mesh['dy']
    alpha = mesh['k'] / (mesh['rho']*mesh['cp'])

    # calculate mesh coefficients
    bxx_c = alpha*dt*(1 - theta) / dx**2
    bxx_n = alpha*dt*theta / dx**2
    byy_c = 0 if curv > 1 else (alpha*(1 - theta)*dt / dy**2)
    byy_n = 0 if curv > 1 else (alpha*theta*dt / dy**2)
    bx_c = curv*alpha*(1 - theta)*dt / (2*dx)
    bx_n = curv*alpha*theta*dt / (2*dx)

    u_in = mesh['u_prev']
    u_mid = np.zeros_like(u_in, float)

    # row slices (across x)
    for j in range(j_arr.size):
        for reg in mesh['regions_x'][j]:

            s = reg['bounds'][0]
            e = reg['bounds'][1]

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



def link_to_mesh(cfg:dict, edge:dict, bc:dict) -> dict:
    """Packs all required data for heat transfer calculation into a link object.
    
    Three types of object can be linked to:
    - environment variables
    - mesh edges
    - lumped capacitor edges

    The link is a string, either split into three (lc, mesh),
    or just the name (environment)
    """

    mode = bc['mode']
    split_link = bc['link'].split('/')

    if len(split_link) == 1:
        link_obj = deepcopy(cfg['environment'][split_link[0]]) | {'type':'environment'}
        if mode == 'radiation':
            link_obj.update({'u4_mean':link_obj['temperature']**4})

    elif split_link[0] == 'meshes':
        mesh = cfg['meshes'][split_link[1]]
        link_obj = deepcopy(mesh['edges'][int(split_link[2])]) | {'type':'mesh_edge'}

        link_obj.update({'hn':mesh['dx'] if link_obj['direction'][0] == 0 else mesh['dy']})
        s, e, n = link_obj['indices']
        u = mesh['u_prev'][s:e+1, n] if link_obj['direction'][0] == 0 else\
            mesh['u_prev'][n, s:e+1].ravel()

        link_obj.update({'u':u})

        if mode == 'radiation':
            u4_mean = np.sum(link_obj['areas']*u**4) / np.sum(link_obj['areas'])
            link_obj.update({'u4_mean':u4_mean})
            link_obj.update({'emissivity':mesh['material']['emissivity']})

        elif mode == 'conduction':

            u_in = (mesh['u_prev'][s:e+1, n+sum(link_obj['direction'])] if\
                    link_obj['direction'][0] == 0 else\
                    mesh['u_prev'][n+sum(link_obj['direction']), s:e+1]).ravel()

            k = (mesh['k'][s:e+1, n] if link_obj['direction'][0] == 0 else\
                mesh['k'][n, s:e+1]).ravel()

            k_in = (mesh['k'][s:e+1, n+sum(link_obj['direction'])] if link_obj['direction'][0]\
                == 0 else mesh['k'][n+sum(link_obj['direction']), s:e+1]).ravel()

            k_bar = 0.5*(k + k_in)

            s_edge, e_edge = edge['indices'][:2]
            edge_pts = np.arange(0, e_edge - s_edge + 1) / (e_edge - s_edge)
            link_pts = np.arange(0, e - s + 1) / (e - s)

            # align values to edge nodes
            u = np.interp(edge_pts, link_pts, u)
            u_in = np.interp(edge_pts, link_pts, u_in)
            k_bar = np.interp(edge_pts, link_pts, k_bar)

            link_obj.update({'u':u, 'u_in':u_in, 'k_bar':k_bar})

    elif split_link[1] == 'lumped_capacitors':

        s_edge, e_edge = edge['indices'][:2]
        lc = cfg['lumped_capacitors'][split_link[1]]
        link_obj = deepcopy(lc['edges'][int(split_link[2])]) | {'type':'lc_edge'}
        link_obj.update({'u':lc['u_prev'] + np.zeros((e_edge - s_edge + 1), float)})

        if mode == 'radiation':
            link_obj.update({'u4_mean':lc['u_prev']**4})
            link_obj.update({'emissivity':lc['material']['emissivity']})

    else:
        raise ValueError

    return link_obj



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
                edge_link.update({'u':mesh['u_prev'][s:e+1, n]})
                edge_link.update({'u_in':mesh['u_prev'][s:e+1, n+d]})
                edge_link.update({'k_bar':0.5*(mesh['k'][s:e+1, n] + mesh['k'][s:e+1, n+d])})

            else:
                edge_link.update({'u':mesh['u_prev'][n, s:e+1]})
                edge_link.update({'u_in':mesh['u_prev'][n+d, s:e+1]})
                edge_link.update({'k_bar':0.5*(mesh['k'][n, s:e+1] + mesh['k'][n+d, s:e+1])})

            state_val = np.zeros((e - s + 1), float)
            flux = 0.0

            for bc in mesh['edge_bcs'][l]:

                boundary_condition = mesh['boundary_conditions'][bc]

                match boundary_condition['mode']:
                    case 'dirichlet':
                        state_type = 'direct'
                        state_val += boundary_condition['value']
                        du = boundary_condition['value'] - edge_link['u_in']
                        flux = edge_link['k_bar']*du / edge_link['hn']
                        break

                    case 'neumann':
                        state_type = 'gradient'
                        state_val += boundary_condition['value']
                        flux = -sum(edge_link['direction'])*edge_link['k_bar']*state_val
                        break

                    case 'conduction':
                        state_type = 'direct'
                        pair_link = link_to_mesh(cfg, edge, boundary_condition)
                        edge_state = conduction(edge_link, pair_link)
                        flux = edge_state['q']
                        state_val = edge_state['u_int']
                        break

                    case 'convection':
                        state_type = 'gradient'
                        pair_link = link_to_mesh(cfg, edge, boundary_condition)
                        q = convection(edge_link, pair_link)
                        state_val -= sum(edge['direction'])*q / edge_link['k_bar']
                        flux += q

                    case 'radiation':
                        state_type = 'gradient'
                        pair_link = link_to_mesh(cfg, edge, boundary_condition)
                        q = radiation(edge_link, pair_link)
                        state_val -= sum(edge['direction'])*q / edge_link['k_bar']
                        flux += q

            edge_states.append({'type':state_type, 'values':state_val})
            fluxes.append(np.sum(flux))
            powers.append(np.sum(flux*edge['areas']))

        mesh.update({'edge_states':edge_states})
        mesh.update({'edge_fluxes_latest':fluxes})
        mesh.update({'edge_powers_latest':powers})



def calc_areas(bounds:tuple, h:float, normal:tuple, curvature:int, depth:float=0.0) -> np.ndarray:
    """Calculates the surface area of mesh edge face elements."""

    # arrange into ascending order edge
    s = min(bounds[0], bounds[1])
    e = max(bounds[0], bounds[1])

    res = max(get_decimal_resolution(n) for n in (s, e, h))
    p = np.round(np.arange(s, e + h, h), res)
    if p[-1] > e:
        p = np.delete(p, -1)

    # planar
    if curvature == 0:
        areas = h*depth*np.r_[0.5, np.ones((p.size - 2), float), 0.5]

    # curved, horizontal
    elif normal[0] == 0:
        if s < 0:
            raise ValueError

        areas = pi*np.ones_like(p)
        half_out = p + 0.5*h

        areas[0] *= half_out[0]**2 - p[0]**2
        areas[-1] *= p[-1]**2 - half_out[-2]**2
        areas[1:-1] *= np.abs(half_out[1:-1]**2 - half_out[:-2]**2)

    # curved, vertical
    else:
        if bounds[2] < 0:
            raise ValueError

        areas = 2*pi*bounds[2]*np.r_[0.5, np.ones(p.size - 2, float), 0.5]

    # reverse area order according to parallel direction
    return areas if e > s else np.flip(areas)

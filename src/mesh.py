"""Contains functions used to manipulate meshes."""

from copy import copy
from math import copysign, pi
import numpy as np

from .util import calc_bc_relations, get_decimal_resolution, calc_face_perimeter, material_properties
from .heat_transfer import conduction, convection, radiation



def create_mesh(mesh_def:dict, force_finer:bool, material:dict):
    """Prepares a mesh dictionary for simulation.
    
    an explanation of meshes can be found here: 
    https://github.com/DeltaFwulf/flux/wiki/Meshes"""

    m = copy(mesh_def)

    x_min = min(pt[0] for line in m['lines'] for pt in line)
    y_min = min(pt[1] for line in m['lines'] for pt in line)

    m = m | grid_resolution(m['lines'], m['dx'], m['dy'], force_finer, min_elements=4)

    m.update({'i':np.arange(0, max((p[0] - x_min) / m['dx'] for l in m['lines'] for p in l) + 1, dtype=int)})
    m.update({'j':np.arange(0, max((p[1] - y_min) / m['dy'] for l in m['lines'] for p in l) + 1, dtype=int)})

    x_res = max(get_decimal_resolution(n) for n in (m['dx'], x_min))
    y_res = max(get_decimal_resolution(n) for n in (m['dy'], y_min))
    m.update({'x':np.round(x_min + m['dx']*m['i'], x_res)})
    m.update({'y':np.round(y_min + m['dy']*m['j'], y_res)})
    m.update({'regions_x':[slice_regions('x', m['x'], y, m['lines']) for y in m['y']]})
    m.update({'regions_y':[slice_regions('y', m['y'], x, m['lines']) for x in m['x']]})

    find_edges(m)
    m.update({'edge_bcs':calc_bc_relations(m['edges'], m['boundary_conditions'])})

    m.update({'u':np.zeros((m['i'].size, m['j'].size, 1), float) + mesh_def['u0']})
    m.update({'u_latest':m['u'][:, :, -1]})
    m.update({'u_prev':m['u'][:, :, -1]})
    m.update({'edge_powers':[np.zeros(1, float) for e in range(len(m['edges']))]})
    m.update({'edge_powers_latest':[np.zeros(1, float) for e in range(len(m['edges']))]})

    props = material_properties(m['u_prev'], material)
    m.update({'k':props['k'],
              'cp':props['cp'],
              'rho':props['rho'],
              'emissivity':props['emissivity']})

    vol = volume(m['regions_x'], m['x'], m['dy'], m['curvature'], m.get('depth'))
    m.update({'mass':m['rho']*vol})
    m.update({'enthalpy':np.sum(m['mass']*m['cp']*m['u_prev'])})

    m.pop('u0')

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
        bnd_a = round((trans_prev['p'] - xp[0]) / abs(xp[1] - xp[0]))
        bnd_b = round((trans['p'] - xp[0]) / abs(xp[1] - xp[0]))

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
            reg.update({'normal':-trans['sign_n']*trans['sign_p']})

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
                edge.update({'normal':(0, reg['normal'])})
                edge.update({'hp':mesh['dx']})
                edge.update({'hn':mesh['dy']})

                edge.update({'areas':edge_area(edge['bounds'],
                                                edge['hp'],
                                                edge['normal'],
                                                mesh['curvature'],
                                                mesh.get('depth'))})

                edge.update({'perimeter':calc_face_perimeter(edge['bounds'],
                                           edge['normal'],
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
                edge.update({'normal':(reg['normal'], 0)})
                edge.update({'hp':mesh['dy']})
                edge.update({'hn':mesh['dx']})

                edge.update({'areas':edge_area(edge['bounds'],
                                                edge['hp'],
                                                edge['normal'],
                                                mesh['curvature'],
                                                mesh.get('depth'))})

                edge.update({'perimeter':calc_face_perimeter(edge['bounds'],
                                           edge['normal'],
                                           mesh['curvature'],
                                           mesh.get('depth'))})

                edges.append(edge)

    # sort edges by line index
    edges = [edges[p] for p in np.argsort([edge['line_index'] for edge in edges])]
    mesh.update({'edges':edges})



def grid_resolution(lines:list, dx0:float, dy0:float, force_finer:bool=True, min_elements:int=1) -> dict:
    """Calculates a mesh resolution that tiles the boundary in both dimensions.
    
    If the calculated resolution is coarser than the original resolution and
    force_finer is True, checks whether original resolution works; if not, the
    calculated resolution is halved until it is <= original resolution.
    """

    pow_x = max(get_decimal_resolution(pt[0]) for line in lines for pt in line)
    pow_y = max(get_decimal_resolution(pt[1]) for line in lines for pt in line)

    x = np.unique(np.array([p[0] for l in lines for p in l]))
    y = np.unique(np.array([p[1] for l in lines for p in l]))

    min_width_x = np.min(np.abs(np.diff(x)))
    min_width_y = np.min(np.abs(np.diff(y)))

    if abs(x[-1] - x[0]) / dx0 == round(abs(x[-1] - x[0]) / dx0):
        dx = dx0
    elif x.size == 2:
        dx = x[1] - x[0]
    else:
        dx = float(np.gcd.reduce(((x[1:] - x[0])*10**pow_x).astype(int))) / 10**pow_x

    while dx > dx0 and force_finer or min_width_x / dx < min_elements:
        dx /= 2

    if abs(y[-1] - y[0]) / dy0 == round(abs(y[-1] - y[0]) / dy0):
        dy = dy0
    elif y.size == 2:
        dy = y[1] - y[0]
    else:
        dy = float(np.gcd.reduce(((y[1:] - y[0])*10**pow_y).astype(int))) / 10**pow_y

    while dy > dy0 and force_finer or min_width_y / dy < min_elements:
        dy /= 2

    return {'dx':dx, 'dy':dy}



def update_temp(mesh:dict, dt:float, curv:int, theta:float) -> np.ndarray:
    """Updates the state of a single mesh over a single timestep via the ADI method."""

    m = mesh
    alpha = m['k'] / (m['rho']*m['cp'])

    # NOTE: dt is halved due to ADI spanning two half steps of length dt / 2
    bx_c = curv*(1 - theta)*dt / (4*m['dx'])
    bx_n = curv*theta*dt / (4*m['dx'])
    bxx_c = dt*(1 - theta) / (2*m['dx']**2)
    bxx_n = dt*theta / (2*m['dx']**2)
    byy_c = dt*(1 - theta) / (2*m['dy']**2)
    byy_n = dt*theta / (2*m['dy']**2)

    u0 = mesh['u_prev']
    u_mid = copy(u0)

    # row slices (across x)
    for j in m['j']:
        for reg in mesh['regions_x'][j]:

            s, e = reg['bounds']

            if reg['type'] == 'edge':

                es = m['edge_states'][reg['line']]

                if es['type'] == 'direct':
                    u_mid[s:e+1, j] = es['values']
                else:
                    u_mid[s:e+1, j] = u0[s:e+1, j+reg['normal']] - m['dy']*es['values']*reg['normal']

                continue

            ts = m['edge_states'][reg['line_s']]['type']
            te = m['edge_states'][reg['line_e']]['type']
            vs = m['edge_states'][reg['line_s']]['values'][j - m['edges'][reg['line_s']]['indices'][0]]
            ve = m['edge_states'][reg['line_e']]['values'][j - m['edges'][reg['line_s']]['indices'][0]]
  
            coeffs = np.zeros((4, e + 1 - s), float)
            coeffs[0, 1:-1] = (-bxx_n + bx_n / m['x'][s+1:e])*alpha[s:e-1, j]
            coeffs[0, -1] = 0.0 if te == 'direct' else -1.0

            coeffs[1, 0] = 1.0 if ts == 'direct' else -1.0
            coeffs[1, 1:-1] = 1 + 2*bxx_n*alpha[s+1:e,j]
            coeffs[1, -1] = 1.0

            coeffs[2, 0] = 0.0 if ts == 'direct' else 1.0
            coeffs[2, 1:-1] = -(bxx_n + bx_n / m['x'][s+1:e])*alpha[s+2:e+1,j]

            coeffs[3, 0] = vs*(1.0 if ts == 'direct' else m['dx'])
            coeffs[3, 1:-1] = byy_c*alpha[s+1:e,j-1]*u0[s+1:e,j-1] +\
                              (1 - 2*byy_c*alpha[s+1:e,j])*u0[s+1:e,j] +\
                              byy_c*alpha[s+1:e,j+1]*u0[s+1:e,j+1]
            coeffs[3, -1] = ve*(1.0 if te == 'direct' else m['dx'])

            u_mid[s:e+1,j] = tdma(coeffs[0,:], coeffs[1,:], coeffs[2,:], coeffs[3,:])

    # column slices (across y)
    u_out = u0
    for i in m['i']:
        for reg in m['regions_y'][i]:

            s, e = reg['bounds']

            if reg['type'] == 'edge':
                es = m['edge_states'][reg['line']]

                if es['type'] == 'direct':
                    u_out[i, s:e+1] = es['values']
                else:
                    u_out[i, s:e+1] = u_mid[i+reg['normal'], s:e+1] - m['dx']*es['values']*reg['normal']

                continue

            ts = m['edge_states'][reg['line_s']]['type']
            te = m['edge_states'][reg['line_e']]['type']
            vs = m['edge_states'][reg['line_s']]['values'][i - m['edges'][reg['line_s']]['indices'][0]]
            ve = m['edge_states'][reg['line_e']]['values'][i - m['edges'][reg['line_s']]['indices'][0]]

            coeffs = np.zeros((4, e + 1 - s), float)

            coeffs[0, 1:-1] = -byy_n*alpha[i, s:e-1]
            coeffs[0, -1] = 0.0 if te == 'direct' else -1.0

            coeffs[1, 0] = 1.0 if ts == 'direct' else -1.0
            coeffs[1, 1:-1] = 1 + 2*byy_n*alpha[i,s+1:e]
            coeffs[1, -1] = 1.0

            coeffs[2, 0] = 0.0 if ts == 'direct' else 1.0
            coeffs[2, 1:-1] = -byy_n*alpha[i,s+2:e+1]

            coeffs[3, 0] = vs*(1 if ts == 'direct' else m['dy'])
            coeffs[3, 1:-1] = (bxx_c + bx_c / m['x'][i])*alpha[i+1,s+1:e]*u_mid[i+1,s+1:e] +\
                              (1 - 2*bxx_c*alpha[i,s+1:e])*u_mid[i,s+1:e] +\
                              (bxx_c - bx_c / m['x'][i])*alpha[i-1,s+1:e]*u_mid[i-1,s+1:e]
            coeffs[3, -1] = ve*(1 if te == 'direct' else m['dy'])

            u_out[i,s:e+1] = tdma(coeffs[0,:], coeffs[1,:], coeffs[2,:], coeffs[3,:])

    return u_out



def tdma(a, b, c, d) -> np.ndarray:
    """Solves for x given a tridiagonal matrix, following Thomas' Algorithm."""

    # NOTE: yes, I am aware of scipy.sparse.linalg.spsolve; it was slower for this
    #       implementation :)

    p = copy(d)
    q = copy(d)

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, d.size):
        dn = b[i] + a[i]*p[i - 1]
        p[i] = - c[i] / dn
        q[i] = (d[i] - a[i]*q[i - 1]) / dn

    d[-1] = q[-1]
    for i in range(d.size - 2, -1, -1):
        d[i] = p[i]*d[i + 1] + q[i]

    return d



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
        link_obj = copy(cfg['environment'][split_link[0]]) | {'type':'environment'}
        if mode == 'radiation':
            link_obj.update({'u4_mean':link_obj['temperature']**4})

    elif split_link[0] == 'meshes':
        mesh = cfg['meshes'][split_link[1]]
        link_obj = copy(mesh['edges'][int(split_link[2])]) | {'type':'mesh_edge'}

        link_obj.update({'hn':mesh['dy'] if link_obj['normal'][0] == 0 else mesh['dx']})
        s, e, n = link_obj['indices']
        u = mesh['u_prev'][s:e+1, n] if link_obj['normal'][0] == 0 else\
            mesh['u_prev'][n, s:e+1].ravel()

        link_obj.update({'u':u})

        if mode == 'radiation':
            u4_mean = np.sum(link_obj['areas']*u**4) / np.sum(link_obj['areas'])
            link_obj.update({'u4_mean':u4_mean})
            link_obj.update({'emissivity':mesh['emissivity']})

        elif mode == 'conduction':

            u_in = (mesh['u_prev'][s:e+1, n+sum(link_obj['normal'])] if\
                    link_obj['normal'][0] == 0 else\
                    mesh['u_prev'][n+sum(link_obj['normal']), s:e+1]).ravel()

            k = (mesh['k'][s:e+1, n] if link_obj['normal'][0] == 0 else\
                mesh['k'][n, s:e+1]).ravel()

            k_in = (mesh['k'][s:e+1, n+sum(link_obj['normal'])] if link_obj['normal'][0]\
                == 0 else mesh['k'][n+sum(link_obj['normal']), s:e+1]).ravel()

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
        link_obj = copy(lc['edges'][int(split_link[2])]) | {'type':'lc_edge'}
        link_obj.update({'u':lc['u_prev'] + np.zeros((e_edge - s_edge + 1), float)})

        if mode == 'radiation':
            link_obj.update({'u4_mean':lc['u_prev']**4})
            link_obj.update({'emissivity':lc['emissivity']})

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

        for l, edge in enumerate(mesh['edges']):

            s, e, n = edge['indices']
            d = sum(edge['normal'])

            edge_link = edge | {'emissivity':mesh['emissivity']}

            # get appropriate slice of temperature, conductivity, etc.
            if edge['normal'][0] == 0: # slice along x axis
                edge_link.update({'u':mesh['u_prev'][s:e+1, n]})
                edge_link.update({'u_in':mesh['u_prev'][s:e+1, n+d]})
                edge_link.update({'k_bar':0.5*(mesh['k'][s:e+1, n] + mesh['k'][s:e+1, n+d])})

            else:
                edge_link.update({'u':mesh['u_prev'][n, s:e+1]})
                edge_link.update({'u_in':mesh['u_prev'][n+d, s:e+1]})
                edge_link.update({'k_bar':0.5*(mesh['k'][n, s:e+1] + mesh['k'][n+d, s:e+1])})

            state_val = np.zeros((e - s + 1), float)

            for bc in mesh['edge_bcs'][l]:

                boundary_condition = mesh['boundary_conditions'][bc]

                match boundary_condition['mode']:
                    case 'dirichlet':
                        state_type = 'direct'
                        state_val += boundary_condition['value']
                        break

                    case 'neumann':
                        state_type = 'gradient'
                        state_val += boundary_condition['value']
                        break

                    case 'conduction':
                        state_type = 'direct'
                        pair_link = link_to_mesh(cfg, edge, boundary_condition)
                        edge_state = conduction(edge_link, pair_link)
                        state_val = edge_state['u_int']
                        break

                    case 'convection':
                        state_type = 'gradient'
                        pair_link = link_to_mesh(cfg, edge, boundary_condition)
                        q = convection(edge_link, pair_link)
                        state_val -= sum(edge['normal'])*q / edge_link['k_bar']

                    case 'radiation':
                        state_type = 'gradient'
                        pair_link = link_to_mesh(cfg, edge, boundary_condition)
                        q = radiation(edge_link, pair_link)
                        state_val -= sum(edge['normal'])*q / edge_link['k_bar']

            edge_states.append({'type':state_type, 'values':state_val})

        mesh.update({'edge_states':edge_states})



def edge_area(bounds:tuple, h:float, normal:tuple, curvature:int, depth:float=0.0) -> np.ndarray:
    """Calculates the surface area of mesh edge face elements."""

    s, e = bounds[:2]
    p = np.linspace(min(s, e), max(s, e), round(abs(s - e) / h + 1))

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

        areas = 2*pi*bounds[2]*h*np.r_[0.5, np.ones(p.size - 2, float), 0.5]

    # ensure element areas for elements in ascending order
    return areas if e > s else np.flip(areas)



def edge_gradient(u:np.ndarray, edge:dict) -> np.ndarray:
    """Calculates a 2nd order approximation to temperature gradient.
    
    Using central differencing, the temperature gradient of the mesh's
    first inboard station (one in from an edge) is estimated with 2nd
    order accuracy. This is to be used for more accurate boundary flux
    approximations.
    """

    s, e, n = edge['indices']
    n_in = n + 2*sum(edge['normal'])

    u_edge = u[s:e+1, n] if edge['normal'][0] == 0 else u[n, s:e+1]
    u_in = u[s:e+1, n_in] if edge['normal'][0] == 0 else u[n_in, s:e+1]

    return sum(edge['normal'])*(u_in - u_edge) / (2*edge['hn'])



def edge_power(u:np.ndarray, k:np.ndarray, edge:dict, curvature:int) -> float:
    """Estimates the thermal power flowing through an edge.

    Positive values represent flows into the mesh, negative
    flows out, regardless of edge direction."""

    s, e, n = edge['indices']
    d = sum(edge['normal'])

    bn = edge['bounds'][2] + d*edge['hn']
    sf = (bn / edge['bounds'][2]) if (edge['normal'][1] == 0 and curvature == 1) else 1
    areas = sf*edge['areas']
    fluxes = edge_gradient(u, edge)*(k[s:e+1, n+d] if edge['normal'][0] == 0 else k[n+d, s:e+1])

    return -d*np.sum(fluxes*areas)



def volume(regions_x:list[dict], x:np.ndarray, dy:float, curvature:int, depth:float=0.0) -> np.ndarray:
    """Calculates the volume around each mesh node."""

    vol = np.zeros((x.size, len(regions_x)), float)
    width = {'internal':1.0, 'edge':0.5, 'void':0.0}
    dx = x[1] - x[0]

    # iterate through all x-slices
    for j, row in enumerate(regions_x):
        for n, reg in enumerate(row):

            reg_last = row[n-1]['type'] if n > 0 and row[n-1]['bounds'][1] == reg['bounds'][0] else 'void'
            reg_next = row[n+1]['type'] if n + 1 < len(row) and row[n+1]['bounds'][0] == reg['bounds'][1] else 'void'
            n_pts = reg['bounds'][1] - reg['bounds'][0] + 1

            # stations 1 to N
            dy_l = dy*np.ones(n_pts - 1, float)*width[reg['type']]
            dy_r = dy*np.r_[np.ones(n_pts - 2, float)*width[reg['type']], width[reg_next]]
            dx_l = 0.5*dx*np.ones(n_pts - 1, float)
            dx_r = 0.5*dx*np.r_[np.ones(n_pts - 2), 0. if reg_next == 'void' else 1.0]

            a, b = reg['bounds'][0] + (1 if reg_last != 'void' else 0), reg['bounds'][1]

            if reg_last == 'void':
                dy_l = np.r_[dy*width[reg['type']], dy_l]
                dy_r = np.r_[dy*width[reg['type']], dy_r]
                dx_l = np.r_[0., dx_l]
                dx_r = np.r_[0.5*dx, dx_r]

            # obtain volumes of all stations, add to vol array
            if curvature == 0:
                vol[a:b + 1, j] = depth*(dy_l*dx_l + dy_r*dx_r)

            else:
                vl = dy_l*(x[a:b + 1]**2 - (x[a:b + 1] - dx_l)**2)
                vr = dy_r*((x[a:b + 1] + dx_r)**2 - x[a:b + 1]**2)
                vol[a:b + 1, j] = pi*(vl + vr)

    return vol

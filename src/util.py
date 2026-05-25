"""Mesh utility functions."""

from math import copysign, pi
import numpy as np

from heat_transfer import conduction, convection, radiation



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



def bound_gradients(config:dict, mesh_name:str) -> float:
    """Calculates the edge temperature gradients for all edges in the named mesh, 
       according to its boundary conditions."""

    mesh = config['meshes'].get(mesh_name)
    edges = mesh['edges']
    for edge in edges: # process thermal data for use

        s, e, n = edge['indices']
        d = sum(edge['direction'])

        # get appropriate slice of temperature, conductivity, etc.
        if edge['direction'][0] == 0: # slice along x axis
            edge.update({'u':mesh['u_last'][s:e+1, n]})
            edge.update({'u_in':mesh['u_last'][s:e+1, n+d]})
            edge.update({'k_bar':0.5*(mesh['k'][s:e+1, n] + mesh['k'][s:e+1, n+d])})

        else:
            edge.update({'u':mesh['u_last'][n, s:e+1]})
            edge.update({'u_in':mesh['u_last'][n+d, s:e+1]})
            edge.update({'k_bar':0.5*(mesh['k'][n, s:e+1] + mesh['k'][n+d, s:e+1])})

    gradients = []

    for edge in edges:
        s, e, n = edge['indices']
        g = np.zeros((e - s + 1), float)
        bc = mesh['bc'][edge['line_index']]

        if bc['mode'] == 'dirichlet':
            gradients.append(g)
            continue

        if bc['mode'] == 'neumann':
            gradients.append(g + bc['value'])
            continue

        link_objects = get_link_objects(config, edge, bc)
        for lo in link_objects:
            g += calc_gradient(edge, lo, bc['mode'])

        gradients.append(g)

    mesh.update({'gradients':gradients})



def get_link_objects(cfg:dict, edge:dict, bc:dict) -> dict:
    """
    Returns all link objects specified by the boundary condition
    in a list.

    Link objects either represent other edges (in same or diff mesh),
    or 'ambients' which are used for i.e. convection to atmosphere, etc.
    """

    link_objects = []
    mode = bc['mode']
    link_names = bc['link']

    for name in link_names:

        if cfg['environment'].get(name) is None and '/' not in name:
            raise ValueError
        if '/' not in name:
            link_obj = cfg['link_objects'].get(name)
            link_objects.append(link_obj)
            continue

        mesh = cfg['meshes'][name.split('/')[0]]
        link_obj = mesh['edges'][int(name.split('/')[1])]
        s, e, n = link_obj['indices']
        link_obj.update({'type':'edge'})
        link_obj.update({'h_norm':mesh['dx'] if link_obj['direction'][0] == 0 else mesh['dy']})

        u = (mesh['u_last'][s:e+1, n] if link_obj['direction'][0] == 0 else\
             mesh['u_last'][n, s:e+1]).ravel()

        if mode == 'radiation':
            u4_mean = np.sum(link_obj['areas']*u**4) / np.sum(link_obj['areas'])
            link_obj.update({'u4_mean':u4_mean})
            link_obj.update({'emissivity':mesh['material']['emissivity']})

        elif mode == 'conduction':

            u_in = (mesh['u_last'][s:e+1, n+sum(link_obj['direction'])] if\
                    link_obj['direction'][0] == 0 else\
                    mesh['u_last'][n+sum(link_obj['direction']), s:e+1]).ravel()

            k = (mesh['k'][s:e+1, n] if link_obj['direction'][0] == 0 else\
                mesh['k'][n, s:e+1]).ravel()

            k_in = (mesh['k'][s:e+1, n+sum(link_obj['direction'])] if link_obj['direction'][0]\
                == 0 else mesh['k'][n+sum(link_obj['direction']), s:e+1]).ravel()

            k_bar = 0.5*(k + k_in)

            s_edge, e_edge = edge['indices'][:2]
            edge_pts = np.arange(0, e_edge - s_edge + 1) / (e_edge - s_edge)
            link_pts = np.arange(0, e - s + 1) / (e - s)

            # interpolate for new array of u, u_in, k_bar
            u = np.array([np.interp(p, link_pts, u) for p in edge_pts], float)
            u_in = np.array([np.interp(p, link_pts, u_in) for p in edge_pts], float)
            k_bar = np.array([np.interp(p, link_pts, k_bar) for p in edge_pts], float)

            link_obj.update({'u':u, 'u_in':u_in, 'k_bar':k_bar})

        link_objects.append(link_obj)

    return link_objects



# XXX: remove function? Put into gradient function?
def calc_gradient(edge:dict, link:dict, mode:str) -> np.ndarray:
    """Calculates the edge temperature gradient given boundary condition."""

    if mode == 'conduction':
        q = conduction(edge, link)
    elif mode == 'radiation':
        q = radiation(edge, link)
    else:
        q = convection(edge, link, mode)

    return -sum(edge['direction'])*q / edge['k_bar']

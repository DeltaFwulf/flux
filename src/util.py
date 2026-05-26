"""Mesh utility functions."""

import numpy as np

from heat_transfer import conduction, convection, radiation

# FIXME: bound_gradients does not currently support multiple transfer modes, only multiple links



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
        bc = mesh['bc'][edge['line_index']]

        match bc['mode']:
            case 'dirichlet':
                g = np.zeros((e - s + 1), float)
            case 'neumann':
                g = np.zeros((e - s + 1), float) + bc['value']
            case _:
                g = sum(calc_gradient(edge, lo, bc['mode']) for lo in\
                        get_link_objects(config, edge, bc))

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



def update_properties(mesh:dict) -> None:
    """Updates the mesh's material properties (k, cp, rho) given temperature."""

    u = mesh['u_last']
    mat = mesh['material']

    k = np.interp(x=u, xp=mat['u'], fp=mat['k'])
    cp = np.interp(x=u, xp=mat['u'], fp=mat['cp'])
    rho = np.interp(x=u, xp=mat['u'], fp=mat['rho'])
    alpha = k / (rho*cp)

    mesh.update({'k':k, 'cp':cp, 'rho':rho, 'diffusivity':alpha})

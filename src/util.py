"""Mesh utility functions."""

from copy import deepcopy
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



def calc_edge_states(cfg:dict) -> None:
    """Updates all mesh edge state arrays in the simulation."""

    for mesh in cfg['meshes'].values():

        # edges now store both a value array and gradient array. One must be set to None
        edge_states = []

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
                        pair_link = get_link_data(cfg, edge, boundary_condition)
                        state_val = conduction(edge_link, pair_link)
                        break

                    case 'convection':
                        state_type = 'gradient'
                        pair_link = get_link_data(cfg, edge, boundary_condition)
                        q = convection(edge_link, pair_link, boundary_condition['mode'])
                        state_val -= sum(edge['direction'])*q / edge_link['k_bar']

                    case 'radiation':
                        state_type = 'gradient'
                        pair_link = get_link_data(cfg, edge, boundary_condition)
                        q = radiation(edge_link, pair_link)
                        state_val -= sum(edge['direction'])*q / edge_link['k_bar']

            edge_states.append({'type':state_type, 'values':state_val})

        mesh.update({'edge_states':edge_states})



def get_link_data(cfg:dict, edge:dict, bc:dict) -> dict:
    """ Packs all required data for heat transfer calculation into a link object."""

    name = bc['link']
    mode = bc['mode']

    if cfg['environment'].get(name) is None and '/' not in name:
        raise ValueError

    if '/' not in name:
        return cfg['environment'].get(name)

    mesh = cfg['meshes'][name.split('/')[0]]
    link_obj = deepcopy(mesh['edges'][int(name.split('/')[1])]) | {'type':'edge'}
    link_obj.update({'h_norm':mesh['dx'] if link_obj['direction'][0] == 0 else mesh['dy']})
    s, e, n = link_obj['indices']
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
        u = np.interp(edge_pts, link_pts, u)
        u_in = np.interp(edge_pts, link_pts, u_in)
        k_bar = np.interp(edge_pts, link_pts, k_bar)

        link_obj.update({'u':u, 'u_in':u_in, 'k_bar':k_bar})

    else: # TODO: add convection setup
        pass

    return link_obj



def update_properties(mesh:dict) -> None:
    """Updates the mesh's material properties (k, cp, rho) given temperature."""

    u = mesh['u_last']
    mat = mesh['material']

    k = np.interp(x=u, xp=mat['u'], fp=mat['k'])
    cp = np.interp(x=u, xp=mat['u'], fp=mat['cp'])
    rho = np.interp(x=u, xp=mat['u'], fp=mat['rho'])
    alpha = k / (rho*cp)

    mesh.update({'k':k, 'cp':cp, 'rho':rho, 'diffusivity':alpha})

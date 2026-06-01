"""Mesh utility functions."""

from os import getcwd
from os.path import join
from copy import deepcopy
import numpy as np
import yaml

from heat_transfer import conduction, convection, radiation



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
                        q = convection(edge_link, pair_link)
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

    if '/' not in name and mode == 'radiation':
        link_obj = deepcopy(cfg['environment'].get(name))
        link_obj.update({'u4_mean':link_obj['temperature']**4})
        return link_obj

    if '/' not in name:
        return cfg['environment'].get(name)

    mesh = cfg['meshes'][name.split('/')[0]]
    link_obj = deepcopy(mesh['edges'][int(name.split('/')[1])]) | {'type':'edge'}
    link_obj.update({'hn':mesh['dx'] if link_obj['direction'][0] == 0 else mesh['dy']})
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

        # align values to edge nodes
        u = np.interp(edge_pts, link_pts, u)
        u_in = np.interp(edge_pts, link_pts, u_in)
        k_bar = np.interp(edge_pts, link_pts, k_bar)

        link_obj.update({'u':u, 'u_in':u_in, 'k_bar':k_bar})

    return link_obj



def get_material(name:str) -> dict:
    """Gets material properties from 'materials.yaml'."""

    with open(join(getcwd(), 'src', 'data', 'materials.yaml'), encoding='utf-8') as f:
        materials = yaml.load(stream=f, Loader=yaml.SafeLoader)

    mat = materials.get(name)
    if mat is None:
        print(f"No material found with name {name}.")
        raise ValueError

    return mat



def update_properties(solid:dict) -> None:
    """Updates a solid object's material properties (k, cp, rho) given a temperature."""

    u = solid['u_last']
    mat = solid['material']

    k = np.interp(x=u, xp=mat['u'], fp=mat['k'])
    cp = np.interp(x=u, xp=mat['u'], fp=mat['cp'])
    rho = np.interp(x=u, xp=mat['u'], fp=mat['rho'])
    alpha = k / (rho*cp)

    solid.update({'k':k, 'cp':cp, 'rho':rho, 'diffusivity':alpha})



def calc_bc_relations(solid:dict):
    """Returns a list of boundary condition indices relevant to each edge in a mesh."""

    edge_bcs = []
    for l in range(len(solid['edges'])):

        # iterate through all boundary conditions, add relevant entries to list
        relevant = [i for i, bc in enumerate(solid['boundary_conditions']) if bc['edge'] == l]
        edge_bcs.append(relevant)

    solid.update({'edge_bcs':edge_bcs})

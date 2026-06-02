"""Common utility functions."""

from os import getcwd
from os.path import join
from copy import deepcopy
import numpy as np
import yaml



def get_link_data(cfg:dict, edge:dict, bc:dict) -> dict:
    """ Packs all required data for heat transfer calculation into a link object."""

    name = bc['link']
    mode = bc['mode']

    if (cfg.get('environment') is None or cfg['environment'].get(name) is None) and '/' not in name:
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
